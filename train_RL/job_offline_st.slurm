#!/bin/bash
#SBATCH --job-name=reft-offline-st-verl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --gres=gpu:1
#SBATCH --constraint gpu80
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=begin,end,fail

set -x

module purge
module load anaconda3/2024.6
conda activate reft-train-rl-vllm

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

export WANDB_MODE="offline"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=true

PROJECT_ROOT="$PROJECT_DIR/reproducing-reft"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

mkdir -p logs

MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B-Base}
MODEL_PATH="$PROJECT_DIR/models/${MODEL_NAME}"

DATASET_NAME=${DATASET_NAME:-gsm8k}

METHOD_LABEL=${METHOD_LABEL:-offline_st}

NUM_GPUS="$(( $(echo $SLURM_JOB_GPUS | grep -o , | wc -l) + 1 ))"

train_batch_size=32
total_rounds=10
lr=1e-6

NEG_REWARD=${NEG_REWARD:--1.0}

CHAT_TEMPLATE='{{ messages[0]["content"] }}'

DATA_DIR="${PROJECT_ROOT}/train_RL/data/${DATASET_NAME}"
SFT_DATASET_PATH="${PROJECT_ROOT}/train_RL/sft_dataset_weighted.py"

run_experiment() {
    local dataset="$1"
    local max_prompt_length="$2"
    local max_new_tokens="$3"

    local exp_name="${MODEL_NAME}_${dataset}_${METHOD_LABEL}"
    local output_dir="${PROJECT_ROOT}/train_RL/outputs/${exp_name}"
    mkdir -p "${output_dir}" "${DATA_DIR}"

    local train_file="${DATA_DIR}/train.parquet"
    if [ ! -f "${train_file}" ]; then
        echo "Generating source parquet for ${dataset}..."
        python "${PROJECT_ROOT}/train_RL/generate_parquet.py" \
            --dataset "${dataset}" \
            --output_dir "${DATA_DIR}"
    fi

    echo ""
    echo "=========================================="
    echo " Starting: ${exp_name}"
    echo " NUM_GPUS=${NUM_GPUS} | dataset=${dataset} | rounds=${total_rounds}"
    echo " neg_reward=${NEG_REWARD}"
    echo " max_prompt_length=${max_prompt_length} max_new_tokens=${max_new_tokens}"
    echo "=========================================="

    local checkpoint="${MODEL_PATH}"

    for round in $(seq 1 ${total_rounds}); do
        local _rollout_suffix=""
        if [ "${METHOD_LABEL}" != "offline_st" ]; then
            _rollout_suffix="_${METHOD_LABEL#offline_st_}"
        fi
        local rollout_file="${DATA_DIR}/${MODEL_NAME}${_rollout_suffix}_round${round}.parquet"
        local round_dir="${output_dir}/round${round}"
        local cumulative_step=$(( save_freq * round ))
        local ckpt_dir="${output_dir}/global_step_${cumulative_step}"
        local round_shard_src="${round_dir}/global_step_${save_freq}"

        echo ""
        echo "── round ${round}/${total_rounds} ─────────────────────────────"
        echo " checkpoint = ${checkpoint}"
        echo " rollouts   = ${rollout_file}"
        echo " ckpt_dir   = ${ckpt_dir}"
        echo "──────────────────────────────────────────"

        if [ -f "${rollout_file}" ]; then
            echo "[round ${round}] rollout file already exists — skipping generation"
        else
            python "${PROJECT_ROOT}/train_RL/generate_rollouts.py" \
                --input_parquet  "${train_file}" \
                --output_parquet "${rollout_file}" \
                --model_path     "${checkpoint}" \
                --dataset        "${dataset}" \
                --max_prompt_length "${max_prompt_length}" \
                --max_new_tokens    "${max_new_tokens}" \
                --temperature 1.0 \
                --neg_reward "${NEG_REWARD}" \
                1> >(tee "${round_dir}_gen.log") \
                2>&1
        fi

        if [ -d "${ckpt_dir}" ] || [ -d "${round_shard_src}" ]; then
            echo "[round ${round}] SFT artifacts already exist — skipping SFT"
        else
            torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} \
                -m verl.trainer.sft_trainer \
                data.train_files="${rollout_file}" \
                data.val_files=null \
                data.train_batch_size=$(( train_batch_size * 8 )) \
                data.micro_batch_size_per_gpu=$(( train_batch_size / NUM_GPUS )) \
                data.messages_key=messages \
                data.pad_mode=no_padding \
                data.max_length=$(( max_prompt_length + max_new_tokens )) \
                data.truncation=right \
                data.custom_cls.path="${SFT_DATASET_PATH}" \
                data.custom_cls.name=WeightedMultiTurnSFTDataset \
                data.ignore_input_ids_mismatch=True \
                model.path="${checkpoint}" \
                model.trust_remote_code=True \
                model.custom_chat_template="'${CHAT_TEMPLATE}'" \
                model.enable_gradient_checkpointing=True \
                engine=fsdp \
                engine.model_dtype=bfloat16 \
                engine.param_offload=True \
                engine.optimizer_offload=True \
                optim.lr=${lr} \
                optim.lr_scheduler_type=constant \
                optim.lr_warmup_steps=0 \
                optim.lr_warmup_steps_ratio=0.0 \
                optim.min_lr_ratio=1.0 \
                trainer.total_epochs=1 \
                trainer.default_local_dir="${round_dir}" \
                trainer.project_name=reproducing-reft \
                trainer.experiment_name="${exp_name}_round${round}" \
                trainer.logger='["console","wandb"]' \
                trainer.save_freq=${save_freq} \
                trainer.test_freq=-1 \
                trainer.n_gpus_per_node=${NUM_GPUS} \
                trainer.nnodes=1 \
                1> >(tee "${round_dir}_sft.log") \
                2>&1
        fi

        if [ -d "${round_shard_src}" ] && [ ! -d "${ckpt_dir}" ]; then
            mv "${round_shard_src}" "${ckpt_dir}"
        fi
        if [ -d "${round_dir}" ]; then
            rm -rf "${round_dir}"
        fi
        if [ ! -d "${ckpt_dir}" ]; then
            echo "ERROR: round ${round} produced no checkpoint at ${ckpt_dir}"
            exit 1
        fi

        if compgen -G "${ckpt_dir}/model_world_size_*_rank_*.pt" > /dev/null; then
            echo "Merging verl SFT shards in ${ckpt_dir}"
            if python -m verl.model_merger merge \
                    --backend fsdp \
                    --local_dir "${ckpt_dir}" \
                    --target_dir "${ckpt_dir}"; then
                rm -f "${ckpt_dir}"/model_world_size_*_rank_*.pt
                rm -f "${ckpt_dir}"/optim_world_size_*_rank_*.pt
                rm -f "${ckpt_dir}"/extra_state_world_size_*_rank_*.pt
            else
                echo "ERROR: verl merge failed for ${ckpt_dir}; aborting"
                exit 1
            fi
        else
            echo "[round ${round}] ${ckpt_dir} already merged — skipping merger"
        fi

        local next_input="${ckpt_dir}"
        if [[ "${MODEL_NAME}" == Qwen3.5* ]]; then
            local merged="${ckpt_dir}_merged"
            if [ -d "${merged}" ]; then
                echo "[round ${round}] ${merged} already exists — skipping Qwen3.5 fix-up"
            else
                echo "Fixing Qwen3.5 layout ${ckpt_dir} -> ${merged}"
                python "${PROJECT_ROOT}/merge_sft_checkpoint.py" \
                    --sft-checkpoint "${ckpt_dir}" \
                    --original-model "${MODEL_PATH}" \
                    --output-dir "${merged}"
            fi
            next_input="${merged}"
        fi

        checkpoint="${next_input}"
    done

    echo ""
    echo " Done: ${exp_name}  (final checkpoint: ${checkpoint})"
}

source "${PROJECT_ROOT}/train_RL/experiments/${DATASET_NAME}.sh"
