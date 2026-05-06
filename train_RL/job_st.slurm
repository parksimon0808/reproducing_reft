#!/bin/bash
#SBATCH --job-name=reft-rl-st
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --gres=gpu:1
#SBATCH --constraint gpu80
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=begin,end,fail

set -x

module purge
module load anaconda3/2024.6
conda activate reft-train-rl

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export WANDB_MODE="offline"

export TOKENIZERS_PARALLELISM=true
mkdir -p logs

PROJECT_ROOT="$PROJECT_DIR/reproducing-reft"

MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B-Base}
MODEL_PATH="$PROJECT_DIR/models/${MODEL_NAME}"

DATASET_NAME=${DATASET_NAME:-gsm8k}
BATCH_SIZE=${BATCH_SIZE:-4}

ST_MODE=${ST_MODE:-online}
METHOD_LABEL=${METHOD_LABEL:-${ST_MODE}_st}

NEG_REWARD=${NEG_REWARD:--1.0}
POS_REWARD=${POS_REWARD:-1.0}

REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score_reinforce}

NUM_GPUS="$(( $(echo $SLURM_JOB_GPUS | grep -o , | wc -l) + 1 ))"

run_experiment() {
    local dataset="$1"
    local max_prompt_length="$2"
    local max_new_tokens="$3"

    local total_batch_size=32
    local total_epochs=10
    local total_steps=$(( save_freq * total_epochs ))

    local steps_per_generation=1
    if [ "${ST_MODE}" == "offline" ]; then
        steps_per_generation=${save_freq}
    fi

    local exp_name="${MODEL_NAME}_${dataset}_${METHOD_LABEL}"
    local output_dir="${PROJECT_ROOT}/train_RL/outputs/${exp_name}"

    mkdir -p "${output_dir}"

    echo ""
    echo "=========================================="
    echo " Starting: ${exp_name}  (GPUs=${NUM_GPUS})"
    echo " mode=${ST_MODE} | neg_reward=${NEG_REWARD} | reward_fn=${REWARD_FN_NAME}"
    echo "=========================================="

    python train_self_training.py \
        --model_name ${MODEL_PATH} \
        --dataset ${dataset} \
        --batch_size ${BATCH_SIZE} \
        --num_generations 8 \
        --steps_per_generation ${steps_per_generation} \
        --num_steps ${total_steps} \
        --pos_reward ${POS_REWARD} \
        --neg_reward ${NEG_REWARD} \
        --lr 1e-6 \
        --max_prompt_length ${max_prompt_length} \
        --max_new_tokens ${max_new_tokens} \
        --eval_every 50 \
        --save_freq ${save_freq} \
        --output_dir ${output_dir} \
        --gradient_checkpointing \
        1> >(tee "${output_dir}/${exp_name}.log") \
        2> >(tee "${output_dir}/${exp_name}.err" >&2)

    echo " Done: ${exp_name}"
}

source "${PROJECT_ROOT}/train_RL/experiments/${DATASET_NAME}.sh"
