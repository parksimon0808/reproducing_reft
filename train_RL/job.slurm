#!/bin/bash
#SBATCH --job-name=reft-rl
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

MODEL_NAME=${MODEL_NAME:-Qwen3.5-0.8B-Base}
MODEL_PATH="$PROJECT_DIR/models/${MODEL_NAME}"

DATASET_NAME=${DATASET_NAME:-gsm8k}
METHOD=${METHOD:-ppo}
BATCH_SIZE=${BATCH_SIZE:-4}

NUM_GPUS="$(( $(echo $SLURM_JOB_GPUS | grep -o , | wc -l) + 1 ))"

run_experiment() {
    local dataset="$1"
    local max_prompt_length="$2"
    local max_new_tokens="$3"

    local total_batch_size=32
    local gradient_accumulation_steps=$((total_batch_size / BATCH_SIZE / NUM_GPUS))

    local exp_name="${MODEL_NAME}_${dataset}_${METHOD}"
    local output_dir="outputs/${exp_name}"

    mkdir -p "${output_dir}"

    echo ""
    echo "=========================================="
    echo " Starting: ${exp_name}  (GPUs=${NUM_GPUS})"
    echo "=========================================="

    local CMD="train.py \
        --method ${METHOD} \
        --model_name ${MODEL_PATH} \
        --dataset ${dataset} \
        --batch_size ${BATCH_SIZE} \
        --mini_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --num_generations 8 \
        --lr 1e-6 \
        --kl_coef 0.05 \
        --max_prompt_length ${max_prompt_length} \
        --max_new_tokens ${max_new_tokens} \
        --eval_every 10 \
        --save_every 10 \
        --save_freq ${save_freq} \
        --output_dir ${output_dir} \
        --gradient_checkpointing"

    if [ "${METHOD}" = "self_training" ]; then
        python ${CMD} \
            --steps_per_generation 1 \
            --pos_reward 1.0 \
            --neg_reward -1.0 \
        1> >(tee "${output_dir}/${exp_name}.log") \
        2> >(tee "${output_dir}/${exp_name}.err" >&2)
    else
        if [ ${NUM_GPUS} -gt 1 ]; then
            accelerate launch \
                --multi_gpu \
                --num_processes ${NUM_GPUS} \
                ${CMD} \
            1> >(tee "${output_dir}/${exp_name}.log") \
            2> >(tee "${output_dir}/${exp_name}.err" >&2)
        else
            accelerate launch \
                --num_processes 1 \
                ${CMD} \
            1> >(tee "${output_dir}/${exp_name}.log") \
            2> >(tee "${output_dir}/${exp_name}.err" >&2)
        fi
    fi

    echo " Done: ${exp_name}"
}

source experiments/${DATASET_NAME}.sh
