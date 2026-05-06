#!/bin/bash
#SBATCH --job-name=reft-rl-ablations
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

: "${ABLATION_TAG:?set ABLATION_TAG (e.g. kl_0, lr_3e-7, K_16, clip_0.2_0.26)}"

MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B-Base}
DATASET_NAME=${DATASET_NAME:-gsm8k}
BATCH_SIZE=${BATCH_SIZE:-4}

if [ "${DATASET_NAME}" == "gsm8k" ]; then
    STEP_IDX=2412
elif [ "${DATASET_NAME}" == "svamp" ]; then
    STEP_IDX=226
elif [ "${DATASET_NAME}" == "math" ]; then
    STEP_IDX=3658
fi

MODEL_PATH="${PROJECT_ROOT}/sft/outputs/${MODEL_NAME}_${DATASET_NAME}/step_${STEP_IDX}_epoch_2"

if [[ "${MODEL_NAME}" == Qwen3.5* ]]; then
    MERGED_PATH="${MODEL_PATH}_merged"
    if [ ! -d "${MERGED_PATH}" ]; then
        echo "Merging SFT checkpoint for ${MODEL_NAME} -> ${MERGED_PATH}"
        python "${PROJECT_ROOT}/merge_sft_checkpoint.py" \
            --sft-checkpoint "${MODEL_PATH}" \
            --original-model "/scratch/gpfs/ARORA/juhyunp/models/${MODEL_NAME}" \
            --output-dir "${MERGED_PATH}"
    else
        echo "Merged checkpoint already exists at ${MERGED_PATH}, skipping merge"
    fi
    MODEL_PATH="${MERGED_PATH}"
fi

NUM_GPUS="$(( $(echo $SLURM_JOB_GPUS | grep -o , | wc -l) + 1 ))"

n_rollouts=${N_ROLLOUTS:-8}
lr=${LR:-1e-6}
kl_coef=${KL_COEF:-0.05}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.2}

run_experiment() {
    local dataset="$1"
    local max_prompt_length="$2"
    local max_new_tokens="$3"

    local total_batch_size=32
    local gradient_accumulation_steps=$((total_batch_size / BATCH_SIZE / NUM_GPUS))

    local exp_name="${MODEL_NAME}_${dataset}_grpo_ablation_${ABLATION_TAG}"
    local output_dir="${PROJECT_ROOT}/train_RL/outputs/${exp_name}"

    mkdir -p "${output_dir}"

    echo ""
    echo "=========================================="
    echo " Starting: ${exp_name}  (GPUs=${NUM_GPUS})"
    echo " kl=${kl_coef} lr=${lr} K=${n_rollouts} clip=${clip_ratio_low}/${clip_ratio_high}"
    echo "=========================================="

    local CMD="train.py \
        --method grpo \
        --model_name ${MODEL_PATH} \
        --dataset ${dataset} \
        --batch_size ${BATCH_SIZE} \
        --mini_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --num_generations ${n_rollouts} \
        --lr ${lr} \
        --kl_coef ${kl_coef} \
        --max_prompt_length ${max_prompt_length} \
        --max_new_tokens ${max_new_tokens} \
        --eval_every 10 \
        --save_every 10 \
        --save_freq ${save_freq} \
        --output_dir ${output_dir} \
        --gradient_checkpointing"

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

    echo " Done: ${exp_name}"
}

source "${PROJECT_ROOT}/train_RL/experiments/${DATASET_NAME}.sh"
