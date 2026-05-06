#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --constraint gpu80
#SBATCH -N 1 -n 1
#SBATCH --mem-per-gpu=120G
#SBATCH --cpus-per-gpu 12
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --partition pli-c
#SBATCH --mail-type=ALL
#SBATCH --job-name=rl
#SBATCH -t 00:15:00
#SBATCH --account pli

module load anaconda3/2024.6
conda activate reft-train-rl

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

export WANDB_MODE="offline"

METHOD="ppo"
MODEL_NAME=Qwen3.5-0.8B-Base
MODEL="$PROJECT_DIR/models/${MODEL_NAME}"
SFT_CKPT=""
DATASET="gsm8k"
NUM_GPUS="$(( $(echo $SLURM_JOB_GPUS| grep -o , | wc -l) + 1 ))"

timestamp=$(date +'%Y%m%d-%H%M%S')
OUTPUT_DIR="outputs/${MODEL_NAME}_${DATASET}_${METHOD}"

CMD="train.py \
    --method ${METHOD} \
    --model_name ${MODEL} \
    --dataset ${DATASET} \
    --batch_size 8 \
    --mini_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_generations 8 \
    --lr 1e-6 \
    --kl_coef 0.05 \
    --max_prompt_length 512 \
    --max_new_tokens 1024 \
    --eval_every 10 \
    --save_every 10 \
    --output_dir ${OUTPUT_DIR} \
    --gradient_checkpointing"

if [ -n "${SFT_CKPT}" ]; then
    CMD="${CMD} --sft_checkpoint ${SFT_CKPT}"
fi

if [ "${METHOD}" = "self_training" ]; then
    python ${CMD} \
        --steps_per_generation 1 \
        --pos_reward 1.0 \
        --neg_reward -1.0
else
    if [ ${NUM_GPUS} -gt 1 ]; then
        accelerate launch \
            --multi_gpu \
            --num_processes ${NUM_GPUS} \
            ${CMD}
    else
        accelerate launch \
            --num_processes 1 \
            ${CMD}
    fi
fi
