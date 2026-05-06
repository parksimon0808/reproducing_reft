#!/bin/bash
#SBATCH --job-name=reft-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --gres=gpu:2
#SBATCH --constraint gpu80
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=begin,end,fail

set -x

module purge
module load anaconda3/2024.6
conda activate reft-sft

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export WANDB_MODE="offline"

export TOKENIZERS_PARALLELISM=true
mkdir -p logs

MODEL_NAME=${MODEL_NAME:-Qwen3.5-0.8B-Base}

DATASET_NAME=${DATASET_NAME:-gsm8k}

NUM_GPUS="$(( $(echo $SLURM_JOB_GPUS | grep -o , | wc -l) + 1 ))"

run_experiment() {
    local dataset="$1"
    local n_epochs="$2"
    local learning_rate="$3"
    local max_input_length="$4"
    local max_new_tokens="$5"
    local batch_size_per_gpu="$6"

    local total_batch_size=48
    local gradient_accumulation_steps=$((total_batch_size / batch_size_per_gpu / NUM_GPUS))

    local exp_name="${MODEL_NAME}_${dataset}"
    local train_file="../generate_data/outputs/${dataset}_og.jsonl"
    local test_file="../generate_data/outputs/${dataset}_test.jsonl"
    local model_dir="outputs/${exp_name}"

    mkdir -p "${model_dir}"

    echo ""
    echo "=========================================="
    echo " Starting: ${exp_name}  (GPUs=${NUM_GPUS})"
    echo "=========================================="

    accelerate launch \
        --num_processes=${NUM_GPUS} \
        --num_machines=1 \
        --mixed_precision=bf16 \
        --dynamo_backend=no \
        train.py \
            --model_name_or_path        ${MODEL_PATH} \
            --tokenizer_name_or_path    ${MODEL_PATH} \
            --dataset                   "${dataset}" \
            --train_file                "${train_file}" \
            --test_file                 "${test_file}" \
            --model_dir                 "${model_dir}" \
            --n_epochs                  "${n_epochs}" \
            --learning_rate             "${learning_rate}" \
            --max_input_length          "${max_input_length}" \
            --max_new_tokens            "${max_new_tokens}" \
            --batch_size                "${batch_size_per_gpu}" \
            --eval_batch_size           "8" \
            --gradient_accumulation_steps "${gradient_accumulation_steps}" \
            --num_workers               "8" \
            --weight_decay              "0" \
            --warmup_step               "-100" \
            --clip_grad_norm            "1" \
            --seed                      "42" \
            --keep_num_ckpt             "100" \
            --logging_epoch_freq        "1" \
            --evaluating_epoch_freq     "1" \
            --saving_epoch_freq         "1" \
            --logging_step_freq         "100" \
            --evaluating_step_freq      "-100" \
            --saving_step_freq          "-100" \
            --wandb_log                 "True" \
            --wandb_project             "ReFT-SFT" \
            --wandb_run_name            "${exp_name}" \
        1> >(tee "${model_dir}/${exp_name}.log") \
        2> >(tee "${model_dir}/${exp_name}.err" >&2)

    echo " Done: ${exp_name}"
}

source experiments/${DATASET_NAME}.sh
