module load anaconda3/2024.6
conda activate reft-train-rl-vllm

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

export WANDB_MODE="offline"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

for DATASET_NAME in gsm8k svamp; do

for postfix in grpo grpo_warmup; do 

for MODEL in Qwen3-1.7B-Base Qwen3.5-0.8B-Base Qwen3.5-2B-Base; do

if [ "${DATASET_NAME}" == "gsm8k" ]; then
    STEP_IDX=226
elif [ "${DATASET_NAME}" == "svamp" ]; then
    STEP_IDX=21
elif [ "${DATASET_NAME}" == "math" ]; then
    STEP_IDX=342
fi

for j in {1..10..1}; do

idx=$((STEP_IDX * j))

checkpoint_path=$PROJECT_DIR/reproducing-reft/train_RL/outputs/${MODEL}_${DATASET_NAME}_${postfix}/global_step_${idx}

if python -m verl.model_merger merge --backend fsdp --local_dir ${checkpoint_path}/actor/ --target_dir ${checkpoint_path}; then
    echo "done"
    rm -rf ${checkpoint_path}/actor/
    rm -rf ${checkpoint_path}/critic/
    rm -rf ${checkpoint_path}/data.pt
fi
done

done
done
done