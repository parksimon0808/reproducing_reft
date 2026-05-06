#!/bin/bash
#SBATCH --job-name=reft-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=120G
#SBATCH --gres=gpu:1
#SBATCH --constraint gpu80
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --account arora
#SBATCH --mail-type=begin,end,fail

PROTOCOLS="zeroshot avg4"
OUTPUT_ROOT=output
PROJECT_ROOT=$PROJECT_DIR/reproducing-reft
MODELS_DIR=$PROJECT_DIR/models

: "${MODEL_BASE_NAME:?set MODEL_BASE_NAME (e.g. Qwen3-1.7B-Base)}"
: "${METHOD:?set METHOD}"
: "${DATASET:?set DATASET}"

source scripts/resolve_checkpoint.sh

module purge
module load anaconda3/2024.6
conda activate reft-train-rl-vllm

mkdir -p logs

run_one() {
    local model_path="$1"
    local method_label="$2"
    local eval_config="$3"
    local out_dir="${OUTPUT_ROOT}/${MODEL_BASE_NAME}/${method_label}/${eval_config}"

    echo ""
    echo "=== [${method_label}] ${eval_config} ==="
    echo "    checkpoint: ${model_path}"
    echo "    output:     ${out_dir}"

    python generate.py \
        --config "configs/${eval_config}.yaml" \
        --model_path "${model_path}" \
        --model_base_name "${MODEL_BASE_NAME}" \
        --method_label "${method_label}" \
        --output_root "${OUTPUT_ROOT}"

    python evaluate.py --input "${out_dir}/generations.jsonl"
}

for epoch in 1 2 3 4 5 6 7 8 9 10; do
    ckpt=$(resolve_checkpoint "${METHOD}" "${MODEL_BASE_NAME}" "${DATASET}" "${epoch}")
    if [ -z "${ckpt}" ]; then
        echo "skip: ${METHOD} ${MODEL_BASE_NAME} trained_on=${DATASET} epoch=${epoch} (checkpoint not found)"
        continue
    fi

    model_path="${ckpt}"
    case "${METHOD}" in
        grpo|grpo_warmup|ppo|ppo_warmup|online_st|online_st_og|grpo_ablation)
            if [ -d "${ckpt}/actor" ]; then
                echo "Merging verl FSDP shards in ${ckpt}"
                if python -m verl.model_merger merge \
                        --backend fsdp \
                        --local_dir "${ckpt}/actor/" \
                        --target_dir "${ckpt}"; then
                    rm -rf "${ckpt}/actor/" "${ckpt}/critic/" "${ckpt}/data.pt"
                else
                    echo "ERROR: verl merge failed for ${ckpt}; skipping epoch=${epoch}"
                    continue
                fi
            fi
            ;;
    esac

    if [[ "${MODEL_BASE_NAME}" == Qwen3.5* ]]; then
        merged="${ckpt}_merged"
        if [ ! -d "${merged}" ]; then
            echo "Fixing Qwen3.5 layout ${ckpt} -> ${merged}"
            python "${PROJECT_ROOT}/merge_sft_checkpoint.py" \
                --sft-checkpoint "${ckpt}" \
                --original-model "${MODELS_DIR}/${MODEL_BASE_NAME}" \
                --output-dir "${merged}"
        fi
        model_path="${merged}"
    fi

    method_label="${METHOD}${ABLATION_TAG:+_${ABLATION_TAG}}_epoch${epoch}"
    for protocol in ${PROTOCOLS}; do
        eval_config="${DATASET}_${protocol}"
        run_one "${model_path}" "${method_label}" "${eval_config}"
    done
done

echo ""
echo "=== Done: ${MODEL_BASE_NAME} / ${METHOD} / trained_on=${DATASET} ==="
