#!/usr/bin/env bash
set -euo pipefail

MODEL_BASE_NAME=${MODEL_BASE_NAME:-Qwen3-1.7B-Base}
DATASET=${DATASET:-gsm8k}
ABLATION_TAGS=${ABLATION_TAGS:-"kl_0 kl_0.001 kl_0.01 kl_0.1 lr_1e-7 lr_3e-7 lr_3e-6 K_4 K_16 clip_0.2_0.26"}

total=0
for tag in ${ABLATION_TAGS}; do
    job_id=$(MODEL_BASE_NAME="${MODEL_BASE_NAME}" METHOD="grpo_ablation" \
                DATASET="${DATASET}" ABLATION_TAG="${tag}" \
                sbatch --export=ALL scripts/run_one.sh | awk '{print $NF}')
    echo "Submitted  tag=${tag}  ->  job ${job_id}"
    total=$((total + 1))
done
echo "Submitted ${total} jobs."
