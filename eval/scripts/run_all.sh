#!/usr/bin/env bash
set -euo pipefail

MODELS=${MODELS:-"Qwen3-1.7B-Base Qwen3.5-0.8B-Base Qwen3.5-2B-Base"}
MODELS="Qwen3-1.7B-Base"
METHODS="offline_st_og"
DATASETS="gsm8k"

total=0
for model in ${MODELS}; do
    for method in ${METHODS}; do
        for dataset in ${DATASETS}; do
            job_id=$(MODEL_BASE_NAME="${model}" METHOD="${method}" DATASET="${dataset}" \
                        sbatch --export=ALL --dependency="" scripts/run_one.sh | awk '{print $NF}')
            echo "Submitted  model=${model}  method=${method}  dataset=${dataset}  ->  job ${job_id}"
            total=$((total + 1))
        done
    done
done
echo "Submitted ${total} jobs."
