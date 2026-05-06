#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="eval"

conda create -y -n "${ENV_NAME}" python=3.10

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

pip install vllm==0.8.4

pip install \
    transformers==4.49.0 \
    datasets==3.3.0 \
    math-verify==0.7.0 \
    pyyaml==6.0.2 \
    numpy==1.26.4 \
    tqdm==4.67.1
