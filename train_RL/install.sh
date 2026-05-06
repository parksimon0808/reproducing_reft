#!/bin/bash
set -e

ENV_NAME=${ENV_NAME:-reft-train-rl}
TORCH_VERSION=${TORCH_VERSION:-2.10.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.25.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.10.0}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
CUDA_INDEX=${CUDA_INDEX:-cu128}
FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-2.8.3}

if [ ! -d "${CUDA_HOME}" ]; then
    echo "ERROR: CUDA toolkit not found at ${CUDA_HOME}"
    exit 1
fi
if [ ! -f "${CUDA_HOME}/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at ${CUDA_HOME}/bin/nvcc"
    exit 1
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME}"
echo "Using nvcc: $(nvcc --version | tail -1)"

conda create -n "${ENV_NAME}" -y python=3.11
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
pip install --upgrade pip setuptools wheel

pip install \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/${CUDA_INDEX}

pip install \
    "trl>=0.16" \
    "transformers>=4.51" \
    "accelerate>=1.2" \
    "deepspeed>=0.15.4" \
    "datasets>=2.21" \
    "peft>=0.12" \
    "wandb>=0.19" \
    "math-verify"

pip install \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/${CUDA_INDEX}

MAX_JOBS=${MAX_JOBS:-4} pip install \
    flash-attn==${FLASH_ATTN_VERSION} \
    --no-build-isolation
MAX_JOBS=${MAX_JOBS:-4} pip install causal-conv1d==1.6.1 --no-build-isolation
pip install flash-linear-attention==0.4.2

python -c "
import sys

errors = []

try:
    import torch
    print(f'  torch {torch.__version__}  CUDA: {torch.version.cuda}')
    if torch.cuda.is_available():
        t = torch.tensor([1, 2, 3]).cuda()
        print(f'  CUDA device: {torch.cuda.get_device_name()}')
    else:
        errors.append('CUDA not available to PyTorch')
except Exception as e:
    errors.append(f'torch: {e}')

try:
    import trl
    print(f'  trl {trl.__version__}')
except Exception as e:
    errors.append(f'trl: {e}')

try:
    import transformers
    print(f'  transformers {transformers.__version__}')
except Exception as e:
    errors.append(f'transformers: {e}')

try:
    import accelerate
    print(f'  accelerate {accelerate.__version__}')
except Exception as e:
    errors.append(f'accelerate: {e}')

try:
    import deepspeed
    print(f'  deepspeed {deepspeed.__version__}')
except Exception as e:
    errors.append(f'deepspeed: {e}')

try:
    import flash_attn
    print(f'  flash-attn {flash_attn.__version__}')
except Exception as e:
    errors.append(f'flash-attn: {e}')

if errors:
    print()
    for err in errors:
        print(f'  ERROR: {err}')
    sys.exit(1)
else:
    print()
    print('All checks passed!')
"
