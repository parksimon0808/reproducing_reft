#!/bin/bash
set -e

ENV_NAME="reft-gen-data"

conda create -n "${ENV_NAME}" python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

pip install openai
pip install datasets
pip install 'math-verify[inference]'
pip install tqdm