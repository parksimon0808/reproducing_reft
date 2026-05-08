## Reproducing "ReFT: Reasoning with reinforced fine-tuning"

This repository contains the code for the final project for COS435 at Princeton University.
Authors: Farah Attia, Santiago Criado, Simon Park

## Quick Links
- `generate_data`: code for generating SFT data
- `sft`: code for running SFT. uses TRL
- `train_RL`: code for running self-training and RL. train.py is based on the original ReFT codebase but updated to match the signature of the new torch and TRL versions
- `eval`: code for evaluating checkpoints
- `merge_sft_checkpoint.py`: For Qwen3.5 models, TRL saves only the language submodule. This code copies over the vision submodule from the original dataset. Code copied from a different project of Simon's