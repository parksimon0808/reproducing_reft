KL_COEF=0     ABLATION_TAG=kl_0     MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_verl_ablations.slurm
KL_COEF=0.001 ABLATION_TAG=kl_0.001 MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_verl_ablations.slurm
KL_COEF=0.01  ABLATION_TAG=kl_0.01  MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_verl_ablations.slurm
KL_COEF=0.1   ABLATION_TAG=kl_0.1   MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_verl_ablations.slurm

LR=1e-7 ABLATION_TAG=lr_1e-7 MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_verl_ablations.slurm
LR=3e-7 ABLATION_TAG=lr_3e-7 MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_verl_ablations.slurm
LR=3e-6 ABLATION_TAG=lr_3e-6 MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_verl_ablations.slurm

N_ROLLOUTS=4  ABLATION_TAG=K_4  MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 09:00:00 job_verl_ablations.slurm
N_ROLLOUTS=16 ABLATION_TAG=K_16 MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 20:00:00 job_verl_ablations.slurm

CLIP_RATIO_LOW=0.2 CLIP_RATIO_HIGH=0.26 ABLATION_TAG=clip_0.2_0.26 MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_verl_ablations.slurm
