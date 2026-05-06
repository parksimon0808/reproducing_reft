MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 20:00:00 job_verl_grpo.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_verl_grpo.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 24:00:00 job_verl_grpo.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=svamp sbatch --gres=gpu:2 --account pli --partition pli-c -t 02:00:00 job_verl_grpo.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=svamp sbatch --gres=gpu:2 --account pli --partition pli-c -t 01:20:00 job_verl_grpo.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=svamp sbatch --gres=gpu:2 --account pli --partition pli-c -t 02:30:00 job_verl_grpo.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=math sbatch --gres=gpu:4 --account pli --partition pli-c -t 24:00:00 job_verl_grpo.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=math sbatch --gres=gpu:4 --account pli --partition pli-c -t 24:00:00 job_verl_grpo.slurm