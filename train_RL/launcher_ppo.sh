MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=svamp sbatch --gres=gpu:2 --account arora --partition "" -t 01:00:00 job_verl.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=svamp sbatch --gres=gpu:2 --account arora --partition "" -t 01:00:00 job_verl.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=svamp sbatch --gres=gpu:2 --account arora --partition "" -t 01:00:00 job_verl.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=svamp sbatch --gres=gpu:2 --account arora --partition "" -t 01:00:00 job_verl_warmup.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=svamp sbatch --gres=gpu:2 --account arora --partition "" -t 01:00:00 job_verl_warmup.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=svamp sbatch --gres=gpu:2 --account arora --partition "" -t 01:00:00 job_verl_warmup.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:4 --account pli --partition pli-c -t 08:00:00 job_verl.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=gsm8k sbatch --gres=gpu:4 --account pli --partition pli-c -t 08:00:00 job_verl.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=gsm8k sbatch --gres=gpu:4 --account pli --partition pli-c -t 08:00:00 job_verl.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:4 --account pli --partition pli-c -t 08:00:00 job_verl_warmup.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=gsm8k sbatch --gres=gpu:4 --account pli --partition pli-c -t 08:00:00 job_verl_warmup.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=gsm8k sbatch --gres=gpu:4 --account pli --partition pli-c -t 08:00:00 job_verl_warmup.slurm

MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=math sbatch --gres=gpu:4 --account pli --partition pli-c -t 24:00:00 job_verl.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=math sbatch --gres=gpu:4 --account pli --partition pli-c -t 24:00:00 job_verl_warmup.slurm