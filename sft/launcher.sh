MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=gsm8k sbatch --ntasks-per-node=4 --gres=gpu:4 --account pli --partition pli-c -t 01:00:00  job.slurm
MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k sbatch --ntasks-per-node=4 --gres=gpu:4 --account pli --partition pli-c -t 02:00:00  job.slurm
MODEL_NAME=Qwen3.5-2B-Base DATASET_NAME=gsm8k sbatch --ntasks-per-node=4 --gres=gpu:4 --account pli --partition pli-c -t 02:00:00  job.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=svamp sbatch --ntasks-per-node=4 --gres=gpu:4 --account arora --partition "" -t 00:20:00  job.slurm
MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=svamp sbatch --ntasks-per-node=4 --gres=gpu:4 --account arora --partition "" -t 00:40:00  job.slurm
MODEL_NAME=Qwen3.5-2B-Base DATASET_NAME=svamp sbatch --ntasks-per-node=4 --gres=gpu:4 --account arora --partition "" -t 00:30:00  job.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=math sbatch --ntasks-per-node=4 --gres=gpu:4 --account pli --partition pli-c -t 04:00:00  job.slurm
MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=math sbatch --ntasks-per-node=4 --gres=gpu:4 --account pli --partition pli-c -t 08:00:00  job.slurm
MODEL_NAME=Qwen3.5-2B-Base DATASET_NAME=math sbatch --ntasks-per-node=4 --gres=gpu:4 --account pli --partition pli-c -t 08:00:00  job.slurm