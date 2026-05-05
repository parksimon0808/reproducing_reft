MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=gsm8k sbatch --ntasks-per-node=1 --gres=gpu:1 --account pli --partition pli-c -t 02:00:00 --mail-user juhyunp@princeton.edu job_og.slurm
MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k sbatch --ntasks-per-node=1 --gres=gpu:1 --account pli --partition pli-c -t 02:00:00 --mail-user juhyunp@princeton.edu job_og.slurm
MODEL_NAME=Qwen3.5-2B-Base DATASET_NAME=gsm8k sbatch --ntasks-per-node=1 --gres=gpu:1 --account pli --partition pli-c -t 02:00:00 --mail-user juhyunp@princeton.edu job_og.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=svamp sbatch --ntasks-per-node=1 --gres=gpu:1 --account arora --partition "" -t 00:20:00 --mail-user juhyunp@princeton.edu job_og.slurm
MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=svamp sbatch --ntasks-per-node=1 --gres=gpu:1 --account arora --partition "" -t 00:40:00 --mail-user juhyunp@princeton.edu job_og.slurm
MODEL_NAME=Qwen3.5-2B-Base DATASET_NAME=svamp sbatch --ntasks-per-node=1 --gres=gpu:1 --account arora --partition "" -t 00:30:00 --mail-user juhyunp@princeton.edu job_og.slurm