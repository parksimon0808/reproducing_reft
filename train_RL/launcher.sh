MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=svamp METHOD=ppo BATCH_SIZE=4 sbatch --gres=gpu:4 --account arora --partition ""      -t 01:00:00 job.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=svamp METHOD=ppo BATCH_SIZE=8 sbatch --gres=gpu:4 --account arora --partition ""      -t 01:00:00 job.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=svamp METHOD=ppo BATCH_SIZE=4 sbatch --gres=gpu:4 --account arora --partition ""      -t 01:00:00 job.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=gsm8k METHOD=ppo BATCH_SIZE=4 sbatch --gres=gpu:4 --account pli   --partition pli-c   -t 08:00:00 job.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=gsm8k METHOD=ppo BATCH_SIZE=8 sbatch --gres=gpu:4 --account pli   --partition pli-c   -t 08:00:00 job.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=gsm8k METHOD=ppo BATCH_SIZE=4 sbatch --gres=gpu:4 --account pli   --partition pli-c   -t 08:00:00 job.slurm

MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=math  METHOD=ppo BATCH_SIZE=4 sbatch --gres=gpu:4 --account pli   --partition pli-c   -t 24:00:00 job.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=svamp METHOD=ppo BATCH_SIZE=4 sbatch --gres=gpu:4 --account arora --partition ""      -t 01:00:00 job_warmup.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=svamp METHOD=ppo BATCH_SIZE=8 sbatch --gres=gpu:4 --account arora --partition ""      -t 01:00:00 job_warmup.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=svamp METHOD=ppo BATCH_SIZE=4 sbatch --gres=gpu:4 --account arora --partition ""      -t 01:00:00 job_warmup.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=gsm8k METHOD=ppo BATCH_SIZE=4 sbatch --gres=gpu:4 --account pli   --partition pli-c   -t 08:00:00 job_warmup.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=gsm8k METHOD=ppo BATCH_SIZE=8 sbatch --gres=gpu:4 --account pli   --partition pli-c   -t 08:00:00 job_warmup.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=gsm8k METHOD=ppo BATCH_SIZE=4 sbatch --gres=gpu:4 --account pli   --partition pli-c   -t 08:00:00 job_warmup.slurm

MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=math  METHOD=ppo BATCH_SIZE=4 sbatch --gres=gpu:4 --account pli   --partition pli-c   -t 24:00:00 job_warmup.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=gsm8k METHOD=grpo BATCH_SIZE=4 sbatch --gres=gpu:2 --account pli --partition pli-c -t 20:00:00 job.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=gsm8k METHOD=grpo BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=gsm8k METHOD=grpo BATCH_SIZE=4 sbatch --gres=gpu:2 --account pli --partition pli-c -t 24:00:00 job.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=svamp METHOD=grpo BATCH_SIZE=4 sbatch --gres=gpu:2 --account pli --partition pli-c -t 02:00:00 job.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=svamp METHOD=grpo BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli --partition pli-c -t 01:20:00 job.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=svamp METHOD=grpo BATCH_SIZE=4 sbatch --gres=gpu:2 --account pli --partition pli-c -t 02:30:00 job.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=math  METHOD=grpo BATCH_SIZE=2 sbatch --gres=gpu:4 --account pli --partition pli-c -t 24:00:00 job.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=math  METHOD=grpo BATCH_SIZE=2 sbatch --gres=gpu:4 --account pli --partition pli-c -t 24:00:00 job.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=gsm8k METHOD=grpo BATCH_SIZE=4 sbatch --gres=gpu:2 --account pli --partition pli-c -t 20:00:00 job_warmup.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=gsm8k METHOD=grpo BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_warmup.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=gsm8k METHOD=grpo BATCH_SIZE=4 sbatch --gres=gpu:2 --account pli --partition pli-c -t 24:00:00 job_warmup.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=svamp METHOD=grpo BATCH_SIZE=4 sbatch --gres=gpu:2 --account pli --partition pli-c -t 02:00:00 job_warmup.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=svamp METHOD=grpo BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli --partition pli-c -t 01:20:00 job_warmup.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=svamp METHOD=grpo BATCH_SIZE=4 sbatch --gres=gpu:2 --account pli --partition pli-c -t 02:30:00 job_warmup.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=math  METHOD=grpo BATCH_SIZE=2 sbatch --gres=gpu:4 --account pli --partition pli-c -t 24:00:00 job_warmup.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=math  METHOD=grpo BATCH_SIZE=2 sbatch --gres=gpu:4 --account pli --partition pli-c -t 24:00:00 job_warmup.slurm

ST_MODE=online MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=gsm8k BATCH_SIZE=4 sbatch --gres=gpu:2 --account pli   --partition pli-c -t 20:00:00 job_st.slurm
ST_MODE=online MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=gsm8k BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli   --partition pli-c -t 12:00:00 job_st.slurm
ST_MODE=online MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=gsm8k BATCH_SIZE=4 sbatch --gres=gpu:2 --account pli   --partition pli-c -t 24:00:00 job_st.slurm

ST_MODE=online MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=svamp BATCH_SIZE=4 sbatch --gres=gpu:4 --account arora --partition ""    -t 01:00:00 job_st.slurm
ST_MODE=online MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=svamp BATCH_SIZE=8 sbatch --gres=gpu:4 --account arora --partition ""    -t 01:00:00 job_st.slurm
ST_MODE=online MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=svamp BATCH_SIZE=4 sbatch --gres=gpu:4 --account arora --partition ""    -t 01:00:00 job_st.slurm

ST_MODE=offline MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=gsm8k BATCH_SIZE=4 sbatch --gres=gpu:2 --account pli   --partition pli-c -t 20:00:00 job_st.slurm
ST_MODE=offline MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=gsm8k BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli   --partition pli-c -t 12:00:00 job_st.slurm
ST_MODE=offline MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=gsm8k BATCH_SIZE=4 sbatch --gres=gpu:2 --account pli   --partition pli-c -t 24:00:00 job_st.slurm

ST_MODE=offline MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=svamp BATCH_SIZE=4 sbatch --gres=gpu:2 --account pli   --partition pli-c -t 02:00:00 job_st.slurm
ST_MODE=offline MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=svamp BATCH_SIZE=8 sbatch --gres=gpu:2 --account arora --partition ""    -t 01:00:00 job_st.slurm
ST_MODE=offline MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=svamp BATCH_SIZE=4 sbatch --gres=gpu:2 --account pli   --partition pli-c -t 04:00:00 job_st.slurm

ST_MODE=online  METHOD_LABEL=online_st_og  MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli   --partition pli-c -t 12:00:00 job_st.slurm
ST_MODE=online  METHOD_LABEL=online_st_og  MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=svamp BATCH_SIZE=8 sbatch --gres=gpu:4 --account arora --partition ""    -t 01:00:00 job_st.slurm

ST_MODE=offline METHOD_LABEL=offline_st_og NEG_REWARD=0.0 MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli   --partition pli-c -t 12:00:00 job_st.slurm
ST_MODE=offline METHOD_LABEL=offline_st_og NEG_REWARD=0.0 MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=svamp BATCH_SIZE=8 sbatch --gres=gpu:2 --account arora --partition ""    -t 01:00:00 job_st.slurm

KL_COEF=0     ABLATION_TAG=kl_0     MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_ablations.slurm
KL_COEF=0.001 ABLATION_TAG=kl_0.001 MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_ablations.slurm
KL_COEF=0.01  ABLATION_TAG=kl_0.01  MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_ablations.slurm
KL_COEF=0.1   ABLATION_TAG=kl_0.1   MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_ablations.slurm

LR=1e-7 ABLATION_TAG=lr_1e-7 MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_ablations.slurm
LR=3e-7 ABLATION_TAG=lr_3e-7 MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_ablations.slurm
LR=3e-6 ABLATION_TAG=lr_3e-6 MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_ablations.slurm

N_ROLLOUTS=4  ABLATION_TAG=K_4  MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli --partition pli-c -t 09:00:00 job_ablations.slurm
N_ROLLOUTS=16 ABLATION_TAG=K_16 MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli --partition pli-c -t 20:00:00 job_ablations.slurm

CLIP_RATIO_LOW=0.2 CLIP_RATIO_HIGH=0.26 ABLATION_TAG=clip_0.2_0.26 MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k BATCH_SIZE=8 sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_ablations.slurm
