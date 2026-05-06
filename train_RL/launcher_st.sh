MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 20:00:00 job_verl_online_st.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_verl_online_st.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 24:00:00 job_verl_online_st.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=svamp sbatch --dependency=afterany:7422999 --gres=gpu:4 --account arora --partition "" -t 01:00:00 job_verl_online_st.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=svamp sbatch --dependency=afterany:7423000 --gres=gpu:4 --account arora --partition "" -t 01:00:00 job_verl_online_st.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=svamp sbatch --dependency=afterany:7423001 --gres=gpu:4 --account arora --partition "" -t 01:00:00 job_verl_online_st.slurm

MODEL_NAME=Qwen3.5-0.8B-Base DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 20:00:00 job_offline_st.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_offline_st.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=gsm8k sbatch --gres=gpu:2 --account pli --partition pli-c -t 24:00:00 job_offline_st.slurm

MODEL_NAME=Qwen3.5-0.8B-Base   DATASET_NAME=svamp sbatch --gres=gpu:2 --account pli --partition pli-c -t 02:00:00 job_offline_st.slurm
MODEL_NAME=Qwen3-1.7B-Base   DATASET_NAME=svamp sbatch --gres=gpu:2 --account arora --partition "" -t 01:00:00 job_offline_st.slurm
MODEL_NAME=Qwen3.5-2B-Base   DATASET_NAME=svamp sbatch --gres=gpu:2 --account pli --partition pli-c -t 04:00:00 job_offline_st.slurm

METHOD_LABEL=online_st_og REWARD_FN_NAME=compute_score_reinforce_zero \
    MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k \
    sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_verl_online_st.slurm
METHOD_LABEL=online_st_og REWARD_FN_NAME=compute_score_reinforce_zero \
    MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=svamp \
    sbatch --gres=gpu:4 --account arora --partition "" -t 01:00:00 job_verl_online_st.slurm

METHOD_LABEL=offline_st_og NEG_REWARD=0.0 \
    MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=gsm8k \
    sbatch --gres=gpu:2 --account pli --partition pli-c -t 12:00:00 job_offline_st.slurm
METHOD_LABEL=offline_st_og NEG_REWARD=0.0 \
    MODEL_NAME=Qwen3-1.7B-Base DATASET_NAME=svamp \
    sbatch --gres=gpu:2 --account arora --partition "" -t 01:00:00 job_offline_st.slurm