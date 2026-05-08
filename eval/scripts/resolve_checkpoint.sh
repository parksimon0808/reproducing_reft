#!/bin/bash
PROJECT_ROOT="${PROJECT_ROOT:-$PROJECT_DIR/reproducing-reft}"

rl_save_freq_for_dataset() {
    case "$1" in
        gsm8k) echo 226 ;;
        svamp) echo 21 ;;
        math)  echo 342 ;;
        *)     echo "" ;;
    esac
}

_resolve_sft() {
    local model_base="$1"
    local dataset="$2"
    local epoch="$3"
    local parent="${PROJECT_ROOT}/sft/outputs/${model_base}_${dataset}"
    find "${parent}" -maxdepth 1 -mindepth 1 -type d \
        -regex ".*/step_[0-9]+_epoch_${epoch}$" 2>/dev/null | head -n 1
}

_resolve_rl_variant() {
    local model_base="$1"
    local dataset="$2"
    local epoch="$3"
    local variant="$4"
    local save_freq
    save_freq=$(rl_save_freq_for_dataset "${dataset}")
    if [ -z "${save_freq}" ]; then
        echo ""
        return
    fi
    local step=$(( epoch * save_freq ))
    local ckpt="${PROJECT_ROOT}/train_RL/outputs/${model_base}_${dataset}_${variant}/global_step_${step}"
    if [ -d "${ckpt}" ]; then
        echo "${ckpt}"
    else
        echo ""
    fi
}

resolve_checkpoint() {
    local method="$1"
    local model_base="$2"
    local dataset="$3"
    local epoch="$4"

    case "${method}" in
        sft)
            _resolve_sft "${model_base}" "${dataset}" "${epoch}"
            ;;
        grpo|grpo_warmup|ppo|ppo_warmup|offline_st|online_st|offline_st_og|online_st_og)
            _resolve_rl_variant "${model_base}" "${dataset}" "${epoch}" "${method}"
            ;;
        grpo_ablation)
            : "${ABLATION_TAG:?set ABLATION_TAG when METHOD=grpo_ablation}"
            _resolve_rl_variant "${model_base}" "${dataset}" "${epoch}" "grpo_ablation_${ABLATION_TAG}"
            ;;
        *)
            echo ""
            ;;
    esac
}
