import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generate_data.utils import extract_answer, verify_answer


def compute_reward(text, gold_answer, dataset_name, reward_type="standard"):
    extracted = extract_answer(text, dataset_name)

    if reward_type == "standard":
        if extracted is not None and verify_answer(extracted, gold_answer, dataset_name):
            return 1.0
        elif extracted is not None:
            return 0.1
        else:
            return 0.0
    elif reward_type == "negative":
        if extracted is not None and verify_answer(extracted, gold_answer, dataset_name):
            return 1.0
        else:
            return -1.0
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")


def compute_batch_rewards(generated_ids, gold_answers, tokenizer, dataset_name,
                          prompt_len, response_mask, reward_type="standard"):
    B = generated_ids.shape[0]
    rewards = []
    for b in range(B):
        resp_ids = generated_ids[b, prompt_len:]
        mask = response_mask[b, prompt_len:].bool()
        valid_ids = resp_ids[mask]
        text = tokenizer.decode(valid_ids, skip_special_tokens=True)
        r = compute_reward(text, gold_answers[b], dataset_name, reward_type)
        rewards.append(r)
    return torch.tensor(rewards, dtype=torch.float32, device=generated_ids.device)
