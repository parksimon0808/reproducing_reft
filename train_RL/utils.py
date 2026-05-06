import json
import sys
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generate_data.utils import format_prompt, extract_answer, verify_answer

_REPO_ROOT = Path(__file__).resolve().parent.parent

_CHAT_TEMPLATE = '{{ messages[0]["content"] }}'

_KEEP_FILES = frozenset({
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
})


def save_checkpoint(model, tokenizer, output_dir, step):
    save_dir = os.path.join(output_dir, f"global_step_{step}")
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="100GB")
    tokenizer.save_pretrained(save_dir)

    with open(os.path.join(save_dir, "chat_template.jinja"), "w") as f:
        f.write(_CHAT_TEMPLATE)

    for entry in os.listdir(save_dir):
        entry_path = os.path.join(save_dir, entry)
        if entry in _KEEP_FILES:
            continue
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)

    return save_dir


def load_jsonl_split(dataset_name, split):
    suffix = "_final" if split == "train" else "_test"
    path = _REPO_ROOT / "generate_data" / "outputs" / f"{dataset_name}{suffix}.jsonl"
    questions, golds = [], []
    with open(path) as f:
        lines = f.readlines()
        n = 8 if split == "train" else 1
        for line in lines[::n]:
            row = json.loads(line)
            questions.append(row["question"])
            golds.append(str(row["gold_answer"]))
    return questions, golds


def reward_weighted_loss(logits, input_ids, response_mask, rewards):
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    mask = response_mask[:, 1:].float()
    weighted = rewards.unsqueeze(1) * token_log_probs * mask

    response_lengths = mask.sum(dim=1)
    valid = response_lengths > 0
    if not valid.any():
        return weighted.sum() * 0.0

    per_sample_loss = -(weighted[valid].sum(dim=1) / response_lengths[valid])
    return per_sample_loss.mean()


def build_masks(generated_ids, prompt_len, prompt_mask, pad_token_id, device):
    B, T = generated_ids.shape
    attn_mask = torch.ones(B, T, device=device, dtype=torch.long)
    attn_mask[:, :prompt_len] = prompt_mask
    attn_mask[:, prompt_len:] = (generated_ids[:, prompt_len:] != pad_token_id).long()

    response_mask = torch.zeros(B, T, device=device)
    response_mask[:, prompt_len:] = attn_mask[:, prompt_len:].float()
    return attn_mask, response_mask


@torch.no_grad()
def evaluate(model, tokenizer, questions, gold_answers, dataset_name,
             max_new_tokens=1024, batch_size=16, device="cuda"):
    model.eval()
    correct = 0
    total = len(questions)

    for start in range(0, total, batch_size):
        batch_q = questions[start:start + batch_size]
        batch_g = gold_answers[start:start + batch_size]
        prompts = [format_prompt(q) for q in batch_q]

        encodings = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        input_ids = encodings.input_ids.to(device)
        attn_mask = encodings.attention_mask.to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        prompt_len = input_ids.shape[1]
        for b in range(len(batch_q)):
            resp_ids = outputs[b, prompt_len:]
            text = tokenizer.decode(resp_ids, skip_special_tokens=True)
            extracted = extract_answer(text, dataset_name)
            if extracted is not None and verify_answer(extracted, batch_g[b], dataset_name):
                correct += 1

    model.train()
    return correct / total if total > 0 else 0.0
