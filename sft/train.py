import json
import os
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    HfArgumentParser,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from generate_data.utils import format_prompt, extract_answer, verify_answer

tqdm = partial(tqdm, ncols=0, leave=False)

SUPPORTED_DATASETS = {"gsm8k", "svamp", "math"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_tokenize_fn(args, tokenizer):
    dataset_name = args["dataset"]
    max_len = args["max_input_length"]

    def tokenize_fn(batch):
        assert tokenizer.eos_token_id is not None, "Tokenizer must have an eos_token_id set"
        out = defaultdict(list)
        keys = list(batch.keys())

        for values in zip(*(batch[k] for k in keys)):
            item = dict(zip(keys, values))

            question = item["question"].strip()
            gold_answer = str(item["gold_answer"]).strip()
            response = (item.get("response") or "").strip()

            prompt_text = format_prompt(question)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

            if response:
                resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
                input_ids = prompt_ids + resp_ids + [tokenizer.eos_token_id]
                labels = [-100] * len(prompt_ids) + resp_ids + [tokenizer.eos_token_id]
            else:
                input_ids = list(prompt_ids)
                labels = [-100] * len(prompt_ids)

            attention_mask = [1] * len(input_ids)

            raw_len = len(input_ids)
            input_ids     = input_ids[:max_len]
            labels        = labels[:max_len]
            attention_mask = attention_mask[:max_len]
            prefix_ids    = prompt_ids[:max_len]
            prefix_mask   = [1] * len(prefix_ids)

            out["input_ids"].append(input_ids)
            out["labels"].append(labels)
            out["attention_mask"].append(attention_mask)
            out["prefix"].append(prefix_ids)
            out["prefix_attention_mask"].append(prefix_mask)
            out["item_id"].append(f"{dataset_name}_{item['index']}")
            out["question"].append(question)
            out["gold_answer"].append(gold_answer)
            out["raw_input_length"].append(raw_len)

        return out

    return tokenize_fn


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        max_seq = max(len(x["input_ids"]) for x in batch)
        max_pfx = max(len(x["prefix"]) for x in batch)

        input_ids, attn_mask, labels = [], [], []
        prefix_lp, prefix_mask_lp = [], []

        for x in batch:
            pad_seq = max_seq - len(x["input_ids"])
            pad_pfx = max_pfx - len(x["prefix"])

            input_ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad_seq)
            attn_mask.append(x["attention_mask"] + [0] * pad_seq)
            labels.append(x["labels"] + [-100] * pad_seq)

            prefix_lp.append([tokenizer.pad_token_id] * pad_pfx + x["prefix"])
            prefix_mask_lp.append([0] * pad_pfx + x["prefix_attention_mask"])

        return {
            "forward_kwargs": {
                "input_ids":      torch.LongTensor(input_ids),
                "attention_mask": torch.LongTensor(attn_mask),
                "labels":         torch.LongTensor(labels),
            },
            "generate_prefix_kwargs": {
                "input_ids":      torch.LongTensor(prefix_lp),
                "attention_mask": torch.LongTensor(prefix_mask_lp),
            },
            "metadata": {
                "item_id":     [x["item_id"] for x in batch],
                "question":    [x["question"] for x in batch],
                "gold_answer": [x["gold_answer"] for x in batch],
            },
        }

    return collate_fn


def prepare_data(args, tokenizer):
    with accelerator.main_process_first():
        raw = DatasetDict({
            "train": Dataset.from_list(load_jsonl(args["train_file"])),
            "test":  Dataset.from_list(load_jsonl(args["test_file"])),
        })
        accelerator.print("Raw dataset:", raw)

        tokenize_fn = build_tokenize_fn(args, tokenizer)
        tokenized = DatasetDict({
            split: ds.map(
                tokenize_fn,
                batched=True,
                remove_columns=ds.column_names,
                num_proc=args["num_proc"],
                load_from_cache_file=False,
            )
            for split, ds in raw.items()
        })
        accelerator.print("Tokenized dataset:", tokenized)
        for split, ds in tokenized.items():
            lens = ds["raw_input_length"]
            truncated = sum(1 for l in lens if l > args["max_input_length"])
            accelerator.print(
                f"  {split}: max={max(lens)} p99={sorted(lens)[int(len(lens)*0.99)]} "
                f"truncated={truncated}/{len(lens)}"
            )

        if accelerator.is_main_process and args["wandb_log"]:
            wandb.config.update({
                "dataset": args["dataset"],
                "train_max_len": max(tokenized["train"]["raw_input_length"]),
                "test_max_len":  max(tokenized["test"]["raw_input_length"]),
            })

    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(
        tokenized["train"], shuffle=True,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        tokenized["test"], shuffle=False,
        batch_size=args["eval_batch_size"],
        num_workers=args["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return (tokenized["train"], train_loader), (tokenized["test"], test_loader)


def save_checkpoint(args, model, tokenizer, save_path, recent_ckpts=None):
    os.makedirs(save_path, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        save_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    tokenizer.save_pretrained(save_path)

    if accelerator.is_main_process and recent_ckpts is not None:
        recent_ckpts.append(save_path)
        if args["keep_num_ckpt"] and len(recent_ckpts) > args["keep_num_ckpt"]:
            old = recent_ckpts.pop(0)
            shutil.rmtree(old)


def run_evaluation(args, model, dataset, dataloader, tokenizer, tag=""):
    model.eval()
    unwrapped = accelerator.unwrap_model(model)

    local_preds, local_ids, local_golds, local_questions = [], [], [], []

    for batch in tqdm(dataloader, disable=not accelerator.is_main_process, desc="Evaluating"):
        prefix_kwargs = {k: v.to(accelerator.device) for k, v in batch["generate_prefix_kwargs"].items()}
        prefix_len = prefix_kwargs["input_ids"].shape[1]
        with torch.no_grad():
            out = unwrapped.generate(
                **prefix_kwargs,
                max_new_tokens=args["max_new_tokens"],
                num_beams=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_only = out[:, prefix_len:]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        local_preds.extend(d.strip() for d in decoded)
        local_ids.extend(batch["metadata"]["item_id"])
        local_golds.extend(batch["metadata"]["gold_answer"])
        local_questions.extend(batch["metadata"]["question"])

    all_preds     = gather_object(local_preds)[:len(dataset)]
    all_ids       = gather_object(local_ids)[:len(dataset)]
    all_golds     = gather_object(local_golds)[:len(dataset)]
    all_questions = gather_object(local_questions)[:len(dataset)]

    if accelerator.is_main_process:
        dataset_name = args["dataset"]
        records, n_correct = [], 0
        for pred, iid, gold, q in zip(all_preds, all_ids, all_golds, all_questions):
            extracted = extract_answer(pred, dataset_name)
            correct = verify_answer(extracted, gold, dataset_name)
            records.append({
                "item_id":    iid,
                "question":   q,
                "gold_answer": gold,
                "prediction": pred,
                "extracted":  extracted,
                "is_correct": bool(correct),
            })
            n_correct += int(correct)

        suffix = f"_{tag}" if tag else ""
        res_path = os.path.join(args["model_dir"].rstrip("/"), f"_res{suffix}.json")
        with open(res_path, "w") as f:
            json.dump(records, f, indent=2)

        acc = n_correct / max(len(records), 1) * 100
        accelerator.print(f"[Eval{(' '+tag) if tag else ''}] accuracy: {acc:.4f}% ({n_correct}/{len(records)})")
        acc_tensor = torch.FloatTensor([acc]).to(accelerator.device)
    else:
        acc_tensor = torch.FloatTensor([-1.0]).to(accelerator.device)

    from accelerate.utils import broadcast
    acc = broadcast(acc_tensor).cpu().numpy().tolist()[0]
    model.train()
    return {"accuracy": acc}


def train_one_epoch(
    args, model, train_dataset, train_dataloader, test_dataset, test_dataloader,
    optimizer, scheduler, tokenizer,
    global_step, epoch, best_eval_log, summary_log, recent_ckpts, prefix=""
):
    model.train()
    clip_norm       = args.get("clip_grad_norm")
    eval_step_freq  = args.get("evaluating_step_freq")
    log_step_freq   = args.get("logging_step_freq")
    save_step_freq  = args.get("saving_step_freq")

    epoch_losses = []

    for batch in tqdm(train_dataloader, disable=not accelerator.is_main_process, desc="Training"):
        with accelerator.accumulate(model):
            loss = model(**batch["forward_kwargs"]).loss
            accelerator.backward(loss)
            if accelerator.sync_gradients and clip_norm:
                accelerator.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad()

        if not accelerator.sync_gradients:
            continue

        global_step += 1
        epoch_losses.append(loss.item())

        eval_log, is_best = {}, False
        if eval_step_freq and global_step % eval_step_freq == 0:
            eval_results = run_evaluation(args, model, test_dataset, test_dataloader, tokenizer, tag=f"step{global_step}")
            eval_log = {f"Eval.Gen.{k}": v for k, v in eval_results.items()}
            current_acc = eval_log.get("Eval.Gen.accuracy", 0)
            if current_acc > best_eval_log.get("Eval.Gen.accuracy_best", 0):
                is_best = True
                best_eval_log["Eval.Gen.accuracy_best"] = current_acc
            summary_log.setdefault("Eval.Gen.accuracy", []).append(current_acc)

        train_log = {}
        if log_step_freq and global_step % log_step_freq == 0:
            train_log = {"T.loss": sum(epoch_losses) / len(epoch_losses)}

        if eval_log or train_log:
            log = {"lr": scheduler.get_last_lr()[0], **train_log, **eval_log, **best_eval_log}
            if accelerator.is_main_process and args["wandb_log"]:
                wandb.log(log, step=global_step)
            formatted = {k: f"{v:.5g}" if isinstance(v, float) else v for k, v in log.items()}
            accelerator.print(f"{prefix}[E={epoch}/{args['n_epochs']} S={global_step}] {formatted}")

        if save_step_freq and global_step % save_step_freq == 0:
            if is_best:
                save_checkpoint(args, model, tokenizer, os.path.join(args["model_dir"], "best"))
            if args["keep_num_ckpt"] > 0:
                save_checkpoint(args, model, tokenizer, os.path.join(args["model_dir"], f"step_{global_step}"), recent_ckpts)

    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    return {"loss": avg_loss}, global_step


def main(args):
    set_seed(args["seed"] + accelerator.process_index)

    if accelerator.is_main_process and args["wandb_log"]:
        wandb.init(project=args["wandb_project"], name=args["wandb_run_name"])
        wandb.config.update(args)

    tokenizer = AutoTokenizer.from_pretrained(args["tokenizer_name_or_path"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        accelerator.print("No pad token found — using eos_token as pad_token")

    (train_dataset, train_loader), (test_dataset, test_loader) = prepare_data(args, tokenizer)

    model_kwargs = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if args["attn_implementation"]:
        model_kwargs["attn_implementation"] = args["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(args["model_name_or_path"], **model_kwargs)
    accelerator.print(f"Model loaded. Vocab size: {len(tokenizer)}")

    if accelerator.is_main_process and args["wandb_log"]:
        wandb.run.summary.update({
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "vocab_size":   len(tokenizer),
        })

    n_epochs = args["n_epochs"]
    total_steps = (len(train_loader) // accelerator.num_processes * n_epochs) // args["gradient_accumulation_steps"]
    warmup_steps = args["warmup_step"] if args["warmup_step"] is not None and args["warmup_step"] >= 0 else int(0.1 * total_steps)

    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=args["learning_rate"], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    accelerator.print(
        f"\n{'='*50}\n"
        f"  Model:       {args['model_name_or_path']}\n"
        f"  Dataset:     {args['dataset']}\n"
        f"  Train size:  {len(train_dataset)}\n"
        f"  Test size:   {len(test_dataset)}\n"
        f"  Epochs:      {n_epochs}\n"
        f"  Batch/device:{args['batch_size']}\n"
        f"  Effective BS:{args['batch_size'] * accelerator.num_processes * args['gradient_accumulation_steps']}\n"
        f"  Total steps: {total_steps}\n"
        f"  Warmup:      {warmup_steps}\n"
        f"  LR:          {args['learning_rate']}\n"
        f"{'='*50}\n"
    )

    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    os.makedirs(args["model_dir"], exist_ok=True)
    global_step  = 0
    best_eval_log = {}
    summary_log  = {}
    recent_ckpts = []

    for epoch in tqdm(range(1, n_epochs + 1), total=n_epochs, desc="Epochs"):
        epoch_log, global_step = train_one_epoch(
            args=args, model=model,
            train_dataset=train_dataset, train_dataloader=train_loader,
            test_dataset=test_dataset,   test_dataloader=test_loader,
            optimizer=optimizer, scheduler=scheduler, tokenizer=tokenizer,
            global_step=global_step, epoch=epoch,
            best_eval_log=best_eval_log, summary_log=summary_log,
            recent_ckpts=recent_ckpts,
        )

        eval_log, is_best = {}, False
        if args["evaluating_epoch_freq"] and epoch % args["evaluating_epoch_freq"] == 0:
            eval_results = run_evaluation(args, model, test_dataset, test_loader, tokenizer, tag=f"epoch{epoch}")
            eval_log = {f"Eval.Gen.{k}": v for k, v in eval_results.items()}
            current_acc = eval_log.get("Eval.Gen.accuracy", 0)
            if current_acc > best_eval_log.get("Eval.Gen.accuracy_best", 0):
                is_best = True
                best_eval_log["Eval.Gen.accuracy_best"] = current_acc
            summary_log.setdefault("Eval.Gen.accuracy", []).append(current_acc)

        train_log = {}
        if args["logging_epoch_freq"] and epoch % args["logging_epoch_freq"] == 0:
            train_log = {f"T.{k}": v for k, v in epoch_log.items()}

        if eval_log or train_log:
            log = {"lr": scheduler.get_last_lr()[0], **train_log, **eval_log, **best_eval_log}
            if accelerator.is_main_process and args["wandb_log"]:
                wandb.log(log, step=global_step)
            formatted = {k: f"{v:.5g}" if isinstance(v, float) else v for k, v in log.items()}
            accelerator.print(f"[E={epoch}/{n_epochs} S={global_step}] {formatted}")

        if args["saving_epoch_freq"] and epoch % args["saving_epoch_freq"] == 0:
            if is_best:
                save_checkpoint(args, model, tokenizer, os.path.join(args["model_dir"], "best"))
            if args["keep_num_ckpt"] > 0:
                save_checkpoint(args, model, tokenizer, os.path.join(args["model_dir"], f"step_{global_step}_epoch_{epoch}"), recent_ckpts)


@dataclass
class Arguments:
    model_name_or_path:     str
    tokenizer_name_or_path: str
    model_dir:              str
    train_file:             str
    test_file:              str
    dataset:                str

    batch_size:             int   = field(default=4)
    eval_batch_size:        int   = field(default=8)
    n_epochs:               int   = field(default=40)
    num_workers:            int   = field(default=8)
    num_proc:               int   = field(default=8)
    learning_rate:          float = field(default=2e-5)
    weight_decay:           float = field(default=1e-6)
    warmup_step:            int   = field(default=0)
    clip_grad_norm:         float = field(default=1.0)
    max_input_length:       int   = field(default=2048)
    max_new_tokens:         int   = field(default=2048)
    gradient_accumulation_steps: int = field(default=1)
    keep_num_ckpt:          int   = field(default=1)
    seed:                   int   = field(default=42)
    attn_implementation:    str   = field(default="flash_attention_2")
    evaluating_epoch_freq:  int   = field(default=1)
    logging_epoch_freq:     int   = field(default=1)
    saving_epoch_freq:      int   = field(default=1000)
    evaluating_step_freq:   int   = field(default=-100)
    logging_step_freq:      int   = field(default=-100)
    saving_step_freq:       int   = field(default=-100)

    wandb_log:              bool  = field(default=False)
    wandb_project:          str   = field(default="reft-sft")
    wandb_run_name:         str   = field(default="default_run")


if __name__ == "__main__":
    NONE_INT = -100
    NONE_STR = "None"

    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = asdict(args)

    for k, v in args.items():
        if v in (NONE_INT, NONE_STR):
            args[k] = None

    if args["dataset"] not in SUPPORTED_DATASETS:
        raise ValueError(f"--dataset must be one of {SUPPORTED_DATASETS}, got {args['dataset']}")

    accelerator = Accelerator(
        gradient_accumulation_steps=args["gradient_accumulation_steps"],
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))],
    )
    accelerator.print(json.dumps(args, indent=2, ensure_ascii=False))
    main(args)
