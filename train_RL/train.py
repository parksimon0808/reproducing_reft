import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback

import wandb

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generate_data.utils import format_prompt, extract_answer, verify_answer
from train_RL.utils import evaluate, load_jsonl_split, save_checkpoint


def logprobs_from_logits(logits, labels):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def discount_cumsum(x, discount):
    out = np.zeros_like(x)
    running = 0.0
    for i in reversed(range(len(x))):
        running = x[i] + discount * running
        out[i] = running
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, default="grpo", choices=["grpo", "ppo", "self_training"])
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "svamp", "math"])
    p.add_argument("--n_epochs", type=int, default=10)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--mini_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)

    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)

    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--max_prompt_length", type=int, default=1024)

    p.add_argument("--kl_coef", type=float, default=0.05)

    p.add_argument("--steps_per_generation", type=int, default=1)
    p.add_argument("--pos_reward", type=float, default=1.0)
    p.add_argument("--neg_reward", type=float, default=-1.0)

    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--save_freq", type=int, default=226)
    p.add_argument("--output_dir", type=str, default="outputs/rl_training")

    p.add_argument("--wandb_project", type=str, default="reproducing-reft")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")

    args = p.parse_args()

    if args.method == "self_training":
        if not cli_flag_provided("num_generations"):
            args.num_generations = 1
        if not cli_flag_provided("lr"):
            args.lr = 1e-6
        if not cli_flag_provided("output_dir"):
            args.output_dir = "outputs/self_training"

    return args


def cli_flag_provided(name):
    flag = f"--{name}"
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def build_dataset(dataset_name, split="train"):
    questions, golds = load_jsonl_split(dataset_name, split)
    prompts = [format_prompt(q) for q in questions]
    return Dataset.from_dict({
        "prompt": prompts,
        "gold_answer": golds,
    })


def make_reward_fn(dataset_name):
    def reward_fn(completions, gold_answer, **kwargs):
        rewards = []
        for text, gold in zip(completions, gold_answer):
            rewards.append(compute_rule_reward(text, gold, dataset_name))
        return rewards
    return reward_fn


def compute_rule_reward(text, gold, dataset_name):
    extracted = extract_answer(text, dataset_name)
    if extracted is not None and verify_answer(extracted, gold, dataset_name):
        return 1.0
    elif extracted is not None:
        return 0.1
    else:
        return 0.0


class GreedyEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_questions, eval_golds, dataset_name,
                 max_new_tokens, eval_every):
        self.tokenizer = tokenizer
        self.eval_questions = eval_questions
        self.eval_golds = eval_golds
        self.dataset_name = dataset_name
        self.max_new_tokens = max_new_tokens
        self.eval_every = eval_every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            return
        if state.global_step % self.eval_every != 0:
            return
        if args.local_process_index != 0:
            return
        if model is None:
            return

        eval_model = getattr(model, "module", model)
        device = next(eval_model.parameters()).device

        try:
            acc = evaluate(
                eval_model, self.tokenizer,
                self.eval_questions, self.eval_golds,
                self.dataset_name, self.max_new_tokens, device=device,
            )
            print(f"  [Eval] Step {state.global_step}: accuracy = {acc:.4f}")
            if wandb.run is not None:
                wandb.log({"eval/accuracy": acc}, step=state.global_step)
        except Exception as e:
            print(f"  [Eval] Step {state.global_step}: failed — {e}")


class EpochCheckpointCallback(TrainerCallback):
    def __init__(self, tokenizer, output_dir, save_freq):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.save_freq = save_freq

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if args.local_process_index != 0 or model is None:
            return
        unwrapped = getattr(model, "module", model)
        epoch = int(round(state.epoch or 0))
        if epoch <= 0:
            return
        step = self.save_freq * epoch
        save_dir = save_checkpoint(unwrapped, self.tokenizer, self.output_dir, step)
        print(f"  [Save] Epoch {epoch}: wrote {save_dir}")


def train_grpo(args):
    from trl import GRPOConfig, GRPOTrainer

    model_path = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset = build_dataset(args.dataset, "train")
    eval_questions, eval_golds = load_jsonl_split(args.dataset, "test")

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(eval_questions)}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Grad accumulation steps: {args.gradient_accumulation_steps}")

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    config = GRPOConfig(
        output_dir=args.output_dir,
        run_name=f"grpo_{args.dataset}",
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=8,
        max_completion_length=args.max_new_tokens,
        learning_rate=args.lr,
        max_grad_norm=1.0,
        temperature=1.0,
        logging_steps=10,
        save_strategy="no",
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="wandb"
    )

    eval_cb = GreedyEvalCallback(
        tokenizer, eval_questions, eval_golds,
        args.dataset, args.max_new_tokens, args.eval_every,
    )
    save_cb = EpochCheckpointCallback(tokenizer, args.output_dir, args.save_freq)

    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=make_reward_fn(args.dataset),
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[eval_cb, save_cb],
    )

    trainer.train()
    print(f"Training complete. Checkpoints under {args.output_dir}")


def ppo_rollout(args, model, unwrapped_model, ref_model, tokenizer,
                query_tensors, query_attention_mask, gold_answers, dataset_name):
    model.eval()
    device = query_tensors.device

    with torch.no_grad():
        gen_output = unwrapped_model.generate(
            input_ids=query_tensors,
            attention_mask=query_attention_mask,
            top_k=0.0, top_p=1.0,
            temperature=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
        )
    completed_tensors = gen_output

    completed_texts = tokenizer.batch_decode(completed_tensors, skip_special_tokens=True)
    correctness = []
    for text, gold in zip(completed_texts, gold_answers):
        correctness.append(compute_rule_reward(text, gold, dataset_name))

    model_input_ids = completed_tensors
    model_attention_mask = (completed_tensors != tokenizer.pad_token_id).long()
    with torch.no_grad():
        lm_logits, _, val = model(input_ids=model_input_ids, attention_mask=model_attention_mask)
        old_logprob = logprobs_from_logits(lm_logits[:, :-1, :], model_input_ids[:, 1:])

        ref_logprob = None
        if ref_model is not None:
            ref_lm_logits, _, _ = ref_model(input_ids=model_input_ids, attention_mask=model_attention_mask)
            ref_logprob = logprobs_from_logits(ref_lm_logits[:, :-1, :], model_input_ids[:, 1:])

    prompt_len = query_tensors.size(1)
    mask = torch.zeros_like(model_input_ids, dtype=torch.bool)
    mask[:, prompt_len - 1: -1] = True

    score_rew = np.zeros(mask.shape)
    score_rew[:, -2] = np.array(correctness)

    # Left-padding can equal eos_token_id (Qwen has pad == eos), so only scan
    # the generation segment when looking for the first EOS.
    gen_eos = (model_input_ids[:, prompt_len:] == tokenizer.eos_token_id)
    has_eos = gen_eos.any(dim=1)
    first_eos_offset = gen_eos.int().argmax(dim=1)
    for bidx in range(model_input_ids.size(0)):
        if not has_eos[bidx]:
            continue
        tidx = prompt_len + first_eos_offset[bidx].item()
        mask[bidx][tidx:] = False
        score_rew[bidx][tidx:] = 0
        score_rew[bidx][tidx - 1] = correctness[bidx]

    kl_rew = None
    rew = score_rew
    if ref_logprob is not None:
        kl = old_logprob - ref_logprob
        kl = (kl.float() * mask[:, :-1]).cpu().numpy()
        kl_rew = np.zeros(mask.shape)
        kl_rew[:, :-1] = -kl
        rew = score_rew + args.kl_coef * kl_rew

    val = (val.float() * mask).cpu().numpy()
    gamma, lam = 0.95, 1.0
    adv = np.zeros_like(rew)
    for i in range(len(rew)):
        cur_rew, cur_val = rew[i], val[i]
        cur_delta = -cur_val[:-1] + cur_rew[:-1] + gamma * cur_val[1:]
        cur_adv = discount_cumsum(cur_delta, discount=gamma * lam)
        cur_adv[:prompt_len - 1] = 0
        adv[i][:-1] = cur_adv

    ret = adv + val

    mask = mask.to(device)
    dtype = old_logprob.dtype
    rew = torch.tensor(rew, device=device, dtype=dtype) * mask
    score_rew = torch.tensor(score_rew, device=device, dtype=dtype) * mask
    if kl_rew is not None:
        kl_rew = torch.tensor(kl_rew, device=device, dtype=dtype) * mask
    ret = torch.tensor(ret, device=device, dtype=dtype) * mask
    val = torch.tensor(val, device=device, dtype=dtype) * mask
    adv = torch.tensor(adv, device=device, dtype=dtype) * mask
    old_logprob = old_logprob * mask[:, :-1]

    model.train()
    return (model_input_ids, model_attention_mask, mask, rew, score_rew,
            kl_rew, ret, correctness, val, old_logprob, ref_logprob, adv)


def train_ppo(args):
    os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
    from trl.experimental.ppo import AutoModelForCausalLMWithValueHead
    from trl.experimental.ppo.ppo_trainer import masked_mean, masked_var
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    from transformers import get_constant_schedule

    def allgather_masked_whiten(values, mask, accelerator, shift_mean=True):
        all_values = accelerator.gather(values)
        all_mask = accelerator.gather(mask)
        mean = masked_mean(all_values, all_mask)
        var = masked_var(all_values, all_mask)
        whitened = (values - mean) * torch.rsqrt(var + 1e-8)
        if not shift_mean:
            whitened += mean
        return whitened

    model_path = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_questions, train_golds = load_jsonl_split(args.dataset, "train")
    eval_questions, eval_golds = load_jsonl_split(args.dataset, "test")

    train_prompts = [format_prompt(q) for q in train_questions]
    train_encodings = tokenizer(
        train_prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=args.max_prompt_length,
    )

    dataset_dicts = []
    for i in range(len(train_prompts)):
        dataset_dicts.append({
            "prefix": train_encodings.input_ids[i].tolist(),
            "prefix_attention_mask": train_encodings.attention_mask[i].tolist(),
            "gold_answer": train_golds[i],
        })

    def collate_fn(batch):
        max_prefix_len = max(len(item["prefix"]) for item in batch)
        prefix_padded, prefix_mask_padded, golds = [], [], []
        for item in batch:
            pad_len = max_prefix_len - len(item["prefix"])
            prefix_padded.append(
                [tokenizer.pad_token_id] * pad_len + item["prefix"])
            prefix_mask_padded.append([0] * pad_len + item["prefix_attention_mask"])
            golds.append(item["gold_answer"])
        return {
            "query_tensors": torch.LongTensor(prefix_padded),
            "query_attention_mask": torch.BoolTensor(prefix_mask_padded),
            "gold_answers": golds,
        }

    train_dataloader = DataLoader(
        dataset_dicts, shuffle=True, batch_size=args.batch_size,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    wandb.init(project=args.wandb_project, name=f"ppo_{args.dataset}", config=vars(args))

    accelerator = Accelerator(mixed_precision="bf16")
    accelerator.print(f"Accelerator: {accelerator.num_processes} process(es), "
                      f"device={accelerator.device}")

    accelerator.print(f"Loading policy model from {model_path}...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    accelerator.print(f"Loading ref model from {model_path}...")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    scheduler = get_constant_schedule(optimizer)

    accelerator.print("Preparing model with accelerator...")
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader,
    )
    if ref_model is not None:
        ref_model = ref_model.to(accelerator.device)

    global_step = 0
    global_iter_num = 0
    os.makedirs(args.output_dir, exist_ok=True)
    clip_grad_norm = 1.0

    n_epochs = args.n_epochs
    print(f"Starting PPO training: {n_epochs} epochs, "
          f"{len(train_dataloader)} batches/epoch")

    for epoch in tqdm(range(1, n_epochs + 1), total=n_epochs):
        model.train()
        epoch_stats = defaultdict(list)

        for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            model.eval()
            unwrapped_model = accelerator.unwrap_model(model)
            rollout_out = ppo_rollout(
                args, model, unwrapped_model, ref_model, tokenizer,
                query_tensors=batch["query_tensors"],
                query_attention_mask=batch["query_attention_mask"],
                gold_answers=batch["gold_answers"],
                dataset_name=args.dataset,
            )
            (model_input_ids, model_attention_mask, mask, rew, score_rew,
             kl_rew, ret, correctness, val, old_logprob, ref_logprob,
             adv) = rollout_out
            model.train()
            adv = allgather_masked_whiten(adv, mask, accelerator)

            batch_size_per_gpu = model_input_ids.size(0)
            mini_batch_size_per_gpu = args.mini_batch_size
            ppo_epochs = 2

            for _ in range(ppo_epochs):
                perms = torch.randperm(batch_size_per_gpu)
                for mini_idx in range(0, len(perms), mini_batch_size_per_gpu):
                    b_inds = perms[mini_idx: mini_idx + mini_batch_size_per_gpu]
                    cur_val = val[b_inds]
                    cur_old_logprob = old_logprob[b_inds]
                    cur_mask = mask[b_inds].float()
                    cur_rew = rew[b_inds]
                    cur_score_rew = score_rew[b_inds]
                    cur_kl_rew = None if kl_rew is None else kl_rew[b_inds]
                    cur_ret = ret[b_inds]
                    cur_adv = adv[b_inds]
                    cur_model_input_ids = model_input_ids[b_inds]
                    cur_model_attention_mask = model_attention_mask[b_inds]

                    resp_len_per_sample = torch.clamp(cur_mask.sum(dim=1), min=1.0)

                    lm_logits, _, vpreds = model(input_ids=cur_model_input_ids, attention_mask=cur_model_attention_mask)
                    logprob = logprobs_from_logits(lm_logits[:, :-1, :], cur_model_input_ids[:, 1:])

                    ratio = torch.exp(logprob - cur_old_logprob)
                    pg_losses = -cur_adv[:, :-1] * ratio
                    pg_losses2 = -cur_adv[:, :-1] * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                    pg_loss = ((torch.max(pg_losses, pg_losses2) * cur_mask[:, :-1]).sum(dim=-1) / resp_len_per_sample).mean()

                    vpredclipped = torch.clamp(vpreds, cur_val - 0.2, cur_val + 0.2)
                    vf_losses1 = (vpreds - cur_ret) ** 2
                    vf_losses2 = (vpredclipped - cur_ret) ** 2
                    vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2) * cur_mask).sum(dim=-1) / resp_len_per_sample).mean()

                    loss = pg_loss + vf_loss

                    accelerator.backward(loss)
                    if clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()
                    model.zero_grad()
                    optimizer.zero_grad()

                    n_correct = sum(1 for c in correctness if c == 1.0)
                    train_acc = n_correct / len(correctness)

                    vf_expl_var_num = masked_var(cur_ret - vpreds, cur_mask)
                    vf_expl_var_dem = masked_var(cur_ret, cur_mask)
                    vf_expl_var = 1.0 - vf_expl_var_num / (vf_expl_var_dem + 1e-8)
                    vf_expl_var = max(-1.0, vf_expl_var.item())

                    mean_reward = masked_mean(cur_rew, cur_mask).item()
                    mean_score_reward = masked_mean(cur_score_rew, cur_mask).item()
                    mean_kl_reward = 0.0 if cur_kl_rew is None else masked_mean(cur_kl_rew, cur_mask).item()

                    mean_seq_kl = -1.0
                    if cur_kl_rew is not None:
                        cur_kl_vals = -cur_kl_rew
                        seq_kl = (cur_kl_vals * cur_mask).sum(dim=1)
                        mean_seq_kl = seq_kl.mean().item()

                    epoch_stats["loss"].append(loss.item())
                    epoch_stats["pg_loss"].append(pg_loss.item())
                    epoch_stats["vf_loss"].append(vf_loss.item())
                    epoch_stats["acc"].append(train_acc)
                    epoch_stats["vf_expl_var"].append(vf_expl_var)

                    if accelerator.is_main_process:
                        wandb.log({
                            "loss/loss": loss.item(),
                            "loss/pg_loss": pg_loss.item(),
                            "loss/vf_loss": vf_loss.item(),
                            "acc/acc": train_acc,
                            "value/vf_expl_var": vf_expl_var,
                            "value/mean_reward": mean_reward,
                            "value/mean_score_reward": mean_score_reward,
                            "value/mean_kl_reward": mean_kl_reward,
                            "policy/mean_seq_kl": mean_seq_kl,
                            "nn/lr": scheduler.get_last_lr()[0],
                        }, step=global_iter_num)

                    global_iter_num += 1

            scheduler.step()
            global_step += 1

            if global_step % 10 == 0:
                avg_loss = np.mean(epoch_stats["loss"][-10:])
                avg_acc = np.mean(epoch_stats["acc"][-10:])
                print(f"  [E={epoch}, S={global_step}] "
                      f"loss={avg_loss:.4f} acc={avg_acc:.4f}")

            if global_step % args.eval_every == 0 and accelerator.is_main_process:
                eval_model = accelerator.unwrap_model(model).pretrained_model
                acc = evaluate(
                    eval_model, tokenizer,
                    eval_questions, eval_golds,
                    args.dataset, args.max_new_tokens,
                )
                print(f"  [Eval] Step {global_step}: accuracy = {acc:.4f}")
                if wandb.run is not None:
                    wandb.log({"eval/accuracy": acc}, step=global_iter_num)

        summary = {k: np.mean(v) for k, v in epoch_stats.items()}
        print(f"[Epoch {epoch}/{n_epochs}] " +
              " ".join(f"{k}={v:.4f}" for k, v in summary.items()))

        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model).pretrained_model
            step = args.save_freq * epoch
            save_dir = save_checkpoint(unwrapped, tokenizer, args.output_dir, step)
            print(f"  [Save] Epoch {epoch}: wrote {save_dir}")

    print(f"Training complete. Checkpoints under {args.output_dir}")


def train_self_training(args):
    from train_RL.train_self_training import run_self_training
    run_self_training(args)


def main():
    args = parse_args()
    print(f"Method:  {args.method}")
    print(f"Model:   {args.model_name}")
    print(f"Dataset: {args.dataset}")

    if args.method == "grpo":
        train_grpo(args)
    elif args.method == "ppo":
        train_ppo(args)
    elif args.method == "self_training":
        mode = "online" if args.steps_per_generation == 1 else "offline"
        print(f"Mode: {mode} (steps_per_generation={args.steps_per_generation})")
        print(f"Rewards: correct={args.pos_reward}, incorrect={args.neg_reward}")
        train_self_training(args)


if __name__ == "__main__":
    main()
