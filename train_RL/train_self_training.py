import argparse
import os
import sys
import random
import wandb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generate_data.utils import format_prompt
from train_RL.utils import reward_weighted_loss, build_masks, evaluate, load_jsonl_split, save_checkpoint
from train_RL.reward import compute_batch_rewards


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--sft_checkpoint", type=str, default=None)
    p.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "svamp", "math"])

    p.add_argument("--pos_reward", type=float, default=1.0)
    p.add_argument("--neg_reward", type=float, default=-1.0)

    p.add_argument("--num_steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_generations", type=int, default=1)
    p.add_argument("--steps_per_generation", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--save_freq", type=int, default=226)
    p.add_argument("--output_dir", type=str, default="outputs/self_training")
    p.add_argument("--wandb_project", type=str, default="reproducing-reft")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")

    return p.parse_args()


def generate_rollouts(model, tokenizer, prompts, gold_answers, args, device):
    encodings = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=args.max_prompt_length,
    )
    prompt_ids = encodings.input_ids.to(device)
    prompt_mask = encodings.attention_mask.to(device)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    model.train()

    prompt_len = prompt_ids.shape[1]
    attn_mask, response_mask = build_masks(
        generated_ids, prompt_len, prompt_mask, tokenizer.pad_token_id, device,
    )

    binary_rewards = compute_batch_rewards(
        generated_ids, gold_answers, tokenizer, args.dataset,
        prompt_len, response_mask, reward_type="standard",
    )
    rewards = torch.where(
        binary_rewards == 1.0,
        torch.tensor(args.pos_reward, device=device),
        torch.tensor(args.neg_reward, device=device),
    )

    return generated_ids, attn_mask, response_mask, rewards


def run_self_training(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = "online" if args.steps_per_generation == 1 else "offline"
    print(f"Mode: {mode} (steps_per_generation={args.steps_per_generation})")
    print(f"Rewards: correct={args.pos_reward}, incorrect={args.neg_reward}")

    wandb.init(project=args.wandb_project, config=vars(args), tags=["self-training", mode])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    checkpoint = args.sft_checkpoint or args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16,
    ).to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_questions, train_golds = load_jsonl_split(args.dataset, "train")
    eval_questions, eval_golds = load_jsonl_split(args.dataset, "test")
    num_train = len(train_questions)
    print(f"Train: {num_train} | Eval: {len(eval_questions)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    rollout = None

    for step in range(1, args.num_steps + 1):
        need_new_rollout = (rollout is None) or ((step - 1) % args.steps_per_generation == 0)

        if need_new_rollout:
            indices = random.sample(range(num_train), args.batch_size)

            prompts = []
            golds = []
            for i in indices:
                for _ in range(args.num_generations):
                    prompts.append(format_prompt(train_questions[i]))
                    golds.append(train_golds[i])

            generated_ids, attn_mask, response_mask, rewards = generate_rollouts(
                model, tokenizer, prompts, golds, args, device,
            )
            rollout = (generated_ids, attn_mask, response_mask, rewards)
        else:
            generated_ids, attn_mask, response_mask, rewards = rollout

        logits = model(generated_ids, attention_mask=attn_mask).logits
        loss = reward_weighted_loss(logits, generated_ids, response_mask, rewards)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        mean_reward = rewards.mean().item()
        accuracy = (rewards == args.pos_reward).float().mean().item()

        if step % 10 == 0 or step == 1:
            print(
                f"Step {step}/{args.num_steps} | "
                f"loss={loss.item():.4f} | reward={mean_reward:.3f} | "
                f"acc={accuracy:.3f}"
            )
        wandb.log({
            "step": step,
            "reward/mean": mean_reward,
            "reward/accuracy": accuracy,
            "loss": loss.item(),
        }, step=step)

        if step % args.eval_every == 0:
            eval_acc = evaluate(
                model, tokenizer, eval_questions, eval_golds,
                args.dataset, args.max_new_tokens, device=device,
            )
            print(f"  Eval accuracy: {eval_acc:.4f}")
            wandb.log({"eval/accuracy": eval_acc}, step=step)

        if step % args.save_freq == 0:
            save_dir = save_checkpoint(model, tokenizer, args.output_dir, step)
            print(f"  Saved checkpoint to {save_dir}")

    print(f"Training complete. Checkpoints under {args.output_dir}")

    wandb.finish()


def main():
    args = parse_args()
    run_self_training(args)


if __name__ == "__main__":
    main()
