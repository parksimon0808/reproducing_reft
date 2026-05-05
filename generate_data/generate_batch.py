import argparse
import json
from pathlib import Path

from collections import Counter

from utils import load_dataset_split, format_prompt

MAX_REQUESTS_PER_FILE = 50000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "svamp", "mathqa", "math"])
    parser.add_argument("--model", type=str, default="gpt-5-nano")
    parser.add_argument("--reasoning_effort", type=str, default="none")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_responses", type=int, default=8)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--debug", action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    questions, gold_answers = load_dataset_split(args.dataset, split="train")

    if args.debug:
        questions = questions[:1]
        gold_answers = gold_answers[:1]

    # Count existing correct responses per question index
    correct_counts = Counter()
    if args.attempt > 1:
        existing_output = Path("outputs") / f"{args.dataset}.jsonl"
        if existing_output.exists():
            with open(existing_output) as f:
                for line in f:
                    record = json.loads(line)
                    if record["is_correct"]:
                        correct_counts[record["index"]] += 1

    output_dir = Path("batch_inputs") / f"attempt_{args.attempt}"
    output_dir.mkdir(parents=True, exist_ok=True)

    file_index = 1
    request_count = 0
    total_requests = 0
    f = open(output_dir / f"{args.dataset}_{file_index}.jsonl", "w")

    for i, (question, gold) in enumerate(zip(questions, gold_answers)):
        num_correct = correct_counts[i]
        if num_correct >= args.num_responses:
            continue

        num_to_generate = args.num_responses - num_correct + 2
        prompt = format_prompt(question)

        for j in range(num_to_generate):
            if request_count >= MAX_REQUESTS_PER_FILE:
                f.close()
                file_index += 1
                f = open(output_dir / f"{args.dataset}_{file_index}.jsonl", "w")
                request_count = 0

            request = {
                "custom_id": f"{args.dataset}-{i}-{j}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": args.temperature,
                    "max_completion_tokens": args.max_tokens,
                },
            }
            f.write(json.dumps(request) + "\n")
            request_count += 1
            total_requests += 1

    f.close()

    print(f"Wrote {total_requests} requests across {file_index} files")


if __name__ == "__main__":
    main()
