import argparse
import json
from datetime import datetime
from pathlib import Path

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "svamp", "mathqa", "math"])
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()

    input_dir = Path("batch_inputs") / f"attempt_{args.attempt}"
    input_files = sorted(input_dir.glob(f"{args.dataset}_*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"No batch input files found for {args.dataset}")

    client = OpenAI()

    log_dir = Path("batch_logs") / f"attempt_{args.attempt}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{args.dataset}.jsonl"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    batches = []
    for input_file in input_files:
        file_obj = client.files.create(
            file=open(input_file, "rb"),
            purpose="batch",
        )

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        batches.append({
            "batch_id": batch.id,
            "input_file_id": file_obj.id,
            "input_file": input_file.name,
        })

        print(f"Submitted {input_file.name} -> batch {batch.id}")

    log_entry = {
        "created_at": timestamp,
        "dataset": args.dataset,
        "batches": batches,
    }
    with open(log_file, "a") as lf:
        lf.write(json.dumps(log_entry) + "\n")

    print(f"Submitted {len(input_files)} batches. Logged to {log_file}")


if __name__ == "__main__":
    main()
