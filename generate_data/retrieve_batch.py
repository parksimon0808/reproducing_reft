import argparse
import json
from pathlib import Path

from openai import OpenAI


def get_latest_batch_ids(dataset, attempt):
    log_file = Path("batch_logs") / f"attempt_{attempt}" / f"{dataset}.jsonl"
    if not log_file.exists():
        raise FileNotFoundError()

    with open(log_file) as f:
        lines = f.read().strip().split("\n")
    entry = json.loads(lines[-1])
    return [b["batch_id"] for b in entry["batches"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "svamp", "math"])
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()

    batch_ids = get_latest_batch_ids(args.dataset, args.attempt)

    client = OpenAI()

    all_completed = True
    for batch_id in batch_ids:
        batch = client.batches.retrieve(batch_id)
        print(
            f"Batch {batch.id}: status={batch.status}  "
            f"Completed: {batch.request_counts.completed}/{batch.request_counts.total}, "
            f"Failed: {batch.request_counts.failed}"
        )
        if batch.status != "completed":
            all_completed = False

    if not all_completed:
        print("Not all batches are completed. Try again later.")
        return

    output_dir = Path("batch_outputs") / f"attempt_{args.attempt}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.dataset}.jsonl"

    with open(output_file, "wb") as out:
        for batch_id in batch_ids:
            batch = client.batches.retrieve(batch_id)

            if batch.error_file_id:
                error_file = output_dir / f"{args.dataset}_errors.jsonl"
                error_content = client.files.content(batch.error_file_id)
                with open(error_file, "ab") as ef:
                    ef.write(error_content.read())
                print(f"Downloaded errors for {batch_id} to {error_file}")

            try:
                content = client.files.content(batch.output_file_id)
                out.write(content.read())
            except:
                print("No output generated")
                continue

    print(f"Downloaded output to {output_file}")


if __name__ == "__main__":
    main()
