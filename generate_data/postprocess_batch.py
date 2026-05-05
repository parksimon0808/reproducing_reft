import argparse
import json
from pathlib import Path

from utils import (
    load_dataset_split,
    format_prompt,
    extract_answer,
    verify_answer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "svamp", "math"])
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()

    questions, gold_answers = load_dataset_split(args.dataset, split="train")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.dataset}.jsonl"

    existing_records = []
    if output_file.exists():
        with open(output_file) as f:
            existing_records = [json.loads(line) for line in f]

    for record in existing_records:
        extracted = extract_answer(record["response"], args.dataset)
        record["is_correct"] = bool(verify_answer(extracted, record["gold_answer"], args.dataset))

    existing_keys = {(r["index"], r["response"]) for r in existing_records}

    batch_output_file = Path("batch_outputs") / f"attempt_{args.attempt}" / f"{args.dataset}.jsonl"
    with open(batch_output_file) as f:
        batch_results = [json.loads(line) for line in f]

    responses_by_index = {}
    for result in batch_results:
        custom_id = result["custom_id"]
        parts = custom_id.rsplit("-", 2)
        question_index = int(parts[-2])
        response_content = result["response"]["body"]["choices"][0]["message"]["content"]
        responses_by_index.setdefault(question_index, []).append(response_content)

    new_records = []
    num_skipped = 0
    for i in sorted(responses_by_index.keys()):
        question = questions[i]
        gold = gold_answers[i]
        prompt = format_prompt(question)

        for response in responses_by_index[i]:
            if (i, response) in existing_keys:
                num_skipped += 1
                continue

            extracted = extract_answer(response, args.dataset)
            is_correct = bool(verify_answer(extracted, gold, args.dataset))

            record = {
                "index": i,
                "question": question,
                "gold_answer": gold,
                "prompt": prompt,
                "response": response,
                "is_correct": is_correct,
            }
            new_records.append(record)
            existing_keys.add((i, response))

    all_records = existing_records + new_records
    with open(output_file, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
