import json
from collections import defaultdict
from pathlib import Path

from utils import load_dataset_split, format_prompt

DATASETS = ["gsm8k", "svamp", "math"]
NUM_RESPONSES = 8
NUM_TEST_EXAMPLES = 100
OUTPUT_DIR = Path(__file__).parent / "outputs"


def write_train_split(dataset):
    input_file = OUTPUT_DIR / f"{dataset}.jsonl"
    output_file = OUTPUT_DIR / f"{dataset}_final.jsonl"
    discarded_file = OUTPUT_DIR / f"{dataset}_discarded.txt"

    with open(input_file) as f:
        records = [json.loads(line) for line in f]

    correct_by_index = defaultdict(list)
    for record in records:
        if record["is_correct"]:
            correct_by_index[record["index"]].append(record)

    all_indices = sorted({r["index"] for r in records})

    final_records = []
    discarded_indices = []

    for idx in all_indices:
        correct = correct_by_index[idx]
        if len(correct) >= NUM_RESPONSES:
            final_records.extend(correct[:NUM_RESPONSES])
        else:
            discarded_indices.append(idx)

    with open(output_file, "w") as f:
        for record in final_records:
            f.write(json.dumps(record) + "\n")

    with open(discarded_file, "w") as f:
        for idx in discarded_indices:
            f.write(f"{idx}\n")

    num_kept = len(all_indices) - len(discarded_indices)
    print(
        f"{dataset}: train {num_kept}/{len(all_indices)} questions kept "
        f"({len(final_records)} total responses), "
        f"{len(discarded_indices)} discarded"
    )


def write_test_split(dataset):
    output_file = OUTPUT_DIR / f"{dataset}_test.jsonl"

    if dataset in ("gsm8k", "svamp"):
        questions, golds = load_dataset_split(dataset, split="test")
    elif dataset == "math":
        discarded_file = OUTPUT_DIR / "math_discarded.txt"
        with open(discarded_file) as f:
            discarded_indices = [int(line.strip()) for line in f if line.strip()]
        all_questions, all_golds = load_dataset_split(dataset, split="train")
        questions = [all_questions[i] for i in discarded_indices]
        golds = [all_golds[i] for i in discarded_indices]

    questions = questions[:NUM_TEST_EXAMPLES]
    golds = golds[:NUM_TEST_EXAMPLES]

    with open(output_file, "w") as f:
        for i, (q, g) in enumerate(zip(questions, golds)):
            record = {
                "index": i,
                "question": q,
                "gold_answer": g,
                "prompt": format_prompt(q),
            }
            f.write(json.dumps(record) + "\n")

    print(f"{dataset}: test  wrote {len(questions)} records → {output_file.name}")


def main():
    for dataset in DATASETS:
        write_train_split(dataset)
        write_test_split(dataset)


if __name__ == "__main__":
    main()
