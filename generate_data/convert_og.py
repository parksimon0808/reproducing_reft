import json
import os

from utils import format_prompt

def convert_data(idx, d):
    new_d = {
        "index": idx,
        "question": d["question"],
        "gold_answer": d["answer_value"],
        "prompt": format_prompt(d["question"]),
        "response": d["answer_cot"],
        "is_correct": True
    }

    return new_d

def convert_dataset(dataset):
    with open(f"outputs/{dataset}_nl.json") as f:
        data = json.load(f)

    new_data = [convert_data(idx, d) for idx, d in enumerate(data)]

    with open(f"outputs/{dataset}_og.jsonl", "w") as f:
        for record in new_data:
            f.write(json.dumps(record) + "\n")

def main():
    for dataset in ["gsm8k", "svamp"]:
        convert_dataset(dataset)


if __name__ == "__main__":
    main()