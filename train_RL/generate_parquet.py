import argparse
import json
import os

import pandas as pd


def _resolve_jsonl(dataset, split):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    suffix = "final" if split == "train" else "test"
    return os.path.join(project_root, "generate_data", "outputs",
                        f"{dataset}_{suffix}.jsonl")


def build_rows(dataset_name, split):
    path = _resolve_jsonl(dataset_name, split)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Expected SFT JSONL at {path}. Run generate_data first.")

    rows = []
    with open(path, "r") as f:
        lines = f.readlines()
        n = 8 if split == "train" else 1
        for i, line in enumerate(lines[::n]):
            item = json.loads(line)
            rows.append({
                "data_source": dataset_name,
                "prompt": [{"role": "user", "content": item["prompt"]}],
                "ability": "math",
                "reward_model": {
                    "ground_truth": item["gold_answer"],
                    "style": "rule",
                },
                "extra_info": {
                    "index": item.get("index", i),
                    "question": item["question"],
                    "split": split,
                },
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["gsm8k", "svamp", "math", "mathqa"])
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split, filename in [("train", "train.parquet"), ("test", "test.parquet")]:
        rows = build_rows(args.dataset, split)
        df = pd.DataFrame(rows)
        out = os.path.join(args.output_dir, filename)
        df.to_parquet(out, index=False)
        print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
