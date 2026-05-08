import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_data.utils import extract_answer, verify_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError("No records found in input file.")

    config_name = records[0]["config_name"]
    benchmark = records[0]["benchmark"]

    print(f"Config:     {config_name}")
    print(f"Benchmark:  {benchmark}")
    print(f"Records:    {len(records)}")

    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["prompt"]].append(rec)

    num_problems = len(grouped)
    samples_per_problem = max(len(v) for v in grouped.values())
    print(f"Problems:   {num_problems}  |  Samples per problem: {samples_per_problem}")

    per_problem_accuracies = []
    per_problem_details = []
    num_solved = 0

    for prompt, recs in grouped.items():
        correct_count = 0
        details = []
        gold = recs[0]["gold_answer"]

        for rec in recs:
            pred = extract_answer(rec["response"], benchmark)
            correct = verify_answer(pred, gold, benchmark)
            correct_count += int(correct)
            details.append({"pred": pred, "correct": correct})

        prob_acc = correct_count / len(recs)
        per_problem_accuracies.append(prob_acc)
        if correct_count > 0:
            num_solved += 1
        per_problem_details.append({
            "prompt": prompt,
            "gold": gold,
            "samples": details,
            "problem_accuracy": prob_acc,
            "solved": correct_count > 0,
        })

    overall_accuracy = sum(per_problem_accuracies) / num_problems
    pass_at_k = num_solved / num_problems

    print(f"\nAccuracy:   {overall_accuracy:.4f}  ({overall_accuracy * 100:.2f}%)")
    print(f"Pass@{samples_per_problem}:     {pass_at_k:.4f}  ({pass_at_k * 100:.2f}%)")

    output_dir = input_path.parent
    results = {
        "config_name": config_name,
        "benchmark": benchmark,
        "num_problems": num_problems,
        "samples_per_problem": samples_per_problem,
        "accuracy": overall_accuracy,
        f"pass@{samples_per_problem}": pass_at_k,
        "per_problem": per_problem_details,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
