import argparse
import json
import re
from pathlib import Path

import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams


_REPO_ROOT = Path(__file__).resolve().parents[1]

_RL_SAVE_FREQ = {"gsm8k": 226, "svamp": 21, "math": 342}


def load_test_records(benchmark):
    path = _REPO_ROOT / "generate_data" / "outputs" / f"{benchmark}_test.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def derive_labels(model_path):
    p = Path(model_path).resolve()
    parts = p.parts

    if "sft" in parts and "outputs" in parts:
        m = re.search(r"epoch_(\d+)", p.name)
        if m is None:
            return None, None
        epoch = int(m.group(1))
        exp_name = p.parent.name
        for ds in ("gsm8k", "svamp", "math"):
            suffix = f"_{ds}"
            if exp_name.endswith(suffix):
                model_base = exp_name[: -len(suffix)]
                return model_base, f"sft_epoch{epoch}"
        return exp_name, f"sft_epoch{epoch}"

    if "train_RL" in parts and "outputs" in parts:
        m = re.search(r"global_step_(\d+)", p.name)
        if m is None:
            return None, None
        step = int(m.group(1))
        exp_name = p.parent.name
        for ds in ("gsm8k", "svamp", "math"):
            marker = f"_{ds}_"
            idx = exp_name.find(marker)
            if idx > 0:
                model_base = exp_name[:idx]
                variant = exp_name[idx + len(marker):]
                save_freq = _RL_SAVE_FREQ.get(ds)
                if save_freq and step % save_freq == 0:
                    return model_base, f"{variant}_epoch{step // save_freq}"
                return model_base, f"{variant}_step{step}"
        return exp_name, f"step{step}"

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_base_name", default=None)
    parser.add_argument("--method_label", default=None)
    parser.add_argument("--output_root", default="output")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.model_path:
        config["model"] = args.model_path

    config_name = config_path.stem
    benchmark = config["benchmark"]

    model_base_name = args.model_base_name
    method_label = args.method_label
    if args.model_path and (model_base_name is None or method_label is None):
        derived_base, derived_label = derive_labels(args.model_path)
        if model_base_name is None:
            model_base_name = derived_base
        if method_label is None:
            method_label = derived_label
    if model_base_name is None:
        model_base_name = Path(config["model"]).name or "unknown_model"
    if method_label is None:
        method_label = "base"

    output_dir = Path(args.output_root) / model_base_name / method_label / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "generations.jsonl"

    print(f"Config:          {config_name}")
    print(f"Model:           {config['model']}")
    print(f"Benchmark:       {benchmark}")
    print(f"Model base:      {model_base_name}")
    print(f"Method label:    {method_label}")
    print(f"Samples/prompt:  {config['num_samples']}")
    print(f"Output:          {output_file}")

    records = load_test_records(benchmark)
    if args.max_examples is not None:
        records = records[: args.max_examples]
    print(f"Dataset size:    {len(records)} examples")

    prompts = [r["prompt"] for r in records]
    gold_answers = [r["gold_answer"] for r in records]

    sampling_params = SamplingParams(
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_new_tokens"],
        n=config["num_samples"],
    )

    llm = LLM(model=config["model"], tensor_parallel_size=1)

    print("Running inference …")
    outputs = llm.generate(prompts, sampling_params)

    with open(output_file, "w") as f:
        for i, output in enumerate(tqdm(outputs, desc="Writing")):
            for completion in output.outputs:
                record = {
                    "prompt": prompts[i],
                    "response": completion.text,
                    "gold_answer": gold_answers[i],
                    "benchmark": benchmark,
                    "config_name": config_name,
                    "model_base_name": model_base_name,
                    "method_label": method_label,
                }
                f.write(json.dumps(record) + "\n")

    print(f"Saved {len(outputs) * config['num_samples']} records to {output_file}")


if __name__ == "__main__":
    main()
