import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve()
OUTPUT_ROOT = ROOT / "output"
PLOTS_ROOT = ROOT / "newPlotting"

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red

STANDARD_METRICS = [
    ("gsm8k_avg4", "accuracy", "gsm8k avg4 accuracy", "GSM8K avg4 accuracy"),
    ("gsm8k_avg4", "pass@4", "gsm8k avg4 pass@4", "GSM8K avg4 pass@4"),
    ("gsm8k_zeroshot", "accuracy", "gsm8k zeroshot accuracy", "GSM8K zeroshot accuracy"),
    ("math_avg4", "accuracy", "math avg4 accuracy", "MATH avg4 accuracy"),
    ("math_avg4", "pass@4", "math avg4 pass@4", "MATH avg4 pass@4"),
    ("math_zeroshot", "accuracy", "math zeroshot accuracy", "MATH zeroshot accuracy"),
    ("svamp_avg4", "accuracy", "svamp avg4 accuracy", "SVAMP avg4 accuracy"),
    ("svamp_avg4", "pass@4", "svamp avg4 pass@4", "SVAMP avg4 pass@4"),
    ("svamp_zeroshot", "accuracy", "svamp zeroshot accuracy", "SVAMP zeroshot accuracy"),
]

GSM8K_COMPARISON_METRICS = [
    ("gsm8k_og_avg4", "gsm8k_avg4", "accuracy", "gsm8k avg4 accuracy", "GSM8K avg4 accuracy"),
    ("gsm8k_og_avg4", "gsm8k_avg4", "pass@4", "gsm8k avg4 pass@4", "GSM8K avg4 pass@4"),
    ("gsm8k_og_zeroshot", "gsm8k_zeroshot", "accuracy", "gsm8k zeroshot accuracy", "GSM8K zeroshot accuracy"),
]

FINAL_RESULTS_SECTIONS = [
    (
        "section1",
        ["sft", "ppo_warmup"],
        {"sft": "SFT", "ppo_warmup": "ReFT (PPO)"},
    ),
    (
        "section2",
        ["grpo", "grpo_warmup", "ppo", "ppo_warmup"],
        {
            "grpo": "ReFT (GRPO, no warm-up)",
            "grpo_warmup": "ReFT (GRPO)",
            "ppo": "ReFT (PPO, no warm-up)",
            "ppo_warmup": "ReFT (PPO)",
        },
    ),
]

FILTERED_SECTIONS = [
    (
        "section1",
        ["sft", "sft_og", "grpo_warmup", "ppo_warmup"],
        {"sft": "SFT", "sft_og": "SFT OG", "grpo_warmup": "ReFT (GRPO)", "ppo_warmup": "ReFT (PPO)"},
    ),
    (
        "section2",
        ["grpo", "grpo_warmup", "ppo", "ppo_warmup"],
        {"grpo": "GRPO", "grpo_warmup": "ReFT (GRPO)", "ppo": "PPO", "ppo_warmup": "ReFT (PPO)"},
    ),
]

SELF_TRAINING_METHODS = ["online_st", "online_st_og", "offline_st", "offline_st_og"]
SELF_TRAINING_LABELS = {
    "online_st": "Online (r=-1)",
    "online_st_og": "Online (r=0)",
    "offline_st": "Offline (r=-1)",
    "offline_st_og": "Offline (r=0)",
}

ALL_METHODS_SPECS = [
    # (method_dir, use_og_config, label)
    ("sft",           False, "SFT"),
    ("sft",           True,  "SFT (original)"),
    ("grpo_warmup",   False, "ReFT (GRPO)"),
    ("ppo_warmup",    False, "ReFT (PPO)"),
    ("grpo",          False, "ReFT (GRPO, no warm-up)"),
    ("ppo",           False, "ReFT (PPO, no warm-up)"),
    ("online_st",     False, "Online (r=-1)"),
    ("online_st_og",  False, "Online (r=0)"),
    ("offline_st",    False, "Offline (r=-1)"),
    ("offline_st_og", False, "Offline (r=0)"),
]
ALL_METHODS_BENCHMARKS = ["svamp", "gsm8k", "math"]
ALL_METHODS_METRICS = [
    ("avg4", "accuracy", "avg4 accuracy"),
    ("avg4", "pass@4", "avg4 pass@4"),
    ("zeroshot", "accuracy", "zeroshot accuracy"),
]
ALL_METHODS_MODELS = ["Qwen3-1.7B-Base", "Qwen3.5-0.8B-Base", "Qwen3.5-2B-Base"]

ABLATION_TAG_RE = re.compile(r"^grpo_ablation_(?P<tag>.+?)_epoch(?P<epoch>\d+)$")
ABLATION_AXIS_ORDER = ["kl", "lr", "K", "clip"]
ABLATION_AXIS_CMAPS = {"kl": "Blues", "lr": "Oranges", "K": "Greens", "clip": "Purples"}
ABLATION_BENCHMARK_METRICS = [
    ("avg4", "accuracy", "avg4 accuracy"),
    ("avg4", "pass@4", "avg4 pass@4"),
    ("zeroshot", "accuracy", "zeroshot accuracy"),
]


def parse_epoch(directory_name: str):
    match = re.search(r"_epoch(\d+)$", directory_name)
    return int(match.group(1)) if match else None


def load_scores(model_base: str, method: str, config_name: str):
    scores = []
    model_dir = OUTPUT_ROOT / model_base
    if not model_dir.exists():
        return scores
    for method_dir in sorted(model_dir.glob(f"{method}_epoch*")):
        if not method_dir.is_dir():
            continue
        epoch = parse_epoch(method_dir.name)
        if epoch is None:
            continue
        results_path = method_dir / config_name / "results.json"
        if not results_path.exists():
            continue
        try:
            with open(results_path) as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        scores.append((epoch, results))
    scores.sort(key=lambda item: item[0])
    return scores


def plot_metric(model_base: str, config_name: str, metric_key: str, title: str, ylabel: str, save_dir: Path, methods: list[str], label_map: dict[str, str]):
    method_lines = {}
    for method in methods:
        scores = load_scores(model_base, method, config_name)
        if not scores:
            continue
        epochs, values = [], []
        for epoch, results in scores:
            if metric_key not in results:
                continue
            epochs.append(epoch)
            values.append(results[metric_key])
        if epochs and values:
            method_lines[method] = (epochs, values)

    if not method_lines:
        print(f"  No data: {config_name}/{metric_key}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    all_values = []
    for method, (epochs, values) in method_lines.items():
        color = COLORS[methods.index(method) % len(COLORS)]
        plt.plot(epochs, values, marker="o", label=label_map.get(method, method), color=color)
        all_values.extend(values)

    plt.title(f"{model_base} — {title}")
    plt.xlabel("Epoch")
    plt.xticks(range(1, 11))
    ax = plt.gca()
    ax.set_xlim(0.8, 10.2)
    ax.margins(x=0.02)
    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        padding = max(0.03, (y_max - y_min) * 0.15)
        ax.set_ylim(max(0.0, y_min - padding), min(1.0, y_max + padding))
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_file = save_dir / f"{config_name}_{metric_key.replace('@', 'at')}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"  Saved: {output_file}")


def plot_metric_mixed_configs(model_base: str, sft_config: str, ppo_config: str, metric_key: str, title: str, ylabel: str, save_dir: Path):
    methods_configs = [("sft", sft_config), ("ppo_warmup", ppo_config)]
    label_map = {"sft": "SFT (original)", "ppo_warmup": "ReFT (PPO)"}
    method_lines = {}
    for method, config_name in methods_configs:
        scores = load_scores(model_base, method, config_name)
        if not scores:
            continue
        epochs, values = [], []
        for epoch, results in scores:
            if metric_key not in results:
                continue
            epochs.append(epoch)
            values.append(results[metric_key])
        if epochs and values:
            method_lines[method] = (epochs, values)

    if not method_lines:
        print(f"  No data: {ppo_config}/{metric_key}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    all_values = []
    ordered_methods = [m for m, _ in methods_configs]
    for method, (epochs, values) in method_lines.items():
        color = COLORS[ordered_methods.index(method) % len(COLORS)]
        plt.plot(epochs, values, marker="o", label=label_map.get(method, method), color=color)
        all_values.extend(values)

    plt.title(f"{model_base} — {title}")
    plt.xlabel("Epoch")
    plt.xticks(range(1, 11))
    ax = plt.gca()
    ax.set_xlim(0.8, 10.2)
    ax.margins(x=0.02)
    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        padding = max(0.03, (y_max - y_min) * 0.15)
        ax.set_ylim(max(0.0, y_min - padding), min(1.0, y_max + padding))
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_file = save_dir / f"{ppo_config}_{metric_key.replace('@', 'at')}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"  Saved: {output_file}")


def run_final_results(model_base: str = "Qwen3-1.7B-Base"):
    print("\n=== Final Results ===")
    save_root = PLOTS_ROOT / "final plots"
    for section_name, methods, label_map in FINAL_RESULTS_SECTIONS:
        section_dir = save_root / section_name
        for config_name, metric_key, title, ylabel in STANDARD_METRICS:
            plot_metric(model_base, config_name, metric_key, title, ylabel, section_dir, methods, label_map)


def run_gsm8k_comparison(model_base: str = "Qwen3-1.7B-Base"):
    print("\n=== GSM8K Comparison ===")
    save_dir = PLOTS_ROOT / "GSM8K_comparison"
    for sft_config, ppo_config, metric_key, title, ylabel in GSM8K_COMPARISON_METRICS:
        plot_metric_mixed_configs(model_base, sft_config, ppo_config, metric_key, title, ylabel, save_dir)


def run_qwen3_filtered(model_base: str = "Qwen3-1.7B-Base"):
    print("\n=== Qwen3-1.7B Filtered ===")
    save_root = PLOTS_ROOT / "qwen3-1.7B_filtered"
    for section_name, methods, label_map in FILTERED_SECTIONS:
        section_dir = save_root / section_name
        for config_name, metric_key, title, ylabel in STANDARD_METRICS:
            plot_metric(model_base, config_name, metric_key, title, ylabel, section_dir, methods, label_map)


def run_self_training(model_base: str = "Qwen3-1.7B-Base"):
    print("\n=== Self Training ===")
    save_dir = PLOTS_ROOT / "(self training)"
    for config_name, metric_key, title, ylabel in STANDARD_METRICS:
        plot_metric(model_base, config_name, metric_key, title, ylabel, save_dir, SELF_TRAINING_METHODS, SELF_TRAINING_LABELS)


def parse_ablation_axis(tag: str):
    for axis in ABLATION_AXIS_ORDER:
        prefix = axis + "_"
        if tag.startswith(prefix):
            return axis, tag[len(prefix):]
    return None, tag


def sort_ablation_value(value_str: str) -> float:
    try:
        return float(value_str)
    except ValueError:
        try:
            return float(value_str.split("_")[-1])
        except ValueError:
            return 0.0


def ablation_label(axis: str, value_str: str) -> str:
    if axis == "clip":
        return f"clip={value_str.replace('_', '/')}"
    return f"{axis}={value_str}"


def discover_ablation_tags(model_base: str):
    model_dir = OUTPUT_ROOT / model_base
    if not model_dir.exists():
        return []
    tags = set()
    for d in model_dir.iterdir():
        if not d.is_dir():
            continue
        m = ABLATION_TAG_RE.match(d.name)
        if m:
            tags.add(m["tag"])
    return sorted(tags)


def load_ablation_scores(model_base: str, tag: str, config_name: str):
    scores = []
    model_dir = OUTPUT_ROOT / model_base
    if not model_dir.exists():
        return scores
    for d in sorted(model_dir.glob(f"grpo_ablation_{tag}_epoch*")):
        if not d.is_dir():
            continue
        epoch = parse_epoch(d.name)
        if epoch is None:
            continue
        results_path = d / config_name / "results.json"
        if not results_path.exists():
            continue
        try:
            with open(results_path) as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        scores.append((epoch, results))
    scores.sort(key=lambda item: item[0])
    return scores


def plot_ablations_combined(model_base: str, benchmark: str, save_dir: Path):
    tags = discover_ablation_tags(model_base)
    if not tags:
        print(f"  No ablations: {model_base}")
        return

    by_axis: dict[str, list] = defaultdict(list)
    for tag in tags:
        axis, value_str = parse_ablation_axis(tag)
        if axis is None:
            continue
        by_axis[axis].append((sort_ablation_value(value_str), tag, value_str))
    for axis in by_axis:
        by_axis[axis].sort()

    # Line specs: (label, kind, identifier, color, linestyle, marker)
    lines = [("default", "method", "grpo_warmup", "black", "--", "s")]
    for axis in ABLATION_AXIS_ORDER:
        items = by_axis.get(axis, [])
        if not items:
            continue
        cmap = plt.get_cmap(ABLATION_AXIS_CMAPS[axis])
        n = len(items)
        for i, (_, tag, value_str) in enumerate(items):
            t = 0.45 + 0.5 * (i / max(n - 1, 1))
            lines.append((ablation_label(axis, value_str), "ablation", tag, cmap(t), "-", "o"))

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    legend_handles, legend_labels, seen = [], [], set()
    drew_anything = False

    for idx, (suffix, metric_key, metric_name) in enumerate(ABLATION_BENCHMARK_METRICS):
        ax = axes[idx]
        config_name = f"{benchmark}_{suffix}"
        title = f"{benchmark} {metric_name}"
        ylabel = f"{benchmark.upper()} {metric_name}"
        all_values = []

        for label, kind, ident, color, ls, marker in lines:
            if kind == "method":
                scores = load_scores(model_base, ident, config_name)
            else:
                scores = load_ablation_scores(model_base, ident, config_name)
            epochs, values = [], []
            for epoch, results in scores:
                if metric_key not in results:
                    continue
                epochs.append(epoch)
                values.append(results[metric_key])
            if not epochs:
                continue
            line, = ax.plot(epochs, values, marker=marker, linestyle=ls, label=label, color=color)
            all_values.extend(values)
            drew_anything = True
            if label not in seen:
                legend_handles.append(line)
                legend_labels.append(label)
                seen.add(label)

        ax.set_title(f"{model_base} — {title}")
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(1, 11))
        ax.set_xlim(0.8, 10.2)
        ax.margins(x=0.02)
        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            padding = max(0.03, (y_max - y_min) * 0.15)
            ax.set_ylim(max(0.0, y_min - padding), min(1.0, y_max + padding))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    if not drew_anything:
        plt.close(fig)
        print(f"  No data: {benchmark} ablations")
        return

    n_cols = min(len(legend_handles), 11)
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.07),
        ncol=n_cols, fontsize=10, frameon=False,
    )
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f"{benchmark}_ablations.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_file}")


def run_ablations(model_base: str = "Qwen3-1.7B-Base"):
    print("\n=== Ablations ===")
    save_dir = PLOTS_ROOT / "ablations"
    for benchmark in ["gsm8k", "math", "svamp"]:
        plot_ablations_combined(model_base, benchmark, save_dir)


def plot_all_methods_for_model(model_base: str, save_dir: Path):
    cmap = plt.get_cmap("tab10")
    method_colors = {label: cmap(i) for i, (_, _, label) in enumerate(ALL_METHODS_SPECS)}

    n_rows = len(ALL_METHODS_BENCHMARKS)
    n_cols = len(ALL_METHODS_METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows))
    legend_handles, legend_labels, seen = [], [], set()
    drew_anything = False

    for row, benchmark in enumerate(ALL_METHODS_BENCHMARKS):
        for col, (suffix, metric_key, metric_name) in enumerate(ALL_METHODS_METRICS):
            ax = axes[row, col]
            title = f"{benchmark} {metric_name}"
            ylabel = f"{benchmark.upper()} {metric_name}"
            all_values = []

            for method, use_og, label in ALL_METHODS_SPECS:
                config_name = f"{benchmark}_og_{suffix}" if use_og else f"{benchmark}_{suffix}"
                scores = load_scores(model_base, method, config_name)
                epochs, values = [], []
                for epoch, results in scores:
                    if metric_key not in results:
                        continue
                    epochs.append(epoch)
                    values.append(results[metric_key])
                if not epochs:
                    continue
                line, = ax.plot(epochs, values, marker="o", label=label, color=method_colors[label])
                all_values.extend(values)
                drew_anything = True
                if label not in seen:
                    legend_handles.append(line)
                    legend_labels.append(label)
                    seen.add(label)

            ax.set_title(f"{model_base} — {title}")
            ax.set_xlabel("Epoch")
            ax.set_xticks(range(1, 11))
            ax.set_xlim(0.8, 10.2)
            ax.margins(x=0.02)
            if all_values:
                y_min, y_max = min(all_values), max(all_values)
                padding = max(0.03, (y_max - y_min) * 0.15)
                ax.set_ylim(max(0.0, y_min - padding), min(1.0, y_max + padding))
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)

    if not drew_anything:
        plt.close(fig)
        print(f"  No data: {model_base}")
        return

    # Order legend per ALL_METHODS_SPECS so SFT comes before SFT (original), etc.
    spec_order = {label: i for i, (_, _, label) in enumerate(ALL_METHODS_SPECS)}
    paired = sorted(zip(legend_labels, legend_handles), key=lambda x: spec_order.get(x[0], 1e9))
    legend_labels = [p[0] for p in paired]
    legend_handles = [p[1] for p in paired]

    legend_cols = min(len(legend_handles), 10)
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=legend_cols, fontsize=11, frameon=False,
    )
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f"{model_base}_all_methods.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_file}")


def run_all_methods(model_bases: list[str] = ALL_METHODS_MODELS):
    print("\n=== All Methods Per Model ===")
    save_dir = PLOTS_ROOT / "all_methods"
    for model_base in model_bases:
        plot_all_methods_for_model(model_base, save_dir)


if __name__ == "__main__":
    model_base = "Qwen3-1.7B-Base"
    run_final_results(model_base)
    run_gsm8k_comparison(model_base)
    run_qwen3_filtered(model_base)
    run_self_training(model_base)
    run_ablations(model_base)
    run_all_methods()
