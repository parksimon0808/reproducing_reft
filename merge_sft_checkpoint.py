import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--original-model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    sft_path = Path(args.sft_checkpoint)
    orig_path = Path(args.original_model)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    orig_safetensors = sorted(orig_path.glob("*.safetensors"))
    print(f"Loading original model from {orig_path} ({len(orig_safetensors)} file(s))")
    orig_weights = {}
    for f in orig_safetensors:
        orig_weights.update(load_file(str(f)))

    sft_safetensors = sorted(sft_path.glob("*.safetensors"))
    print(f"Loading SFT checkpoint from {sft_path} ({len(sft_safetensors)} file(s))")
    sft_weights = {}
    for f in sft_safetensors:
        sft_weights.update(load_file(str(f)))

    sft_has_verl_layout = any(
        k.startswith("model.language_model.visual.") for k in sft_weights
    )
    sft_has_language_model = any(k.startswith("model.language_model.") for k in sft_weights)
    orig_has_language_model = any(k.startswith("model.language_model.") for k in orig_weights)

    if sft_has_verl_layout:
        print("Source uses verl RL merge layout; unwrapping `model.language_model.` prefix")
        TEXT_PREFIX = "model.language_model.language_model.language_model."
        VISUAL_PREFIX = "model.language_model.visual."
        remapped = {}
        for k, v in sft_weights.items():
            if k.startswith(TEXT_PREFIX):
                new_key = "model.language_model." + k[len(TEXT_PREFIX):]
            elif k.startswith(VISUAL_PREFIX):
                new_key = "model.visual." + k[len(VISUAL_PREFIX):]
            else:
                new_key = k
            remapped[new_key] = v
        sft_weights = remapped
    elif not sft_has_language_model and orig_has_language_model:
        print("Source uses CausalLM keys (model.*), remapping to multimodal (model.language_model.*)")
        remapped = {}
        for k, v in sft_weights.items():
            if k.startswith("model."):
                new_key = "model.language_model." + k[len("model."):]
                remapped[new_key] = v
            else:
                remapped[k] = v
        sft_weights = remapped

    merged = {}

    non_text_count = 0
    for k, v in orig_weights.items():
        if not k.startswith("model.language_model."):
            merged[k] = v
            non_text_count += 1

    text_count = 0
    for k, v in sft_weights.items():
        merged[k] = v
        text_count += 1

    print(f"Merged: {text_count} SFT text weights + {non_text_count} original visual/mtp weights = {len(merged)} total")

    out_safetensors = out_path / "model.safetensors"
    print(f"Saving merged weights to {out_safetensors}")
    save_file(merged, str(out_safetensors))

    shutil.copy2(str(orig_path / "config.json"), str(out_path / "config.json"))
    print("Copied config.json from original model")

    for fname in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                   "generation_config.json", "preprocessor_config.json",
                   "video_preprocessor_config.json"]:
        src = sft_path / fname
        if not src.exists():
            src = orig_path / fname
        if src.exists():
            shutil.copy2(str(src), str(out_path / fname))
            print(f"Copied {fname}")

    print(f"\nDone! Merged checkpoint at: {out_path}")


if __name__ == "__main__":
    main()
