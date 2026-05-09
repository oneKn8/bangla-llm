#!/usr/bin/env python3
"""Compare Bangla model checkpoints on the same prompts.

This script loads one model at a time, runs the same prompt set, and prints a
compact side-by-side report. It is meant for quick qualitative debugging of:

  - raw base checkpoints
  - repaired base checkpoints
  - SFT / chat checkpoints
  - LoRA adapters (auto-detected)

Examples:
    python3 scripts/compare_models.py

    python3 scripts/compare_models.py \
        --prompt "বাংলাদেশের রাজধানী" \
        --prompt "বর্ষার রাতে ছাদের উপর দাঁড়িয়ে আমি" \
        --output comparison.txt

    python3 scripts/compare_models.py \
        --model "old-base::training/checkpoints/step-4000::raw" \
        --model "repaired-best::brev_artifacts/base_repair_20260402_033659/best::raw"
"""

from __future__ import annotations

import argparse
import gc
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "finetune"))

from bangla_tokenizer import load_bangla_tokenizer  # noqa: E402
from generate import _is_lora_model, _load_lora_model  # noqa: E402

CHATML_END = "<|im_end|>"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path
    mode: str


DEFAULT_MODELS = [
    ModelSpec("old-base", ROOT / "training/checkpoints/step-4000", "raw"),
    ModelSpec(
        "repaired-best",
        ROOT / "brev_artifacts/base_repair_20260402_033659/best",
        "raw",
    ),
    ModelSpec("old-sft", ROOT / "checkpoints/kotha-1-sft", "chat"),
]

DEFAULT_PROMPTS = [
    "বাংলাদেশের রাজধানী",
    "আজ সকালে ঢাকা শহরের আকাশ ছিল",
    "বর্ষার রাতে ছাদের উপর দাঁড়িয়ে আমি",
    "আমি আজ খুব ভেঙে পড়েছি",
    "৫ + ৩ =",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple Bangla checkpoints on the same prompts.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model spec in the form 'label::path::mode' where mode is "
            "'raw' or 'chat'. Can be repeated. If omitted, sensible local "
            "defaults are used."
        ),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to test. Can be repeated. If omitted, a default set is used.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=96,
        help="Maximum number of new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0 uses greedy decoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. 'cuda' or 'cpu'. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to save the text report to.",
    )
    return parser.parse_args()


def _parse_model_spec(spec: str) -> ModelSpec:
    parts = spec.split("::")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --model value: {spec!r}. Expected 'label::path::mode'.",
        )

    label, raw_path, mode = parts
    mode = mode.strip().lower()
    if mode not in {"raw", "chat"}:
        raise ValueError(
            f"Invalid mode {mode!r} for model {label!r}. Use 'raw' or 'chat'.",
        )

    return ModelSpec(label=label.strip(), path=(ROOT / raw_path).resolve(), mode=mode)


def _resolve_models(args: argparse.Namespace) -> list[ModelSpec]:
    if args.model:
        models = [_parse_model_spec(spec) for spec in args.model]
    else:
        models = [spec for spec in DEFAULT_MODELS if spec.path.exists()]

    if not models:
        raise SystemExit("No valid models found. Pass --model explicitly.")

    missing = [spec for spec in models if not spec.path.exists()]
    if missing:
        details = ", ".join(f"{spec.label}={spec.path}" for spec in missing)
        raise SystemExit(f"These model paths do not exist: {details}")

    return models


def _resolve_prompts(args: argparse.Namespace) -> list[str]:
    prompts = [prompt.strip() for prompt in args.prompt if prompt.strip()]
    return prompts or DEFAULT_PROMPTS


def _device_and_dtype(device_arg: str | None) -> tuple[str, torch.dtype]:
    if device_arg:
        device = device_arg
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return device, dtype


def _format_prompt(prompt: str, mode: str) -> str:
    if mode == "raw":
        return prompt
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


@torch.inference_mode()
def _generate_text(
    model,
    tokenizer,
    prompt: str,
    mode: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    formatted = _format_prompt(prompt, mode)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    generation_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": 1.1,
    }
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = 0.9
        generation_kwargs["top_k"] = 50
    else:
        generation_kwargs["do_sample"] = False

    output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[0, input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=False)

    if mode == "chat" and CHATML_END in response:
        response = response[: response.index(CHATML_END)]
    if "</s>" in response:
        response = response[: response.index("</s>")]

    return response.strip()


def _resolve_tokenizer_source(model_path: Path) -> str:
    repo_tokenizer = ROOT / "tokenizer/output/bangla_bpe_32k.model"
    candidates = [
        model_path,
        repo_tokenizer,
    ]
    for candidate in candidates:
        try:
            load_bangla_tokenizer(str(candidate))
            return str(candidate)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"Could not find bangla_bpe_32k.model for {model_path}. Tried {candidates}"
    )


def _load_full_checkpoint(model_path: Path, dtype: torch.dtype, device: str):
    tokenizer_source = _resolve_tokenizer_source(model_path)
    tokenizer = load_bangla_tokenizer(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
    model.eval()
    return model, tokenizer


def _load_model(spec: ModelSpec, dtype: torch.dtype, device: str):
    if _is_lora_model(spec.path):
        return _load_lora_model(spec.path, dtype, device)
    return _load_full_checkpoint(spec.path, dtype, device)


def _unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _render_report(results: list[dict], prompts: list[str]) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("Bangla Model Comparison")
    lines.append("=" * 72)
    lines.append("")

    for prompt_index, prompt in enumerate(prompts, start=1):
        lines.append(f"[Prompt {prompt_index}] {prompt}")
        lines.append("-" * 72)
        for item in results:
            output = item["outputs"][prompt]
            wrapped = textwrap.fill(output or "<empty>", width=72)
            lines.append(
                f"{item['label']} [{item['mode']}, {item['seconds']:.1f}s]:",
            )
            lines.append(wrapped)
            lines.append("")
        lines.append("")

    lines.append("Model Paths")
    lines.append("-" * 72)
    for item in results:
        lines.append(f"{item['label']}: {item['path']}")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    models = _resolve_models(args)
    prompts = _resolve_prompts(args)
    device, dtype = _device_and_dtype(args.device)

    print(f"Using device: {device} ({dtype})")
    print(f"Prompts: {len(prompts)}")
    print(f"Models: {', '.join(spec.label for spec in models)}")
    print()

    results: list[dict] = []

    for spec in models:
        print(f"Loading {spec.label} from {spec.path} [{spec.mode}]...")
        started = time.perf_counter()
        model, tokenizer = _load_model(spec, dtype, device)
        outputs = {}
        for prompt in prompts:
            print(f"  Generating for: {prompt}")
            outputs[prompt] = _generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                mode=spec.mode,
                device=device,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        elapsed = time.perf_counter() - started
        results.append(
            {
                "label": spec.label,
                "path": str(spec.path),
                "mode": spec.mode,
                "seconds": elapsed,
                "outputs": outputs,
            },
        )
        _unload_model(model)
        del tokenizer
        gc.collect()
        print()

    report = _render_report(results, prompts)
    print(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
