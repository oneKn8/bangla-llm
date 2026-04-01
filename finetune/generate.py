#!/usr/bin/env python3
"""Simple generation / inference script for testing fine-tuned Bangla models.

Supports:
  - Full fine-tuned models
  - LoRA adapter models (auto-detected and merged)

Example usage:
    # Full model
    python generate.py --model ./output/sft-full --prompt "বাংলাদেশের রাজধানী কোথায়?"

    # LoRA model (auto-detects adapter_config.json)
    python generate.py --model ./output/sft-lora --prompt "একটি কবিতা লিখুন।"

    # Custom generation params
    python generate.py --model ./output/sft-full \\
        --prompt "৫ + ৩ = কত?" \\
        --max-tokens 128 \\
        --temperature 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _is_lora_model(model_path: Path) -> bool:
    """Check whether the model directory contains a PEFT adapter."""
    return (model_path / "adapter_config.json").exists()


def _load_lora_model(
    model_path: Path,
    dtype: torch.dtype,
    device: str,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """Load a LoRA adapter model, merge weights, and return model + tokenizer."""
    from peft import PeftModel

    # Read adapter_config to find the base model
    adapter_config_path = model_path / "adapter_config.json"
    with open(adapter_config_path, encoding="utf-8") as fh:
        adapter_config = json.load(fh)
    base_model_name = adapter_config.get("base_model_name_or_path", "")

    if not base_model_name:
        logger.error(
            "adapter_config.json does not specify base_model_name_or_path. "
            "Cannot load LoRA model without knowing the base."
        )
        sys.exit(1)

    logger.info("Loading base model: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    logger.info("Loading LoRA adapter from: %s", model_path)
    model = PeftModel.from_pretrained(base_model, str(model_path))

    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _load_full_model(
    model_path: Path,
    dtype: torch.dtype,
    device: str,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """Load a full (non-LoRA) fine-tuned model."""
    logger.info("Loading full model from: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _format_prompt(prompt: str) -> str:
    """Wrap a user prompt in ChatML format for generation."""
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


@torch.inference_mode()
def _generate(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> str:
    """Generate text from a prompt and return the assistant response."""
    formatted = _format_prompt(prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    generation_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = 0.9
        generation_kwargs["top_k"] = 50

    output_ids = model.generate(**inputs, **generation_kwargs)

    # Decode only the newly generated tokens
    generated_ids = output_ids[0, input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # Strip trailing <|im_end|> if present
    end_token = "<|im_end|>"
    if end_token in response:
        response = response[: response.index(end_token)]

    return response.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a fine-tuned Bangla model.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the fine-tuned model directory.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The user prompt to generate a response for.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate (default: 256).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. 0 = greedy (default: 0.7).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. 'cuda', 'cpu'). Auto-detected if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_path = args.model.resolve()

    if not model_path.exists():
        logger.error("Model path does not exist: %s", model_path)
        sys.exit(1)

    if _is_lora_model(model_path):
        model, tokenizer = _load_lora_model(model_path, dtype, device)
    else:
        model, tokenizer = _load_full_model(model_path, dtype, device)

    logger.info("Model loaded on %s. Generating...", device)

    response = _generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        device=device,
    )

    # Print separator and response
    print("\n" + "=" * 60)
    print(f"Prompt: {args.prompt}")
    print("-" * 60)
    print(f"Response:\n{response}")
    print("=" * 60)


if __name__ == "__main__":
    main()
