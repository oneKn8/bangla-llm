#!/usr/bin/env python3
"""Supervised Fine-Tuning (SFT) script for Bangla LLM.

Uses HuggingFace TRL ``SFTTrainer`` with optional LoRA (PEFT).

Example usage:
    # Full fine-tune
    python sft.py \\
        --base-model ./pretrained-model \\
        --data ./data/sft_general.jsonl \\
        --output ./output/sft-full \\
        --epochs 3

    # LoRA fine-tune
    python sft.py \\
        --base-model ./pretrained-model \\
        --data ./data/sft_general.jsonl \\
        --output ./output/sft-lora \\
        --use-lora --lora-r 16 --lora-alpha 32 \\
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ChatML special tokens used for instruction-tuning
CHATML_TOKENS: dict[str, str] = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, returning a list of dicts."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at line %d: %s", lineno, exc)
    return records


def _format_chatml(messages: list[dict[str, str]]) -> str:
    """Convert a list of {role, content} messages to a ChatML string.

    Format:
        <|im_start|>user
        ...content...<|im_end|>
        <|im_start|>assistant
        ...content...<|im_end|>
    """
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"{CHATML_TOKENS['im_start']}{role}\n{content}{CHATML_TOKENS['im_end']}")
    return "\n".join(parts)


def _build_dataset(data_path: Path) -> Dataset:
    """Load JSONL and build a HuggingFace ``Dataset`` with a 'text' column."""
    records = _load_jsonl(data_path)
    if not records:
        logger.error("No records loaded from %s", data_path)
        sys.exit(1)

    texts: list[str] = []
    for rec in records:
        messages = rec.get("messages", [])
        if not messages:
            continue
        texts.append(_format_chatml(messages))

    logger.info("Built dataset with %d samples from %s", len(texts), data_path)
    return Dataset.from_dict({"text": texts})


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------


def _load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    """Load tokenizer from model directory, adding ChatML tokens if missing."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token (%s)", tokenizer.eos_token)

    # Add ChatML special tokens if not already present
    special_tokens_to_add: list[str] = []
    for token in CHATML_TOKENS.values():
        if token not in tokenizer.get_vocab():
            special_tokens_to_add.append(token)

    if special_tokens_to_add:
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens_to_add}
        )
        logger.info("Added %d ChatML special tokens to tokenizer", num_added)

    return tokenizer


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    use_bf16: bool = True,
) -> AutoModelForCausalLM:
    """Load causal LM, resize embeddings if tokenizer was extended."""
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    logger.info("Loading model from %s (dtype=%s)", model_path, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # Resize embeddings to account for any added special tokens
    model.resize_token_embeddings(len(tokenizer))
    return model


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------


def _get_lora_config(
    r: int,
    alpha: int,
    target_modules: list[str] | None = None,
) -> Any:
    """Build a PEFT LoraConfig."""
    from peft import LoraConfig, TaskType

    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    logger.info(
        "LoRA config: r=%d, alpha=%d, target_modules=%s",
        r,
        alpha,
        target_modules,
    )
    return config


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train(args: argparse.Namespace) -> None:
    """Run the SFT training loop."""
    from trl import SFTConfig, SFTTrainer

    # Load components
    tokenizer = _load_tokenizer(args.base_model)
    model = _load_model(args.base_model, tokenizer, use_bf16=True)
    dataset = _build_dataset(args.data)

    # LoRA
    peft_config = None
    if args.use_lora:
        peft_config = _get_lora_config(r=args.lora_r, alpha=args.lora_alpha)

    # Training config via SFTConfig (TRL 1.0)
    output_dir = str(args.output)
    training_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        bf16=True,
        fp16=False,
        max_length=args.max_seq_len,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting training: %d epochs, lr=%e, batch=%d, max_len=%d",
                args.epochs, args.lr, args.batch_size, args.max_seq_len)

    trainer.train()

    # Save
    logger.info("Saving model to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning for Bangla LLM.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to the pre-trained base model directory (or HF model ID).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the JSONL training data file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the fine-tuned model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size (default: 4).",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=False,
        help="Use LoRA (PEFT) instead of full fine-tuning.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _train(args)


if __name__ == "__main__":
    main()
