#!/usr/bin/env python3
"""Phase 3: Full SFT (all parameters, no LoRA).

Full parameter supervised fine-tuning on curated instruction pairs.
Uses ChatML format with <|im_start|> and <|im_end|> tokens.

Usage:
    python finetune/phase3_full_sft.py \
        --base-model checkpoints/kotha-cpt \
        --train-data finetune/data/kotha_top100k_train.jsonl \
        --val-data finetune/data/kotha_top100k_val.jsonl \
        --output checkpoints/kotha-1-sft
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
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
)

# Add parent dir to path for bangla_tokenizer import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bangla_tokenizer import BanglaTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHATML_TOKENS = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}

# ChatML template with {% generation %} markers so TRL's assistant_only_loss can
# mask everything except assistant turns -- loss is computed on the completion
# only, not on the (unpredictable) user/system prompt tokens.
CHATML_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] + '<|im_end|>' }}{% endgeneration %}"
    "{% else %}"
    "{{ message['content'] + '<|im_end|>' }}"
    "{% endif %}"
    "{% if not loop.last %}{{ '\n' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '\n<|im_start|>assistant\n' }}{% endif %}"
)


def add_chatml_tokens(tokenizer: PreTrainedTokenizerBase) -> int:
    """Add ChatML special tokens if not present. Returns number added."""
    tokens_to_add = []
    for token in CHATML_TOKENS.values():
        if token not in tokenizer.get_vocab():
            tokens_to_add.append(token)
    if tokens_to_add:
        num = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        logger.info("Added %d ChatML tokens", num)
        return num
    return 0


def format_chatml(messages: list[dict[str, str]]) -> str:
    """Convert messages to ChatML string."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"{CHATML_TOKENS['im_start']}{role}\n{content}{CHATML_TOKENS['im_end']}")
    return "\n".join(parts)


def load_sft_dataset(path: Path) -> Dataset:
    """Load JSONL with 'messages' field, format as ChatML text."""
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                messages = obj.get("messages", [])
                if len(messages) >= 2:
                    texts.append(format_chatml(messages))
            except json.JSONDecodeError:
                continue

    logger.info("Loaded %d samples from %s", len(texts), path)
    return Dataset.from_dict({"text": texts})


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Full SFT")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    # Import TRL here so it fails fast if not installed
    from trl import SFTConfig, SFTTrainer

    # Load tokenizer
    sp_path = Path(args.base_model) / "bangla_bpe_32k.model"
    if not sp_path.exists():
        sp_path = Path("tokenizer/output/bangla_bpe_32k.model")
    logger.info("Loading tokenizer from %s", sp_path)
    tokenizer = BanglaTokenizer(sp_model_path=str(sp_path))
    add_chatml_tokens(tokenizer)
    tokenizer.chat_template = CHATML_CHAT_TEMPLATE

    # Load model
    logger.info("Loading model from %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    )
    # Resize embeddings for new ChatML tokens
    model.resize_token_embeddings(len(tokenizer))
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{total_params:,}")

    # Load datasets
    train_dataset = load_sft_dataset(args.train_data)
    val_dataset = load_sft_dataset(args.val_data)

    # Calculate steps
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = steps_per_epoch * args.epochs
    logger.info("Train samples: %d, Effective batch: %d", len(train_dataset), effective_batch)
    logger.info("Steps/epoch: %d, Total steps: %d", steps_per_epoch, total_steps)

    # SFT config
    training_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        max_length=args.max_seq_len,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        dataloader_num_workers=0,  # NO worker duplication
        max_grad_norm=1.0,
        report_to="none",
        optim="adamw_torch",
        eval_strategy="epoch",
        # Train on completions only: the {% generation %} spans in the chat
        # template mark assistant tokens; everything else is masked to -100.
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting full SFT: %d epochs, lr=%e, %d train samples",
                args.epochs, args.lr, len(train_dataset))
    trainer.train()

    # Save final model + tokenizer
    logger.info("Saving model to %s", args.output)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    main()
