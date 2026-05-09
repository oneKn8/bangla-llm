#!/usr/bin/env python3
"""Phase 2: Continued pre-training on textbook data.

Improves base model Bengali fluency before SFT.
Uses causal LM objective (same as pre-training) with low learning rate.

Usage:
    python finetune/phase2_continued_pretrain.py \
        --base-model training/checkpoints/step-4000 \
        --data finetune/data/cpt_textbook.jsonl \
        --output checkpoints/kotha-cpt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Add parent dir to path for bangla_tokenizer import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bangla_tokenizer import BanglaTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_text_dataset(data_path: Path, tokenizer, max_seq_len: int) -> Dataset:
    """Load JSONL with 'text' field and tokenize for causal LM."""
    texts = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if len(text) > 50:
                    texts.append(text)
            except json.JSONDecodeError:
                continue

    logger.info("Loaded %d text passages", len(texts))

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=4)
    # Filter out empty sequences
    ds = ds.filter(lambda x: len(x["input_ids"]) > 10)
    logger.info("Tokenized dataset: %d samples (after filtering empty)", len(ds))
    return ds


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Continued pre-training")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    # Load tokenizer (SP model must be in base-model dir or --tokenizer)
    sp_path = Path(args.base_model) / "bangla_bpe_32k.model"
    if not sp_path.exists():
        # Fallback: check common location
        sp_path = Path("tokenizer/output/bangla_bpe_32k.model")
    logger.info("Loading tokenizer from %s", sp_path)
    tokenizer = BanglaTokenizer(sp_model_path=str(sp_path))

    # Load model
    logger.info("Loading model from %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{total_params:,}")

    # Load and tokenize data
    dataset = load_text_dataset(args.data, tokenizer, args.max_seq_len)

    # Data collator for causal LM (masks padding, shifts labels internally)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM, not masked LM
    )

    # Calculate steps for logging
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(dataset) // effective_batch
    total_steps = steps_per_epoch * args.epochs
    logger.info("Effective batch: %d, Steps/epoch: %d, Total steps: %d",
                effective_batch, steps_per_epoch, total_steps)

    training_args = TrainingArguments(
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
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        dataloader_num_workers=0,  # NO worker duplication
        max_grad_norm=1.0,
        report_to="none",
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    logger.info("Starting continued pre-training: %d epochs, lr=%e", args.epochs, args.lr)
    trainer.train()

    # Save final
    logger.info("Saving model to %s", args.output)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("Phase 2 complete.")


if __name__ == "__main__":
    main()
