#!/usr/bin/env python3
"""Phase 4: News-domain LoRA fine-tuning.

Trains a LoRA adapter on top of the SFT model for news-specific tasks
(summarization, headline generation, news Q&A).

Usage:
    python finetune/phase4_news_lora.py \
        --base-model checkpoints/kotha-1-sft \
        --data finetune/data/kotha_news_train.jsonl \
        --output checkpoints/kotha-1-news-lora
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM

from bangla_tokenizer import load_bangla_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHATML_TOKENS = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}


def format_chatml(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        parts.append(f"{CHATML_TOKENS['im_start']}{msg['role']}\n{msg['content']}{CHATML_TOKENS['im_end']}")
    return "\n".join(parts)


def load_sft_dataset(path: Path) -> Dataset:
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
    parser = argparse.ArgumentParser(description="Phase 4: News LoRA")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    from trl import SFTConfig, SFTTrainer

    # Load tokenizer
    tokenizer = load_bangla_tokenizer(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info("Loading model from %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    logger.info("LoRA config: r=%d, alpha=%d", args.lora_r, args.lora_alpha)

    # Load data
    dataset = load_sft_dataset(args.data)

    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(dataset) // effective_batch
    logger.info("Samples: %d, Steps/epoch: %d", len(dataset), steps_per_epoch)

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
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        report_to="none",
        optim="adamw_torch",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    logger.info("Starting news LoRA: %d epochs, lr=%e", args.epochs, args.lr)
    trainer.train()

    logger.info("Saving LoRA adapter to %s", args.output)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("Phase 4 complete.")


if __name__ == "__main__":
    main()
