# Bangla-LLM 300M Build Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a 300M-parameter Bangla language model from scratch, fine-tune a news variant for Dr. Khan, release on HuggingFace, and draft an arXiv paper.

**Architecture:** LLaMA-style decoder-only transformer (18 layers, d=1024, GQA). Pre-train on ~5GB cleaned Bangla text (Culturax + Wikipedia + newspapers). SFT + DPO for instruction following. News-domain LoRA fine-tune as separate deliverable.

**Tech Stack:** PyTorch, HuggingFace Transformers + Accelerate, SentencePiece, Brev (A100), Colab Pro

**Timeline:** ~2 weeks (data this week, training next week)

**Budget:** ~$60 Brev credits + Colab Pro

---

## Phase 1: Expand Data Corpus (Days 1-3)

We have 3.7GB from Culturax (545K docs). Need to add Wikipedia and newspapers to reach 5-6GB target. Newspapers are critical for the news fine-tune.

### Task 1.1: Run Wikipedia Collector

**Files:**
- Run: `data-pipeline/collect.py`
- Collector: `data-pipeline/collectors/wikipedia.py`
- Output: `data-pipeline/data/raw/wikipedia/`

**Step 1: Verify collector works**

```bash
cd /home/oneknight/projects/bangla-llm/data-pipeline
python3 -c "from collectors.wikipedia import WikipediaCollector; print('OK')"
```

Expected: OK (no import errors)

**Step 2: Run Wikipedia collection**

```bash
cd /home/oneknight/projects/bangla-llm/data-pipeline
python3 collect.py wikipedia
```

Expected: Downloads bnwiki dump (~1.2GB), extracts articles to raw/wikipedia/. Takes 30-60 min depending on network. Should yield ~800MB clean text.

**Step 3: Verify output**

```bash
wc -l data/raw/wikipedia/*.jsonl
head -1 data/raw/wikipedia/*.jsonl | python3 -c "import sys,json; print(json.loads(sys.stdin.readline()).keys())"
```

Expected: JSONL files with text/title/source fields.

**Step 4: Commit**

```bash
git add -A && git commit -m "data: collect Bengali Wikipedia dump"
```

---

### Task 1.2: Run Newspaper Collectors

**Files:**
- Collector: `data-pipeline/collectors/newspaper.py`
- Config: `data-pipeline/config.py` (NEWSPAPERS dict)
- Output: `data-pipeline/data/raw/newspapers/`

**Step 1: Run Prothom Alo collector**

```bash
cd /home/oneknight/projects/bangla-llm/data-pipeline
python3 collect.py newspaper --source prothomalo
```

Expected: Crawls sitemap, downloads articles at 1 req/sec. Rate-limited, takes 2-5 hours. Run in background.

**Step 2: Run Kaler Kantho collector**

```bash
python3 collect.py newspaper --source kalerkantho
```

**Step 3: Run Ittefaq collector**

```bash
python3 collect.py newspaper --source ittefaq
```

Note: All three can run in parallel in separate terminals. Combined output target: ~1-1.5GB raw.

**Step 4: Verify output**

```bash
du -sh data/raw/newspapers/
wc -l data/raw/newspapers/*.jsonl
```

**Step 5: Commit**

```bash
git add -A && git commit -m "data: collect Bangla newspaper articles (PA, KK, Ittefaq)"
```

---

### Task 1.3: Run HF Corpus Collectors (Sangraha + CC-100)

**Files:**
- Collector: `data-pipeline/collectors/hf_corpus.py`
- Config: `data-pipeline/config.py` (HF_DATASETS dict)
- Output: `data-pipeline/data/raw/hf_corpus/`

**Step 1: Run Sangraha collection**

```bash
cd /home/oneknight/projects/bangla-llm/data-pipeline
python3 collect.py hf_corpus --dataset sangraha --max-docs 500000
```

Expected: Streams Bengali subset from AI4Bharat/Sangraha. Cap at 500K docs to stay within storage budget.

**Step 2: Run CC-100 collection**

```bash
python3 collect.py hf_corpus --dataset cc100 --max-docs 300000
```

**Step 3: Verify**

```bash
du -sh data/raw/hf_corpus/
```

**Step 4: Commit**

```bash
git add -A && git commit -m "data: collect Sangraha and CC-100 Bengali corpora"
```

---

### Task 1.4: Process and Merge All Sources

**Files:**
- Pipeline: `data-pipeline/pipeline_parallel.py`
- Output: `data-pipeline/data/processed/` (merged, deduped)

**Step 1: Run processing pipeline on new raw data**

```bash
cd /home/oneknight/projects/bangla-llm/data-pipeline
python3 pipeline_parallel.py --input data/raw/ --output data/processed/all_sources.jsonl --workers 4
```

This runs: normalize -> lang_detect -> quality_filter -> dedup on all new raw data.

**Step 2: Cross-deduplicate against existing Culturax**

```bash
python3 -c "
from processing.dedup import CrossSourceDeduplicator
dedup = CrossSourceDeduplicator()
dedup.run(
    existing='data/processed/culturax.jsonl',
    new='data/processed/all_sources.jsonl',
    output='data/processed/merged_corpus.jsonl'
)
"
```

If CrossSourceDeduplicator doesn't exist, implement it: load MinHash index from culturax, filter all_sources against it, then concatenate.

**Step 3: Verify final corpus size**

```bash
du -sh data/processed/merged_corpus.jsonl
wc -l data/processed/merged_corpus.jsonl
```

Target: 5-6GB, 1-3M documents.

**Step 4: Tag newspaper articles for news fine-tune**

```bash
python3 -c "
import json
news_count = 0
with open('data/processed/merged_corpus.jsonl') as f:
    for line in f:
        doc = json.loads(line)
        if doc.get('source', '') in ('prothomalo', 'kalerkantho', 'ittefaq'):
            news_count += 1
print(f'News articles: {news_count}')
"
```

Save news article IDs/indices for Phase 4 news fine-tune.

**Step 5: Generate corpus statistics**

```bash
python3 -c "
from processing.stats import generate_report
generate_report('data/processed/merged_corpus.jsonl', 'data/reports/final_corpus_stats.json')
"
```

**Step 6: Commit**

```bash
git add -A && git commit -m "data: merge and deduplicate all sources into final corpus"
```

---

## Phase 2: Train Tokenizer (Day 3-4)

### Task 2.1: Create Tokenizer Training Script

**Files:**
- Create: `tokenizer/train_tokenizer.py`
- Create: `tokenizer/audit_vocab.py`
- Test: `tokenizer/test_tokenizer.py`

**Step 1: Write tokenizer training script**

Create `tokenizer/train_tokenizer.py`:

```python
"""Train a 32K BPE tokenizer on cleaned Bangla corpus using SentencePiece."""

import argparse
import random
import sentencepiece as spm
from pathlib import Path


def sample_corpus(input_path: str, output_path: str, target_gb: float = 1.5):
    """Sample ~1.5GB from corpus for tokenizer training."""
    target_bytes = int(target_gb * 1024**3)
    current_bytes = 0
    lines = []

    with open(input_path) as f:
        all_lines = f.readlines()

    random.seed(42)
    random.shuffle(all_lines)

    with open(output_path, "w") as out:
        for line in all_lines:
            import json
            doc = json.loads(line)
            text = doc.get("text", "")
            out.write(text + "\n")
            current_bytes += len(text.encode("utf-8"))
            if current_bytes >= target_bytes:
                break

    print(f"Sampled {current_bytes / 1024**3:.2f} GB for tokenizer training")


def train(input_path: str, model_prefix: str, vocab_size: int = 32000):
    """Train SentencePiece BPE tokenizer."""
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9999,
        byte_fallback=True,
        normalization_rule_name="identity",
        split_by_whitespace=True,
        split_digits=True,
        num_threads=8,
        train_extremely_large_corpus=True,
        pad_id=3,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        user_defined_symbols=[
            "<|user|>", "<|assistant|>", "<|end|>", "<|system|>",
            "<|pad|>", "<|sep|>", "<|cls|>", "<|mask|>",
        ],
    )
    print(f"Tokenizer saved: {model_prefix}.model, {model_prefix}.vocab")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Path to merged corpus JSONL")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--sample-gb", type=float, default=1.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = output_dir / "tokenizer_sample.txt"
    model_prefix = str(output_dir / "bangla_32k")

    print("Step 1: Sampling corpus...")
    sample_corpus(args.corpus, str(sample_path), args.sample_gb)

    print("Step 2: Training tokenizer...")
    train(str(sample_path), model_prefix, args.vocab_size)

    # Clean up sample file
    sample_path.unlink()
    print("Done.")


if __name__ == "__main__":
    main()
```

**Step 2: Write vocab audit script**

Create `tokenizer/audit_vocab.py`:

```python
"""Audit tokenizer vocabulary for script contamination."""

import argparse
import sentencepiece as spm

DEVANAGARI_RANGE = range(0x0900, 0x0980)
ASSAMESE_ONLY = {0x09F0, 0x09F1}
BLOCKED_RANGES = [
    range(0x0900, 0x0980),  # Devanagari
    range(0x0B00, 0x0B80),  # Odia
    range(0x0B80, 0x0C00),  # Tamil
    range(0x0C00, 0x0C80),  # Telugu
    range(0x0C80, 0x0D00),  # Kannada
    range(0x0D00, 0x0D80),  # Malayalam
    range(0x0A00, 0x0A80),  # Gurmukhi
    range(0x0A80, 0x0B00),  # Gujarati
]


def audit(model_path: str) -> dict:
    sp = spm.SentencePieceProcessor(model_file=model_path)
    contaminated = []

    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        for char in piece:
            cp = ord(char)
            if cp in ASSAMESE_ONLY:
                contaminated.append((i, piece, "Assamese-only"))
                break
            for r in BLOCKED_RANGES:
                if cp in r:
                    contaminated.append((i, piece, f"Blocked script U+{cp:04X}"))
                    break

    result = {
        "vocab_size": sp.get_piece_size(),
        "contaminated_count": len(contaminated),
        "contaminated_tokens": contaminated,
        "pass": len(contaminated) == 0,
    }

    if contaminated:
        print(f"FAIL: {len(contaminated)} contaminated tokens found:")
        for idx, piece, reason in contaminated[:20]:
            print(f"  [{idx}] '{piece}' -- {reason}")
    else:
        print(f"PASS: All {sp.get_piece_size()} tokens clean.")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .model file")
    args = parser.parse_args()
    audit(args.model)


if __name__ == "__main__":
    main()
```

**Step 3: Write tokenizer validation test**

Create `tokenizer/test_tokenizer.py`:

```python
"""Validate tokenizer quality after training."""

import argparse
import sentencepiece as spm


COMMON_WORDS = ["বাংলাদেশ", "সরকার", "তিনি", "করা", "হয়েছে", "প্রথম", "বিশ্ব"]


def validate(model_path: str, corpus_path: str, num_samples: int = 1000):
    sp = spm.SentencePieceProcessor(model_file=model_path)

    # Test 1: Common words should be single tokens
    print("Test 1: Common word tokenization")
    for word in COMMON_WORDS:
        tokens = sp.encode(word, out_type=str)
        status = "PASS (single)" if len(tokens) == 1 else f"INFO ({len(tokens)} tokens)"
        print(f"  '{word}' -> {tokens}  [{status}]")

    # Test 2: Tokens-per-word ratio on corpus sample
    print(f"\nTest 2: Efficiency on {num_samples} corpus samples")
    import json
    total_words = 0
    total_tokens = 0
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            doc = json.loads(line)
            text = doc.get("text", "")
            words = text.split()
            tokens = sp.encode(text)
            total_words += len(words)
            total_tokens += len(tokens)

    ratio = total_tokens / total_words if total_words > 0 else 0
    print(f"  Words: {total_words:,}, Tokens: {total_tokens:,}")
    print(f"  Tokens/word ratio: {ratio:.2f} (target: 1.3-1.8)")
    if 1.0 <= ratio <= 2.5:
        print("  PASS")
    else:
        print("  WARN: ratio outside expected range")

    # Test 3: Roundtrip
    print("\nTest 3: Encode-decode roundtrip")
    test_text = "বাংলাদেশের রাজধানী ঢাকা।"
    encoded = sp.encode(test_text)
    decoded = sp.decode(encoded)
    if decoded == test_text:
        print(f"  PASS: '{test_text}' roundtrips correctly")
    else:
        print(f"  FAIL: '{test_text}' -> '{decoded}'")

    # Test 4: Count total tokens in full corpus
    print(f"\nTest 4: Full corpus token count")
    total = 0
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            total += len(sp.encode(doc.get("text", "")))
    print(f"  Total tokens: {total:,}")
    print(f"  This determines training epochs.")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    validate(args.model, args.corpus, args.samples)


if __name__ == "__main__":
    main()
```

**Step 4: Commit**

```bash
git add tokenizer/ && git commit -m "feat: add tokenizer training, audit, and validation scripts"
```

---

### Task 2.2: Train and Validate Tokenizer

**Step 1: Install SentencePiece**

```bash
pip install sentencepiece
```

**Step 2: Train tokenizer on merged corpus**

```bash
cd /home/oneknight/projects/bangla-llm
python3 tokenizer/train_tokenizer.py \
    --corpus data-pipeline/data/processed/merged_corpus.jsonl \
    --output-dir tokenizer/output \
    --vocab-size 32000 \
    --sample-gb 1.5
```

Expected: Creates `tokenizer/output/bangla_32k.model` and `bangla_32k.vocab`. Takes 10-30 min on CPU.

**Step 3: Audit vocabulary for contamination**

```bash
python3 tokenizer/audit_vocab.py --model tokenizer/output/bangla_32k.model
```

Expected: PASS with 0 contaminated tokens. If FAIL, retrain with stricter filtering.

**Step 4: Validate tokenizer quality**

```bash
python3 tokenizer/test_tokenizer.py \
    --model tokenizer/output/bangla_32k.model \
    --corpus data-pipeline/data/processed/merged_corpus.jsonl
```

Expected: tokens/word ratio 1.3-1.8, common words as single tokens, clean roundtrip.

**CRITICAL:** Note the total token count from Test 4. This determines training epochs:
- If <1B tokens: plan 3-4 epochs to reach ~2-3B effective tokens
- If >1B tokens: 2 epochs sufficient

**Step 5: Commit**

```bash
git add tokenizer/output/ && git commit -m "feat: trained 32K Bangla BPE tokenizer"
```

---

## Phase 3: Pre-train 300M Model (Days 5-8)

### Task 3.1: Create Training Infrastructure

**Files:**
- Create: `training/config.py`
- Create: `training/model.py`
- Create: `training/dataset.py`
- Create: `training/train.py`

**Step 1: Write model config**

Create `training/config.py`:

```python
"""LLaMA-style model configuration for Bangla-LLM 300M."""

from dataclasses import dataclass


@dataclass
class BanglaLLMConfig:
    d_model: int = 1024
    n_layers: int = 18
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ffn: int = 4096
    vocab_size: int = 32000
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    weight_tying: bool = True
    dropout: float = 0.0

    def to_hf_config(self):
        """Convert to HuggingFace LlamaConfig."""
        from transformers import LlamaConfig
        return LlamaConfig(
            hidden_size=self.d_model,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            num_key_value_heads=self.n_kv_heads,
            intermediate_size=self.d_ffn,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_seq_len,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
            tie_word_embeddings=self.weight_tying,
            hidden_act="silu",
            attention_dropout=self.dropout,
        )
```

**Step 2: Write dataset loader**

Create `training/dataset.py`:

```python
"""Streaming dataset for pre-training on tokenized Bangla corpus."""

import json
import torch
from torch.utils.data import IterableDataset
from pathlib import Path


class BanglaPretrainDataset(IterableDataset):
    """Packs documents into fixed-length sequences for causal LM training."""

    def __init__(self, corpus_path: str, tokenizer_path: str, seq_len: int = 2048):
        self.corpus_path = corpus_path
        self.seq_len = seq_len

        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    def __iter__(self):
        buffer = []
        with open(self.corpus_path) as f:
            for line in f:
                doc = json.loads(line)
                text = doc.get("text", "")
                tokens = [self.bos_id] + self.sp.encode(text) + [self.eos_id]
                buffer.extend(tokens)

                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[:self.seq_len + 1]
                    buffer = buffer[self.seq_len:]
                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)
                    yield {"input_ids": input_ids, "labels": labels}


def tokenize_corpus(corpus_path: str, tokenizer_path: str, output_path: str):
    """Pre-tokenize entire corpus to binary for faster training."""
    import sentencepiece as spm
    import numpy as np

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    all_tokens = []

    with open(corpus_path) as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            text = doc.get("text", "")
            tokens = [sp.bos_id()] + sp.encode(text) + [sp.eos_id()]
            all_tokens.extend(tokens)
            if (i + 1) % 100000 == 0:
                print(f"  Tokenized {i+1} documents, {len(all_tokens):,} tokens")

    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(output_path)
    print(f"Saved {len(all_tokens):,} tokens to {output_path}")
    return len(all_tokens)
```

**Step 3: Write training script**

Create `training/train.py`:

```python
"""Pre-training script for Bangla-LLM 300M using HuggingFace Transformers + Accelerate."""

import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import LlamaForCausalLM, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from pathlib import Path

from config import BanglaLLMConfig


class TokenDataset(IterableDataset):
    """Load pre-tokenized binary file and yield fixed-length sequences."""

    def __init__(self, token_file: str, seq_len: int = 2048, epoch: int = 0):
        self.data = np.memmap(token_file, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.epoch = epoch

    def __iter__(self):
        # Shuffle start offset per epoch for variety
        rng = np.random.default_rng(seed=self.epoch)
        offset = rng.integers(0, self.seq_len)
        idx = offset

        while idx + self.seq_len + 1 <= len(self.data):
            chunk = self.data[idx:idx + self.seq_len + 1].astype(np.int64)
            x = torch.from_numpy(chunk[:-1])
            y = torch.from_numpy(chunk[1:])
            yield {"input_ids": x, "labels": y}
            idx += self.seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True, help="Pre-tokenized binary file")
    parser.add_argument("--output-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum,
    )

    # Model
    cfg = BanglaLLMConfig()
    hf_config = cfg.to_hf_config()
    if args.resume:
        model = LlamaForCausalLM.from_pretrained(args.resume)
        accelerator.print(f"Resumed from {args.resume}")
    else:
        model = LlamaForCausalLM(hf_config)
        accelerator.print(f"Initialized {sum(p.numel() for p in model.parameters()):,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Calculate steps
    data = np.memmap(args.tokens, dtype=np.uint16, mode="r")
    tokens_per_epoch = len(data)
    seqs_per_epoch = tokens_per_epoch // args.seq_len
    steps_per_epoch = seqs_per_epoch // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs

    accelerator.print(f"Tokens: {tokens_per_epoch:,}")
    accelerator.print(f"Steps/epoch: {steps_per_epoch:,}, Total steps: {total_steps:,}")
    accelerator.print(f"Effective batch: {args.batch_size * args.grad_accum * args.seq_len:,} tokens")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # Prepare with accelerate
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        dataset = TokenDataset(args.tokens, args.seq_len, epoch=epoch)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        dataloader = accelerator.prepare(dataloader)

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dataloader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % 50 == 0:
                avg_loss = epoch_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                accelerator.print(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                )

            if global_step % args.save_every == 0:
                ckpt_path = output_dir / f"checkpoint-{global_step}"
                accelerator.wait_for_everyone()
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(str(ckpt_path))
                accelerator.print(f"Saved checkpoint: {ckpt_path}")

        # End of epoch
        avg_loss = epoch_loss / max(epoch_steps, 1)
        accelerator.print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

        # Save epoch checkpoint
        ckpt_path = output_dir / f"epoch-{epoch+1}"
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(str(ckpt_path))

    # Save final model
    final_path = output_dir / "final"
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(str(final_path))
    accelerator.print(f"Training complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
```

**Step 4: Commit**

```bash
git add training/ && git commit -m "feat: add pre-training infrastructure (config, dataset, train script)"
```

---

### Task 3.2: Pre-tokenize Corpus

**Step 1: Install dependencies**

```bash
pip install sentencepiece numpy
```

**Step 2: Pre-tokenize corpus to binary**

```bash
cd /home/oneknight/projects/bangla-llm
python3 -c "
from training.dataset import tokenize_corpus
total = tokenize_corpus(
    'data-pipeline/data/processed/merged_corpus.jsonl',
    'tokenizer/output/bangla_32k.model',
    'training/data/train_tokens.bin'
)
print(f'Total tokens: {total:,}')
"
```

Expected: Writes binary file of uint16 token IDs. Note the total count.

**Step 3: Commit**

```bash
git add -A && git commit -m "data: pre-tokenize corpus to binary format"
```

---

### Task 3.3: Train on Brev A100

**Step 1: Upload to Brev instance**

```bash
# Create Brev instance with A100 80GB
# rsync project files
rsync -avz --exclude '.venv' --exclude 'data-pipeline/data/raw' \
    /home/oneknight/projects/bangla-llm/ brev:/home/user/bangla-llm/
```

**Step 2: Install dependencies on Brev**

```bash
pip install torch transformers accelerate sentencepiece numpy
```

**Step 3: Launch training**

```bash
cd /home/user/bangla-llm/training
accelerate launch train.py \
    --tokens data/train_tokens.bin \
    --output-dir checkpoints \
    --epochs 3 \
    --batch-size 8 \
    --grad-accum 32 \
    --lr 3e-4 \
    --warmup-steps 2000 \
    --save-every 1000 \
    --seq-len 2048
```

Expected: ~6-12 hours on A100 80GB. Monitor loss -- should decrease steadily.

**Step 4: Download final checkpoint**

```bash
rsync -avz brev:/home/user/bangla-llm/training/checkpoints/final/ \
    /home/oneknight/projects/bangla-llm/training/checkpoints/final/
```

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: pre-trained Bangla-LLM 300M base model"
```

---

## Phase 4: Fine-tuning (Days 9-10)

### Task 4.1: Create Instruction Dataset

**Files:**
- Create: `finetune/create_sft_data.py`
- Output: `finetune/data/sft_bangla.jsonl`

**Step 1: Write SFT data creation script**

Create `finetune/create_sft_data.py` that:
1. Translates 5K high-quality instruction pairs from OpenOrca/Alpaca to Bangla using Claude API
2. Creates 500 Bangla-native prompts (Bangladesh facts, Bangla grammar, news summarization, etc.)
3. Outputs ChatML-formatted JSONL

Key format:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Step 2: Create news-specific SFT data**

From the newspaper articles collected in Phase 1, create:
- 1000 news summarization pairs (article -> summary)
- 500 news Q&A pairs (article + question -> answer)
- 500 headline generation pairs (article -> headline)

Save to `finetune/data/sft_news.jsonl`

**Step 3: Commit**

```bash
git add finetune/ && git commit -m "feat: create SFT instruction datasets (general + news)"
```

---

### Task 4.2: Run SFT

**Files:**
- Create: `finetune/sft.py`

Use TRL's SFTTrainer or HuggingFace Trainer with LoRA for efficiency.

```bash
cd /home/oneknight/projects/bangla-llm
python3 finetune/sft.py \
    --base-model training/checkpoints/final \
    --data finetune/data/sft_bangla.jsonl \
    --output finetune/checkpoints/sft \
    --epochs 3 \
    --lr 2e-5
```

Expected: ~1-2 hours on A100 or Colab Pro.

**Step 2: Commit**

```bash
git add -A && git commit -m "feat: SFT fine-tuned Bangla-LLM"
```

---

### Task 4.3: News Domain LoRA Fine-tune (Dr. Khan Deliverable)

**Step 1: Fine-tune on news data**

```bash
python3 finetune/sft.py \
    --base-model finetune/checkpoints/sft \
    --data finetune/data/sft_news.jsonl \
    --output finetune/checkpoints/news-lora \
    --use-lora \
    --lora-r 16 \
    --epochs 3 \
    --lr 1e-4
```

Expected: ~30 min. Creates a small LoRA adapter specifically for Bangla news tasks.

**Step 2: Test news model**

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained('finetune/checkpoints/news-lora')
# Test with a news summarization prompt
prompt = '<|user|>এই সংবাদটি সংক্ষেপে বলুন:<|end|><|assistant|>'
# Generate and print
"
```

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: news-domain LoRA fine-tune for Bangla news LLM"
```

---

## Phase 5: Evaluation (Days 11-12)

### Task 5.1: Run Benchmarks

**Files:**
- Create: `evaluation/run_benchmarks.py`
- Create: `evaluation/human_eval_prompts.py`

**Step 1: Write benchmark runner**

Test on:
- Perplexity on held-out test set
- IndicNLPSuite (sentiment, NER, NLI) -- zero-shot and few-shot
- Tokenizer efficiency vs LLaMA/GPT tokenizers
- News-specific: summarization ROUGE scores on held-out newspaper articles

**Step 2: Run benchmarks**

```bash
python3 evaluation/run_benchmarks.py \
    --model finetune/checkpoints/sft \
    --news-model finetune/checkpoints/news-lora \
    --tokenizer tokenizer/output/bangla_32k.model \
    --output evaluation/results/
```

**Step 3: Commit**

```bash
git add evaluation/ && git commit -m "feat: evaluation benchmarks and results"
```

---

## Phase 6: Release and Paper (Days 13-14)

### Task 6.1: HuggingFace Hub Release

**Step 1: Create model card and upload**

- Upload base model, SFT model, news LoRA, tokenizer
- Write model card with training details, benchmarks, limitations
- License: Apache 2.0

**Step 2: Quantize for edge deployment**

```bash
# GGUF conversion for llama.cpp
python3 -c "
# Convert to GGUF Q4_K_M and Q8_0
"
```

**Step 3: Commit and push**

```bash
git add -A && git commit -m "feat: HuggingFace release with model cards and quantized versions"
```

---

### Task 6.2: Draft arXiv Paper

**Files:**
- Create: `paper/bangla_llm.tex` (or `paper/bangla_llm.md` for initial draft)

**Paper structure:**
1. Abstract -- First from-scratch Bangla LLM with contamination-controlled pipeline
2. Introduction -- Motivation (under-resourced language, no dedicated Bangla LLM)
3. Data Pipeline -- 11 sources, contamination controls, deduplication
4. Model Architecture -- 300M LLaMA-style, design decisions
5. Training -- Pre-training setup, loss curves
6. Fine-tuning -- SFT + DPO + news-domain LoRA
7. Evaluation -- Benchmarks, human eval, tokenizer efficiency
8. Related Work -- Existing Bangla NLP, multilingual models
9. Conclusion + Future Work (1B scale-up)

**Target venue:** arXiv cs.CL (Computation and Language)

**Co-author:** Dr. Latifur Khan (pending his agreement)

---

## Summary

| Phase | Days | Cost | Output |
|-------|------|------|--------|
| 1. Data | 1-3 | $0 | 5-6GB merged corpus |
| 2. Tokenizer | 3-4 | $0 | bangla_32k.model |
| 3. Pre-train | 5-8 | $15-30 | 300M base model |
| 4. Fine-tune | 9-10 | $5-10 | SFT + news LoRA |
| 5. Evaluate | 11-12 | $0 | Benchmark results |
| 6. Release + Paper | 13-14 | $0 | HF release + arXiv draft |
| **Total** | **~14 days** | **~$20-40** | **Published Bangla LLM** |

## Critical Path

Data collection (especially newspapers) is the bottleneck -- start all collectors Day 1 and let them run overnight. Everything else builds sequentially.

## Risk Mitigation

- **If newspaper crawling fails:** Fall back to Culturax + Wikipedia + Sangraha. Still enough data.
- **If training diverges:** Reduce LR, increase warmup, check data quality.
- **If Brev credits run out:** Switch to Colab Pro A100 (40GB, may need gradient checkpointing).
- **If tokenizer is contaminated:** Re-run pipeline with stricter thresholds, retrain tokenizer.
