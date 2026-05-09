#!/usr/bin/env python3
"""Curate and split SFT data for Kotha-1 fine-tuning.

Merges all SFT data sources, applies quality filters, deduplicates,
and creates train/val/test splits.

Usage:
    python finetune/curate_sft_data.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

BANGLA_RANGE = re.compile(r"[\u0980-\u09FF]")
DATA_DIR = Path("finetune/data")
OUTPUT_PREFIX = "kotha_sft"

# Quality thresholds
MIN_RESPONSE_CHARS = 50
MAX_RESPONSE_CHARS = 4096
MIN_BANGLA_RATIO = 0.4  # 40% Bengali chars in response
TRAIN_RATIO = 0.95
VAL_RATIO = 0.025
TEST_RATIO = 0.025
SEED = 42


def bangla_char_ratio(text: str) -> float:
    """Fraction of characters that are Bengali script."""
    if not text:
        return 0.0
    bangla_count = len(BANGLA_RANGE.findall(text))
    # Only count non-whitespace, non-punctuation chars
    alpha_count = sum(1 for c in text if not c.isspace() and c not in ".,;:!?-()[]{}\"'`/\\|@#$%^&*+=~<>0123456789")
    if alpha_count == 0:
        return 0.0
    return bangla_count / alpha_count


def normalize_instruction(text: str) -> str:
    """Normalize instruction for dedup comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())[:80]


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file, return list of message dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "messages" in obj and len(obj["messages"]) >= 2:
                    records.append(obj)
            except json.JSONDecodeError:
                continue
    return records


def main() -> None:
    random.seed(SEED)

    # --- Load all sources ---
    sources = {}
    for jsonl_file in sorted(DATA_DIR.glob("sft_*.jsonl")):
        records = load_jsonl(jsonl_file)
        sources[jsonl_file.name] = len(records)
        print(f"  Loaded {len(records):>7,} from {jsonl_file.name}")

    all_records = []
    for jsonl_file in sorted(DATA_DIR.glob("sft_*.jsonl")):
        all_records.extend(load_jsonl(jsonl_file))

    print(f"\nTotal raw: {len(all_records):,}")

    # --- Filter ---
    stats = Counter()
    seen_instructions: set[str] = set()
    filtered: list[dict] = []

    for rec in all_records:
        user_msg = rec["messages"][0].get("content") or ""
        asst_msg = rec["messages"][1].get("content") or ""

        # Response length
        if len(asst_msg) < MIN_RESPONSE_CHARS:
            stats["too_short"] += 1
            continue

        if len(asst_msg) > MAX_RESPONSE_CHARS:
            stats["too_long"] += 1
            continue

        # Bengali ratio
        ratio = bangla_char_ratio(asst_msg)
        if ratio < MIN_BANGLA_RATIO:
            stats["low_bangla"] += 1
            continue

        # Dedup by instruction (normalized)
        norm = normalize_instruction(user_msg)
        if norm in seen_instructions:
            stats["duplicate"] += 1
            continue
        seen_instructions.add(norm)

        # Empty content check
        if not user_msg.strip() or not asst_msg.strip():
            stats["empty"] += 1
            continue

        filtered.append(rec)

    print(f"\nFiltering results:")
    for reason, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count:,} removed")
    print(f"\nAfter filtering: {len(filtered):,}")

    # --- Shuffle and split ---
    random.shuffle(filtered)

    n = len(filtered)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # Test gets the rest
    train = filtered[:n_train]
    val = filtered[n_train : n_train + n_val]
    test = filtered[n_train + n_val :]

    print(f"\nSplits:")
    print(f"  Train: {len(train):,}")
    print(f"  Val:   {len(val):,}")
    print(f"  Test:  {len(test):,}")

    # --- Write outputs ---
    for name, data in [("train", train), ("val", val), ("test", test)]:
        out_path = DATA_DIR / f"{OUTPUT_PREFIX}_{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Wrote {out_path}")

    # --- Quality sample ---
    print(f"\n--- Random sample (10 pairs) ---\n")
    for rec in random.sample(train, min(10, len(train))):
        q = rec["messages"][0]["content"][:80]
        a = rec["messages"][1]["content"][:120]
        print(f"  Q: {q}")
        print(f"  A: {a}...")
        print()

    # --- Stats ---
    response_lengths = [len(r["messages"][1]["content"]) for r in train]
    avg_len = sum(response_lengths) / len(response_lengths)
    print(f"Avg response length: {avg_len:.0f} chars")
    print(f"Min response length: {min(response_lengths)} chars")
    print(f"Max response length: {max(response_lengths)} chars")


if __name__ == "__main__":
    main()
