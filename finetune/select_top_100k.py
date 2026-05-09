#!/usr/bin/env python3
"""Select top 100K highest quality pairs from curated SFT data.

Quality scoring:
- Response length (longer = more detailed, up to a point)
- Bengali char ratio (higher = more native)
- Instruction length (moderate = better formed questions)
- Penalize very short responses even if above threshold
"""

import json
import random
import re
from pathlib import Path

BANGLA_RANGE = re.compile(r"[\u0980-\u09FF]")
SEED = 42
TARGET = 100_000
TRAIN_RATIO = 0.95


def bangla_ratio(text: str) -> float:
    if not text:
        return 0.0
    bangla = len(BANGLA_RANGE.findall(text))
    total = sum(1 for c in text if not c.isspace())
    return bangla / max(total, 1)


def quality_score(rec: dict) -> float:
    user = rec["messages"][0].get("content") or ""
    asst = rec["messages"][1].get("content") or ""

    # Bengali ratio in response (0-1, higher = better)
    br = bangla_ratio(asst)

    # Response length score (sweet spot: 200-2000 chars)
    rlen = len(asst)
    if rlen < 100:
        len_score = 0.2
    elif rlen < 300:
        len_score = 0.5
    elif rlen < 2000:
        len_score = 1.0
    else:
        len_score = 0.8  # very long, slightly penalize

    # Instruction quality (not too short, not too long)
    ilen = len(user)
    if ilen < 10:
        instr_score = 0.3
    elif ilen < 200:
        instr_score = 1.0
    else:
        instr_score = 0.7

    return (br * 0.5) + (len_score * 0.3) + (instr_score * 0.2)


def main():
    random.seed(SEED)
    data_dir = Path("finetune/data")

    # Load full curated train set
    train_path = data_dir / "kotha_sft_train.jsonl"
    records = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records):,} from train set")

    # Score all records
    scored = [(quality_score(r), r) for r in records]
    scored.sort(key=lambda x: -x[0])

    # Take top 100K
    top = [r for _, r in scored[:TARGET]]
    print(f"Selected top {len(top):,} by quality score")

    # Score distribution
    scores = [s for s, _ in scored[:TARGET]]
    print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
    print(f"Mean score: {sum(scores)/len(scores):.3f}")

    # Shuffle for training
    random.shuffle(top)

    # Split
    n_train = int(len(top) * TRAIN_RATIO)
    train = top[:n_train]
    val = top[n_train:]

    # Write
    for name, data in [("train", train), ("val", val)]:
        out = data_dir / f"kotha_top100k_{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data):,} to {out}")

    # Keep original test set as-is
    print(f"Test set unchanged: {data_dir / 'kotha_sft_test.jsonl'}")

    # Sample check
    print(f"\n--- Top 5 quality samples ---\n")
    for score, rec in scored[:5]:
        q = rec["messages"][0]["content"][:80]
        a = rec["messages"][1]["content"][:100]
        print(f"  [{score:.3f}] Q: {q}")
        print(f"           A: {a}...")
        print()


if __name__ == "__main__":
    main()
