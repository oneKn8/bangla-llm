#!/usr/bin/env python3
"""Validate tokenizer quality against a Bangla corpus.

Runs four tests:
  1. Common Bangla words should be single tokens.
  2. Tokens-per-word ratio on a corpus sample (target: 1.3-1.8).
  3. Encode-decode roundtrip on a reference sentence.
  4. Total token count across the FULL corpus (critical for epoch planning).

Usage:
    python test_tokenizer.py \
        --model /path/to/bangla_bpe_32k.model \
        --corpus /path/to/corpus.jsonl \
        --samples 1000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import sentencepiece as spm

SEED = 42

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
COMMON_WORDS: list[str] = [
    "বাংলাদেশ",
    "সরকার",
    "তিনি",
    "করা",
    "হয়েছে",
    "প্রথম",
    "বিশ্ব",
]

ROUNDTRIP_SENTENCE: str = "বাংলাদেশের রাজধানী ঢাকা।"


def load_model(model_path: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def sample_texts(corpus_path: str, n: int) -> list[str]:
    """Read JSONL corpus and return *n* randomly sampled text strings."""
    texts: list[str] = []
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            if text:
                texts.append(text)

    rng = random.Random(SEED)
    if len(texts) > n:
        texts = rng.sample(texts, n)
    return texts


# ---------------------------------------------------------------------------
# Test 1: Single-token common words
# ---------------------------------------------------------------------------
def test_single_token_words(sp: spm.SentencePieceProcessor) -> bool:
    """Check whether common Bangla words encode as a single token each."""
    print("-" * 60)
    print("TEST 1: Common Bangla words as single tokens")
    print("-" * 60)

    all_single = True
    for word in COMMON_WORDS:
        tokens = sp.encode(word, out_type=str)
        token_count = len(tokens)
        status = "OK" if token_count == 1 else "SPLIT"
        if token_count != 1:
            all_single = False
        print(f"  {word:<16}  tokens={token_count}  {tokens}  [{status}]")

    single_count = sum(
        1 for w in COMMON_WORDS if len(sp.encode(w, out_type=str)) == 1
    )
    print()
    print(
        f"  Result: {single_count}/{len(COMMON_WORDS)} words are single tokens."
    )
    passed = single_count >= len(COMMON_WORDS) // 2  # At least half should pass
    print(f"  Verdict: {'PASS' if passed else 'WARN'}")
    print()
    return passed


# ---------------------------------------------------------------------------
# Test 2: Tokens-per-word ratio
# ---------------------------------------------------------------------------
def test_tokens_per_word(
    sp: spm.SentencePieceProcessor,
    texts: list[str],
) -> bool:
    """Compute average tokens-per-word on sampled corpus texts."""
    print("-" * 60)
    print("TEST 2: Tokens-per-word ratio on corpus sample")
    print("-" * 60)

    total_tokens = 0
    total_words = 0

    for text in texts:
        words = text.split()
        if not words:
            continue
        ids = sp.encode(text, out_type=int)
        total_words += len(words)
        total_tokens += len(ids)

    if total_words == 0:
        print("  No words found in sample. SKIP.")
        print()
        return False

    ratio = total_tokens / total_words
    in_range = 1.3 <= ratio <= 1.8

    print(f"  Sampled documents: {len(texts)}")
    print(f"  Total words:       {total_words:,}")
    print(f"  Total tokens:      {total_tokens:,}")
    print(f"  Tokens/word:       {ratio:.4f}")
    print(f"  Target range:      1.3 - 1.8")
    print(f"  Verdict:           {'PASS' if in_range else 'WARN (out of range)'}")
    print()
    return in_range


# ---------------------------------------------------------------------------
# Test 3: Encode-decode roundtrip
# ---------------------------------------------------------------------------
def test_roundtrip(sp: spm.SentencePieceProcessor) -> bool:
    """Encode then decode a reference sentence and check for exact match."""
    print("-" * 60)
    print("TEST 3: Encode-decode roundtrip")
    print("-" * 60)

    ids = sp.encode(ROUNDTRIP_SENTENCE, out_type=int)
    tokens = sp.encode(ROUNDTRIP_SENTENCE, out_type=str)
    decoded = sp.decode(ids)

    match = decoded.strip() == ROUNDTRIP_SENTENCE.strip()

    print(f"  Input:   {ROUNDTRIP_SENTENCE}")
    print(f"  Tokens:  {tokens}")
    print(f"  IDs:     {ids}")
    print(f"  Decoded: {decoded}")
    print(f"  Match:   {match}")
    print(f"  Verdict: {'PASS' if match else 'FAIL'}")
    print()
    return match


# ---------------------------------------------------------------------------
# Test 4: Full corpus token count
# ---------------------------------------------------------------------------
def test_full_corpus_token_count(
    sp: spm.SentencePieceProcessor,
    corpus_path: str,
) -> int:
    """Count total tokens across the FULL corpus. Critical for epoch planning."""
    print("-" * 60)
    print("TEST 4: Full corpus token count")
    print("-" * 60)

    total_tokens = 0
    doc_count = 0
    bad_lines = 0
    t0 = time.monotonic()

    with open(corpus_path, "r", encoding="utf-8") as fh:
        batch: list[str] = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            text = obj.get("text", "")
            if not text:
                continue

            batch.append(text)
            doc_count += 1

            # Process in batches for efficiency
            if len(batch) >= 1000:
                encoded = sp.encode(batch, out_type=int)
                total_tokens += sum(len(ids) for ids in encoded)
                batch.clear()

        # Flush remaining
        if batch:
            encoded = sp.encode(batch, out_type=int)
            total_tokens += sum(len(ids) for ids in encoded)

    elapsed = time.monotonic() - t0

    print(f"  Corpus:         {corpus_path}")
    print(f"  Documents:      {doc_count:,}")
    print(f"  Skipped lines:  {bad_lines:,}")
    print(f"  Total tokens:   {total_tokens:,}")
    if doc_count > 0:
        print(f"  Avg tokens/doc: {total_tokens / doc_count:,.1f}")
    print(f"  Elapsed:        {elapsed:.1f}s")
    if elapsed > 0:
        docs_per_sec = doc_count / elapsed
        print(f"  Throughput:     {docs_per_sec:,.0f} docs/s")
    print()
    return total_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate tokenizer quality against a Bangla corpus.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the .model file",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to JSONL corpus",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of documents to sample for ratio test (default: 1000)",
    )
    args = parser.parse_args()

    sp = load_model(args.model)
    print()
    print("=" * 60)
    print("TOKENIZER QUALITY REPORT")
    print(f"Model: {args.model}")
    print(f"Vocab size: {sp.get_piece_size()}")
    print("=" * 60)
    print()

    texts = sample_texts(args.corpus, args.samples)

    r1 = test_single_token_words(sp)
    r2 = test_tokens_per_word(sp, texts)
    r3 = test_roundtrip(sp)
    total_tokens = test_full_corpus_token_count(sp, args.corpus)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Test 1 (single-token words):  {'PASS' if r1 else 'WARN'}")
    print(f"  Test 2 (tokens/word ratio):   {'PASS' if r2 else 'WARN'}")
    print(f"  Test 3 (roundtrip):           {'PASS' if r3 else 'FAIL'}")
    print(f"  Test 4 (total corpus tokens): {total_tokens:,}")
    print()

    if not r3:
        print("CRITICAL: Roundtrip test failed. Tokenizer may be broken.")
        sys.exit(1)

    if r1 and r2 and r3:
        print("All quality checks passed.")
    else:
        print("Some checks produced warnings. Review output above.")


if __name__ == "__main__":
    main()
