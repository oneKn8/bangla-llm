#!/usr/bin/env python3
"""Train a 32K BPE tokenizer for Bangla using SentencePiece.

Reads a JSONL corpus (each line: {"text": "..."}), samples ~N GB of text,
trains a byte-fallback BPE model with identity normalization, and writes
the .model + .vocab files to the output directory.

Usage:
    python train_tokenizer.py \
        --corpus /path/to/corpus.jsonl \
        --output-dir /path/to/output \
        --vocab-size 32000 \
        --sample-gb 1.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
from pathlib import Path

import sentencepiece as spm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPECIAL_TOKENS: list[str] = [
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|system|>",
    "<|pad|>",
    "<|sep|>",
    "<|cls|>",
    "<|mask|>",
]

SEED = 42
BYTES_PER_GB = 1 << 30  # 1 GiB


def sample_corpus(
    corpus_path: str,
    sample_bytes: int,
    seed: int = SEED,
) -> list[str]:
    """Read JSONL, extract text fields, shuffle, and return up to *sample_bytes* of text.

    Lines are shuffled deterministically so the sample is reproducible.
    """
    log.info("Reading corpus from %s ...", corpus_path)

    texts: list[str] = []
    bad_lines = 0
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                bad_lines += 1
                if bad_lines <= 5:
                    log.warning("Skipping malformed JSON at line %d", line_no)
                continue
            text = obj.get("text", "")
            if text:
                texts.append(text)

    if bad_lines > 5:
        log.warning("Total malformed lines skipped: %d", bad_lines)
    log.info("Loaded %d documents from corpus.", len(texts))

    if not texts:
        log.error("Corpus contains no usable text. Aborting.")
        sys.exit(1)

    rng = random.Random(seed)
    rng.shuffle(texts)

    sampled: list[str] = []
    total = 0
    for t in texts:
        encoded_len = len(t.encode("utf-8"))
        sampled.append(t)
        total += encoded_len
        if total >= sample_bytes:
            break

    actual_gb = total / BYTES_PER_GB
    log.info(
        "Sampled %d documents (%.2f GB of %.2f GB target).",
        len(sampled),
        actual_gb,
        sample_bytes / BYTES_PER_GB,
    )
    return sampled


def write_sample_file(texts: list[str], path: str) -> None:
    """Write sampled texts to a plain-text file, one document per line."""
    log.info("Writing sample file to %s ...", path)
    with open(path, "w", encoding="utf-8") as fh:
        for text in texts:
            # Replace newlines within a document so each doc is one line.
            fh.write(text.replace("\n", " ") + "\n")
    size_mb = os.path.getsize(path) / (1 << 20)
    log.info("Sample file written: %.1f MB", size_mb)


def train(
    sample_path: str,
    output_dir: str,
    vocab_size: int,
) -> Path:
    """Run SentencePiece BPE training and return the path to the .model file."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    model_prefix = str(output_dir_path / "bangla_bpe_32k")

    log.info(
        "Starting SentencePiece training (vocab_size=%d) ...",
        vocab_size,
    )

    spm.SentencePieceTrainer.train(
        input=sample_path,
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
        # Special token IDs
        pad_id=3,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        # User-defined symbols
        user_defined_symbols=SPECIAL_TOKENS,
    )

    model_path = Path(f"{model_prefix}.model")
    vocab_path = Path(f"{model_prefix}.vocab")

    if not model_path.exists():
        log.error("Training failed: %s not found.", model_path)
        sys.exit(1)

    log.info("Model saved: %s", model_path)
    log.info("Vocab saved: %s", vocab_path)
    return model_path


def verify_model(model_path: Path) -> None:
    """Quick sanity check: load model and encode a test sentence."""
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    test = "বাংলাদেশের রাজধানী ঢাকা।"
    tokens = sp.encode(test, out_type=str)
    ids = sp.encode(test, out_type=int)
    decoded = sp.decode(ids)

    log.info("Verification:")
    log.info("  Input:   %s", test)
    log.info("  Tokens:  %s", tokens)
    log.info("  IDs:     %s", ids)
    log.info("  Decoded: %s", decoded)
    log.info("  Vocab size: %d", sp.get_piece_size())

    if decoded.strip() != test.strip():
        log.warning("Roundtrip mismatch! decoded=%r", decoded)
    else:
        log.info("  Roundtrip: OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a 32K BPE tokenizer for Bangla using SentencePiece.",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to JSONL corpus (each line: {\"text\": \"...\"})",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write .model and .vocab files",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000)",
    )
    parser.add_argument(
        "--sample-gb",
        type=float,
        default=1.5,
        help="Approximate GB of text to sample for training (default: 1.5)",
    )
    args = parser.parse_args()

    corpus_path = os.path.abspath(args.corpus)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isfile(corpus_path):
        log.error("Corpus file not found: %s", corpus_path)
        sys.exit(1)

    sample_bytes = int(args.sample_gb * BYTES_PER_GB)

    # Sample from the corpus
    texts = sample_corpus(corpus_path, sample_bytes)

    # Write to a temporary plain-text file for SentencePiece
    sample_fd, sample_path = tempfile.mkstemp(
        suffix=".txt", prefix="bangla_sp_sample_"
    )
    os.close(sample_fd)

    try:
        write_sample_file(texts, sample_path)
        model_path = train(sample_path, output_dir, args.vocab_size)
        verify_model(model_path)
    finally:
        # Clean up the temporary sample file
        if os.path.exists(sample_path):
            os.remove(sample_path)
            log.info("Cleaned up sample file: %s", sample_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
