"""Dataset utilities for Bangla-LLM pre-training.

Provides corpus pre-tokenization to binary format and a memory-mapped
IterableDataset for efficient streaming during training.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import IterableDataset


def tokenize_corpus(
    corpus_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
) -> int:
    """Pre-tokenize an entire JSONL corpus to a binary uint16 file.

    Each document is encoded as: [bos_id] + tokenized_text + [eos_id].
    Tokens are written as little-endian uint16 values (valid for vocab < 65536).

    Args:
        corpus_path: Path to input JSONL file. Each line must have a "text" field.
        tokenizer_path: Path to a trained SentencePiece .model file.
        output_path: Path for the output binary token file.

    Returns:
        Total number of tokens written.
    """
    corpus_path = Path(corpus_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_path))

    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    total_tokens = 0
    doc_count = 0

    with open(corpus_path, "r", encoding="utf-8") as fin, open(output_path, "wb") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)
            text = doc.get("text", "")
            if not text:
                continue

            token_ids = sp.Encode(text, out_type=int)
            # Prepend BOS, append EOS
            token_ids = [bos_id] + token_ids + [eos_id]

            # Write as uint16 little-endian
            fout.write(struct.pack(f"<{len(token_ids)}H", *token_ids))
            total_tokens += len(token_ids)
            doc_count += 1

            if doc_count % 100_000 == 0:
                print(
                    f"  [{doc_count:,} docs] {total_tokens:,} tokens written",
                    flush=True,
                )

    print(f"Done: {doc_count:,} docs, {total_tokens:,} tokens -> {output_path}")
    return total_tokens


class TokenDataset(IterableDataset):
    """Memory-mapped dataset over a pre-tokenized binary uint16 file.

    Yields contiguous sequences of (input_ids, labels) where labels are
    input_ids shifted right by one position, suitable for causal LM training.

    The dataset shuffles the starting offset per epoch using the epoch number
    as a random seed, ensuring different sequence boundaries each epoch while
    remaining deterministic.

    Args:
        token_file: Path to binary uint16 token file produced by tokenize_corpus.
        seq_len: Sequence length for each training sample (default 2048).
        epoch: Current epoch number, used as shuffle seed (default 0).
    """

    def __init__(
        self,
        token_file: str | Path,
        seq_len: int = 2048,
        epoch: int = 0,
    ) -> None:
        super().__init__()
        self.token_file = Path(token_file)
        self.seq_len = seq_len
        self.epoch = epoch

        # Memory-map the file as uint16
        self.tokens = np.memmap(
            self.token_file, dtype=np.uint16, mode="r",
        )
        self.n_tokens = len(self.tokens)

        # We need seq_len + 1 tokens per sample (input + 1 shifted label)
        self.n_sequences = max(0, (self.n_tokens - 1) // self.seq_len)

    def __len__(self) -> int:
        """Approximate number of sequences in the dataset."""
        return self.n_sequences

    def __iter__(self):
        """Yield (input_ids, labels) pairs with epoch-based offset shuffling."""
        # Compute a deterministic random offset for this epoch
        rng = np.random.RandomState(seed=self.epoch)
        offset = rng.randint(0, min(self.seq_len, self.n_tokens))

        # Generate all valid start indices from the offset
        starts = list(range(offset, self.n_tokens - self.seq_len, self.seq_len))

        # Shuffle the order of sequences within the epoch
        rng.shuffle(starts)

        for start in starts:
            # Read seq_len + 1 tokens: first seq_len are input, last seq_len are labels
            chunk = self.tokens[start : start + self.seq_len + 1].astype(np.int64)
            input_ids = torch.from_numpy(chunk[:-1].copy())
            labels = torch.from_numpy(chunk[1:].copy())

            yield {"input_ids": input_ids, "labels": labels}
