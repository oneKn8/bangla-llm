"""Deduplication: SHA-256 exact dedup + MinHash LSH near-dedup.

Two passes:
1. Exact dedup via SHA-256 hash of full text
2. Near-dedup via MinHash LSH (5-char shingles, Jaccard threshold 0.8)

Designed to run per-source first, then cross-source.
State is disk-backed for large datasets that don't fit in RAM.
"""

import hashlib
import pickle
from pathlib import Path

from datasketch import MinHash, MinHashLSH

from config import (
    MINHASH_NUM_PERM,
    MINHASH_SHINGLE_SIZE,
    MINHASH_THRESHOLD,
    STATE_DIR,
)


class Deduplicator:
    """Two-pass deduplication engine."""

    def __init__(self, name: str = "global"):
        """
        Args:
            name: Identifier for this dedup session (e.g., "wikipedia", "global")
        """
        self.name = name
        self.state_path = STATE_DIR / f"dedup_{name}.pkl"
        self.exact_hashes: set[str] = set()
        self.lsh = MinHashLSH(
            threshold=MINHASH_THRESHOLD,
            num_perm=MINHASH_NUM_PERM,
        )
        self._lsh_count = 0
        self.stats = {
            "total": 0,
            "exact_dupes": 0,
            "near_dupes": 0,
            "kept": 0,
        }

    def load_state(self) -> None:
        """Load saved dedup state from disk."""
        if self.state_path.exists():
            with open(self.state_path, "rb") as f:
                data = pickle.load(f)
                self.exact_hashes = data.get("exact_hashes", set())
                self.stats = data.get("stats", self.stats)
                print(
                    f"[dedup:{self.name}] Loaded state: "
                    f"{len(self.exact_hashes)} hashes, {self.stats}"
                )

    def save_state(self) -> None:
        """Save dedup state to disk. LSH index is not serialized."""
        with open(self.state_path, "wb") as f:
            pickle.dump(
                {"exact_hashes": self.exact_hashes, "stats": self.stats},
                f,
            )

    def is_duplicate(self, text: str, doc_id: str = "") -> tuple[bool, str]:
        """Check if text is a duplicate.

        Args:
            text: Document text
            doc_id: Unique document ID (used as LSH key)

        Returns:
            Tuple of (is_dup, reason)
        """
        self.stats["total"] += 1

        # Pass 1: Exact dedup via SHA-256
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in self.exact_hashes:
            self.stats["exact_dupes"] += 1
            return True, "exact_duplicate"
        self.exact_hashes.add(text_hash)

        # Pass 2: Near-dedup via MinHash LSH
        minhash = self._compute_minhash(text)
        try:
            result = self.lsh.query(minhash)
            if result:
                self.stats["near_dupes"] += 1
                return True, f"near_duplicate_of:{result[0]}"
        except ValueError:
            # LSH can throw if key already exists
            pass

        # Insert into LSH index
        key = doc_id or text_hash[:16]
        try:
            self.lsh.insert(key, minhash)
            self._lsh_count += 1
        except ValueError:
            # Duplicate key - already indexed
            pass

        self.stats["kept"] += 1
        return False, ""

    @staticmethod
    def _compute_minhash(text: str) -> MinHash:
        """Compute MinHash signature from character shingles."""
        m = MinHash(num_perm=MINHASH_NUM_PERM)
        # Generate character n-grams (shingles)
        for i in range(len(text) - MINHASH_SHINGLE_SIZE + 1):
            shingle = text[i : i + MINHASH_SHINGLE_SIZE]
            m.update(shingle.encode("utf-8"))
        return m

    def get_stats(self) -> dict:
        return dict(self.stats)


def dedup_documents(
    docs: list[dict],
    deduplicator: Deduplicator | None = None,
) -> list[dict]:
    """Deduplicate a list of documents.

    Args:
        docs: List of document dicts
        deduplicator: Existing deduplicator instance, or None to create new

    Returns:
        List of unique documents
    """
    if deduplicator is None:
        deduplicator = Deduplicator()

    kept = []
    for doc in docs:
        is_dup, reason = deduplicator.is_duplicate(
            doc["text"], doc_id=doc.get("id", "")
        )
        if not is_dup:
            kept.append(doc)

    return kept
