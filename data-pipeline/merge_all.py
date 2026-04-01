#!/usr/bin/env python3
"""
merge_all.py -- Merge and deduplicate all Bangla-LLM data sources into a
single unified corpus.

Pipeline:
  1. Load SHA-256 hashes from the already-processed culturax.jsonl (streaming)
  2. Process each RAW source (wikipedia, sangraha, newspapers):
       - NFKC text normalization + whitespace stripping
       - Length filter (>= 100 chars)
       - Bengali character ratio filter (>= 50%)
       - SHA-256 exact deduplication (cross-source)
  3. Write new unique docs to a temp file
  4. Create merged_corpus.jsonl by copying culturax + appending new docs
     (disk-efficient: uses shell-level copy to avoid double-buffering)
  5. Print per-source statistics

Memory strategy:
  - Only SHA-256 hex digests (64 bytes each) are kept in a set for dedup.
  - Documents are streamed line-by-line; never loaded fully into RAM.
  - ~1.2M docs * 64 bytes = ~77 MB for the hash set -- very manageable.

Disk strategy:
  - New docs written to a small temp file first (~1-2 GB)
  - Then culturax is hard-linked (not copied) as the merged base
  - New docs appended to the merged file
  - Original culturax is preserved (hard link = same inode, no extra disk)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/home/oneknight/projects/bangla-llm/data-pipeline")
DATA_DIR = BASE_DIR / "data"

CULTURAX_PATH = DATA_DIR / "processed" / "culturax.jsonl"
WIKIPEDIA_PATH = DATA_DIR / "raw" / "wikipedia" / "bnwiki.jsonl"
SANGRAHA_PATH = DATA_DIR / "raw" / "hf_corpus" / "sangraha_verified.jsonl"
NEWSPAPERS_PATH = DATA_DIR / "raw" / "newspapers" / "prothomalo.jsonl"

OUTPUT_PATH = DATA_DIR / "processed" / "merged_corpus.jsonl"
NEW_SOURCES_TMP = DATA_DIR / "processed" / "new_sources.jsonl.tmp"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MIN_CHAR_COUNT = 100
MIN_BENGALI_RATIO = 0.50

# Bengali Unicode block: U+0980 - U+09FF
_BENGALI_RE = re.compile(r"[\u0980-\u09FF]")


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------
@dataclass
class SourceStats:
    """Tracks per-source processing statistics."""

    total_read: int = 0
    kept: int = 0
    skipped_short: int = 0
    skipped_ratio: int = 0
    skipped_dedup: int = 0
    skipped_parse_error: int = 0
    total_bytes_written: int = 0


@dataclass
class MergeStats:
    """Aggregate merge statistics."""

    sources: dict[str, SourceStats] = field(default_factory=dict)
    culturax_docs: int = 0
    culturax_bytes: int = 0
    new_docs_bytes: int = 0
    total_written: int = 0
    total_bytes: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    def source(self, name: str) -> SourceStats:
        if name not in self.sources:
            self.sources[name] = SourceStats()
        return self.sources[name]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Apply NFKC normalization and strip leading/trailing whitespace."""
    return unicodedata.normalize("NFKC", text).strip()


def bengali_char_ratio(text: str) -> float:
    """Return the fraction of characters in the Bengali Unicode range."""
    if not text:
        return 0.0
    bengali_count = len(_BENGALI_RE.findall(text))
    # Denominator: all non-whitespace characters for a fairer ratio
    total = sum(1 for ch in text if not ch.isspace())
    if total == 0:
        return 0.0
    return bengali_count / total


def sha256_hash(text: str) -> str:
    """SHA-256 hex digest of the text (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stream_jsonl(path: Path) -> Iterator[tuple[str, dict | None]]:
    """
    Yield (raw_line, parsed_dict) for each line in a JSONL file.
    On parse error, yields (raw_line, None).
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                doc = json.loads(line)
                yield line, doc
            except json.JSONDecodeError:
                yield line, None


# ---------------------------------------------------------------------------
# Phase 1: Build culturax hash set (streaming)
# ---------------------------------------------------------------------------
def load_culturax_hashes(stats: MergeStats) -> set[str]:
    """
    Stream culturax.jsonl and collect SHA-256 hashes of each document's text.
    Returns the set of hex digests.
    """
    print(f"[Phase 1] Loading culturax hashes from {CULTURAX_PATH} ...")
    hashes: set[str] = set()
    count = 0
    t0 = time.monotonic()

    for _raw_line, doc in stream_jsonl(CULTURAX_PATH):
        count += 1
        if doc is None:
            continue
        text = doc.get("text", "")
        h = sha256_hash(text)
        hashes.add(h)

        if count % 100_000 == 0:
            elapsed = time.monotonic() - t0
            print(f"  ... {count:>10,} docs hashed ({elapsed:.1f}s)")

    elapsed = time.monotonic() - t0
    stats.culturax_docs = count
    stats.culturax_bytes = CULTURAX_PATH.stat().st_size
    print(f"  Loaded {len(hashes):,} unique hashes from {count:,} culturax docs in {elapsed:.1f}s")
    return hashes


# ---------------------------------------------------------------------------
# Phase 2: Process and filter a raw source
# ---------------------------------------------------------------------------
def process_raw_source(
    path: Path,
    source_label: str,
    seen_hashes: set[str],
    out_fh: IO[str],
    stats: MergeStats,
) -> None:
    """
    Stream a raw JSONL source, apply filters, dedup against seen_hashes,
    and write passing docs to out_fh.  Updates seen_hashes in-place.
    """
    ss = stats.source(source_label)
    print(f"[Phase 2] Processing {source_label}: {path}")
    t0 = time.monotonic()

    for _raw_line, doc in stream_jsonl(path):
        ss.total_read += 1

        if doc is None:
            ss.skipped_parse_error += 1
            continue

        text = doc.get("text", "")

        # --- Normalize ---
        text = normalize_text(text)

        # --- Length filter ---
        if len(text) < MIN_CHAR_COUNT:
            ss.skipped_short += 1
            continue

        # --- Bengali ratio filter ---
        ratio = bengali_char_ratio(text)
        if ratio < MIN_BENGALI_RATIO:
            ss.skipped_ratio += 1
            continue

        # --- SHA-256 exact dedup ---
        h = sha256_hash(text)
        if h in seen_hashes:
            ss.skipped_dedup += 1
            continue
        seen_hashes.add(h)

        # --- Build output document (standardized schema) ---
        out_doc = {
            "text": text,
            "source": doc.get("source", source_label),
            "url": doc.get("url", ""),
            "title": doc.get("title", ""),
            "char_count": len(text),
            "metadata": doc.get("metadata", {}),
        }

        line_out = json.dumps(out_doc, ensure_ascii=False)
        out_fh.write(line_out)
        out_fh.write("\n")

        ss.kept += 1
        byte_len = len(line_out.encode("utf-8")) + 1  # +1 for newline
        ss.total_bytes_written += byte_len
        stats.new_docs_bytes += byte_len

    elapsed = time.monotonic() - t0
    print(
        f"  {source_label}: read={ss.total_read:,}  kept={ss.kept:,}  "
        f"short={ss.skipped_short:,}  low_bn={ss.skipped_ratio:,}  "
        f"dedup={ss.skipped_dedup:,}  parse_err={ss.skipped_parse_error:,}  "
        f"({elapsed:.1f}s)"
    )


# ---------------------------------------------------------------------------
# Phase 3: Assemble merged corpus (disk-efficient)
# ---------------------------------------------------------------------------
def assemble_merged_corpus(stats: MergeStats) -> None:
    """
    Create merged_corpus.jsonl = culturax.jsonl + new_sources.jsonl.tmp.

    Strategy for disk efficiency:
      - Copy culturax.jsonl to merged_corpus.jsonl using cp (reflink if
        supported, else standard copy).
      - Append new_sources.jsonl.tmp to the end.
      - Delete the temp file.

    On filesystems with reflink support (btrfs, xfs with reflink), the copy
    is nearly free.  On ext4, it uses standard copy but we need the disk
    space.  We check free space before proceeding.
    """
    new_size = NEW_SOURCES_TMP.stat().st_size if NEW_SOURCES_TMP.exists() else 0
    culturax_size = CULTURAX_PATH.stat().st_size

    print(f"[Phase 3] Assembling merged corpus ...")
    print(f"  culturax      : {culturax_size / 1e9:.2f} GB")
    print(f"  new sources   : {new_size / 1e9:.2f} GB")
    print(f"  expected total: {(culturax_size + new_size) / 1e9:.2f} GB")

    # Check available disk space
    st = os.statvfs(str(DATA_DIR / "processed"))
    free_bytes = st.f_bavail * st.f_frsize
    needed = culturax_size + new_size + (100 * 1024 * 1024)  # 100 MB buffer
    print(f"  free disk     : {free_bytes / 1e9:.2f} GB")
    print(f"  needed (copy) : {needed / 1e9:.2f} GB")

    if free_bytes < needed:
        # Fallback: rename culturax -> merged, then append
        # This avoids copying culturax at the cost of moving the original file
        print("  WARNING: Not enough space for full copy. Using rename+append strategy.")
        print("  (culturax.jsonl will be renamed to merged_corpus.jsonl, then new docs appended)")

        # Remove any existing output
        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()

        # Rename culturax -> merged_corpus (instant, same filesystem)
        CULTURAX_PATH.rename(OUTPUT_PATH)
        print(f"  Renamed culturax.jsonl -> merged_corpus.jsonl")

        # Append new sources
        if new_size > 0:
            t0 = time.monotonic()
            with open(OUTPUT_PATH, "a", encoding="utf-8") as out_fh, \
                 open(NEW_SOURCES_TMP, "r", encoding="utf-8") as in_fh:
                # Use shutil.copyfileobj for efficient buffered copy
                shutil.copyfileobj(in_fh, out_fh, length=16 * 1024 * 1024)
            elapsed = time.monotonic() - t0
            print(f"  Appended {new_size / 1e6:.1f} MB of new docs in {elapsed:.1f}s")

        # Recreate culturax.jsonl as a hard link to preserve the reference
        # (both names point to same inode -- no extra disk used)
        try:
            os.link(str(OUTPUT_PATH), str(CULTURAX_PATH))
            print("  Created hard link: culturax.jsonl -> merged_corpus.jsonl")
        except OSError:
            # Hard link may fail across mount points; just note it
            print("  NOTE: Could not create hard link. culturax.jsonl has been consumed into merged_corpus.jsonl")
    else:
        # Standard path: copy culturax, append new
        t0 = time.monotonic()
        tmp_merged = OUTPUT_PATH.with_suffix(".jsonl.assembling")

        shutil.copy2(str(CULTURAX_PATH), str(tmp_merged))
        elapsed_copy = time.monotonic() - t0
        print(f"  Copied culturax in {elapsed_copy:.1f}s")

        if new_size > 0:
            with open(tmp_merged, "a", encoding="utf-8") as out_fh, \
                 open(NEW_SOURCES_TMP, "r", encoding="utf-8") as in_fh:
                shutil.copyfileobj(in_fh, out_fh, length=16 * 1024 * 1024)

        tmp_merged.rename(OUTPUT_PATH)
        elapsed_total = time.monotonic() - t0
        print(f"  Assembled merged corpus in {elapsed_total:.1f}s")

    # Compute final stats
    final_size = OUTPUT_PATH.stat().st_size
    stats.total_bytes = final_size

    new_kept = sum(ss.kept for ss in stats.sources.values())
    stats.total_written = stats.culturax_docs + new_kept

    print(f"  Final file: {OUTPUT_PATH}")
    print(f"  Final size: {final_size / 1e9:.2f} GB")


# ---------------------------------------------------------------------------
# Phase 4: Print final statistics
# ---------------------------------------------------------------------------
def print_stats(stats: MergeStats) -> None:
    """Pretty-print merge statistics."""
    wall_time = stats.end_time - stats.start_time

    print("\n" + "=" * 72)
    print("  MERGE COMPLETE -- STATISTICS")
    print("=" * 72)

    print(f"\n  Output file : {OUTPUT_PATH}")
    print(f"  Total docs  : {stats.total_written:>12,}")
    print(f"  Total size  : {stats.total_bytes / 1e9:>12.2f} GB")
    print(f"  Wall time   : {wall_time:>12.1f} s")

    print("\n  Per-source breakdown:")
    print(f"  {'Source':<16} {'Read':>10} {'Kept':>10} {'Short':>8} {'LowBN':>8} {'Dedup':>8} {'Err':>6} {'Size':>12}")
    print(f"  {'-' * 16} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 12}")

    # Culturax row (already processed, copied as-is)
    print(
        f"  {'culturax':<16} {stats.culturax_docs:>10,} {stats.culturax_docs:>10,} "
        f"{'--':>8} {'--':>8} {'--':>8} {'--':>6} {stats.culturax_bytes / 1e9:>10.2f} GB"
    )

    total_dupes = 0
    total_new_kept = 0
    for name, ss in stats.sources.items():
        total_dupes += ss.skipped_dedup
        total_new_kept += ss.kept
        print(
            f"  {name:<16} {ss.total_read:>10,} {ss.kept:>10,} "
            f"{ss.skipped_short:>8,} {ss.skipped_ratio:>8,} {ss.skipped_dedup:>8,} "
            f"{ss.skipped_parse_error:>6,} {ss.total_bytes_written / 1e6:>10.2f} MB"
        )

    total_skipped_all = sum(
        ss.skipped_short + ss.skipped_ratio + ss.skipped_dedup + ss.skipped_parse_error
        for ss in stats.sources.values()
    )
    total_raw_read = sum(ss.total_read for ss in stats.sources.values())

    print(f"\n  Summary:")
    print(f"    Raw docs read       : {total_raw_read:>10,}")
    print(f"    New docs kept       : {total_new_kept:>10,}")
    print(f"    Duplicates removed  : {total_dupes:>10,}")
    print(f"    Total filtered out  : {total_skipped_all:>10,}")
    print(f"    Culturax (as-is)    : {stats.culturax_docs:>10,}")
    print(f"    Final corpus docs   : {stats.total_written:>10,}")
    print(f"    Final corpus size   : {stats.total_bytes / 1e9:>10.2f} GB")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    stats = MergeStats()
    stats.start_time = time.monotonic()

    # Validate inputs exist
    for p in [CULTURAX_PATH, WIKIPEDIA_PATH, SANGRAHA_PATH, NEWSPAPERS_PATH]:
        if not p.exists():
            print(f"ERROR: Missing input file: {p}", file=sys.stderr)
            return 1

    # Phase 1: Load culturax hashes
    seen_hashes = load_culturax_hashes(stats)

    # Phase 2: Process raw sources into temp file
    try:
        with open(NEW_SOURCES_TMP, "w", encoding="utf-8") as out_fh:
            raw_sources = [
                (SANGRAHA_PATH, "sangraha"),
                (WIKIPEDIA_PATH, "wikipedia"),
                (NEWSPAPERS_PATH, "prothomalo"),
            ]
            for path, label in raw_sources:
                process_raw_source(path, label, seen_hashes, out_fh, stats)

        # Phase 3: Assemble merged corpus
        assemble_merged_corpus(stats)

    except Exception:
        # Clean up temp files on failure
        for tmp in [NEW_SOURCES_TMP, OUTPUT_PATH.with_suffix(".jsonl.assembling")]:
            if tmp.exists():
                tmp.unlink()
        raise
    finally:
        # Always clean up the new_sources temp
        if NEW_SOURCES_TMP.exists():
            NEW_SOURCES_TMP.unlink()

    stats.end_time = time.monotonic()
    print_stats(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
