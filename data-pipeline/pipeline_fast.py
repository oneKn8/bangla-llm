"""Fast pipeline for pre-cleaned datasets (CulturaX, etc).

Only runs: normalize + SHA-256 exact dedup.
Skips: fasttext lang detection, quality filters, MinHash near-dedup.

Rationale: CulturaX already applied language ID, IQR quality filtering,
URL-based exact dedup, AND MinHash LSH near-dedup (5-gram shingles,
Jaccard 0.8 threshold) -- same params as our pipeline. Re-running those
is redundant. We still normalize (Bengali-specific bnorm + cleanup) and
SHA-256 exact dedup (catches content-identical docs with different URLs
that CulturaX's URL-based dedup missed).

Design: Single-pass streaming to avoid needing 2x disk space.
Reads input sequentially, normalizes in-process, SHA-256 dedup in-line,
writes output directly. No temp chunks, no splitting.

Usage:
    python pipeline_fast.py                              # defaults
    python pipeline_fast.py --input FILE                 # specific input
    python pipeline_fast.py --output-dir /path/to/dir    # output elsewhere (e.g. Drive)
    python pipeline_fast.py --workers 8                  # parallel workers
"""

import argparse
import hashlib
import multiprocessing as mp
import os
import sys
from pathlib import Path

import orjson
from tqdm import tqdm

from config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs
from processing.stats import StatsCollector

DEFAULT_WORKERS = 4
BATCH_SIZE = 10_000  # docs per batch sent to workers


def _normalize_batch(docs: list[bytes]) -> list[bytes | None]:
    """Normalize a batch of docs. Runs in a worker process.

    Returns list of serialized docs (or None if doc should be dropped).
    """
    from processing.normalize import normalize_document

    results = []
    for raw_line in docs:
        try:
            doc = orjson.loads(raw_line)
        except orjson.JSONDecodeError:
            results.append(None)
            continue

        doc = normalize_document(doc)

        if not doc["text"].strip() or doc["char_count"] < 50:
            results.append(None)
            continue

        results.append(orjson.dumps(doc))
    return results


def process_file(
    input_path: Path,
    output_path: Path,
    num_workers: int,
    stats: StatsCollector,
) -> int:
    """Process a single JSONL file: streaming normalize + SHA-256 dedup.

    Returns number of docs kept.
    """
    file_size = input_path.stat().st_size
    if file_size == 0:
        print(f"[pipeline-fast] {input_path.name}: empty, skipping")
        return 0

    print(f"\n--- {input_path.name} ({file_size / (1024**3):.1f}GB) ---")
    print(f"[pipeline-fast] Output: {output_path}")

    seen_hashes: set[str] = set()
    kept = 0
    total = 0
    dropped_normalize = 0
    exact_dupes = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with mp.Pool(num_workers) as pool, \
         open(input_path, "rb") as fin, \
         open(output_path, "wb") as fout:

        batch: list[bytes] = []
        pbar = tqdm(fin, desc="Processing", unit=" lines")

        for line in pbar:
            line = line.strip()
            if not line:
                continue

            batch.append(line)

            if len(batch) >= BATCH_SIZE:
                _flush_batch(
                    batch, pool, fout, seen_hashes, stats,
                    counters := {"kept": 0, "total": 0, "dropped": 0, "dupes": 0},
                )
                kept += counters["kept"]
                total += counters["total"]
                dropped_normalize += counters["dropped"]
                exact_dupes += counters["dupes"]
                batch = []

                pbar.set_postfix(
                    kept=f"{kept:,}",
                    dupes=exact_dupes,
                    rate=f"{kept / max(total, 1):.1%}",
                )

        # Flush remaining
        if batch:
            _flush_batch(
                batch, pool, fout, seen_hashes, stats,
                counters := {"kept": 0, "total": 0, "dropped": 0, "dupes": 0},
            )
            kept += counters["kept"]
            total += counters["total"]
            dropped_normalize += counters["dropped"]
            exact_dupes += counters["dupes"]

    print(
        f"[pipeline-fast] {input_path.name}: {kept:,}/{total:,} kept, "
        f"{exact_dupes:,} exact dupes, {dropped_normalize:,} dropped by normalize"
    )
    return kept


def _flush_batch(
    batch: list[bytes],
    pool: mp.Pool,
    fout,
    seen_hashes: set[str],
    stats: StatsCollector,
    counters: dict,
):
    """Normalize a batch in parallel, then SHA-256 dedup serially."""
    # Split batch across workers
    n_workers = pool._processes
    sub_batches = [[] for _ in range(n_workers)]
    for i, line in enumerate(batch):
        sub_batches[i % n_workers].append(line)

    # Filter empty sub-batches
    sub_batches = [sb for sb in sub_batches if sb]

    # Parallel normalize
    results = pool.map(_normalize_batch, sub_batches)

    # Serial SHA-256 dedup + write
    for sub_result, sub_input in zip(results, sub_batches):
        for normalized_line, original_line in zip(sub_result, sub_input):
            counters["total"] += 1

            # Parse original for stats
            try:
                orig_doc = orjson.loads(original_line)
                source = orig_doc.get("source", "unknown")
                input_chars = orig_doc.get("char_count", len(orig_doc.get("text", "")))
            except orjson.JSONDecodeError:
                continue

            stats.record_input(source, input_chars)

            if normalized_line is None:
                counters["dropped"] += 1
                stats.record_rejected(source, "normalize_dropped")
                continue

            doc = orjson.loads(normalized_line)
            text_hash = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()

            if text_hash in seen_hashes:
                counters["dupes"] += 1
                stats.record_rejected(source, "exact_duplicate")
                continue

            seen_hashes.add(text_hash)
            stats.record_kept(source, doc["char_count"])
            fout.write(normalized_line + b"\n")
            counters["kept"] += 1


def process_all(
    num_workers: int = DEFAULT_WORKERS,
    target_file: str | None = None,
    output_dir: str | None = None,
):
    """Process raw JSONL files with streaming normalize + SHA-256 dedup."""
    ensure_dirs()
    stats = StatsCollector()
    out_dir = Path(output_dir) if output_dir else PROCESSED_DIR

    # Find input files
    if target_file:
        raw_files = [Path(target_file)]
    else:
        raw_files = []
        for subdir in sorted(RAW_DIR.iterdir()):
            if not subdir.is_dir():
                continue
            for jsonl_file in sorted(subdir.glob("*.jsonl")):
                raw_files.append(jsonl_file)

    if not raw_files:
        print("[pipeline-fast] No raw JSONL files found.")
        return stats

    print(f"[pipeline-fast] {len(raw_files)} files, {num_workers} workers")
    print(f"[pipeline-fast] Output dir: {out_dir}")
    print("[pipeline-fast] Mode: normalize + SHA-256 exact dedup (no fasttext, no MinHash)")
    total_kept = 0

    for raw_file in raw_files:
        output_path = out_dir / raw_file.name
        kept = process_file(raw_file, output_path, num_workers, stats)
        total_kept += kept

    print(f"\n[pipeline-fast] Total: {total_kept:,} documents passed all filters")
    stats.print_summary()

    # Save report to output dir or default
    report_dir = out_dir / "reports" if output_dir else REPORTS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report = stats.generate_report()
    report_path = report_dir / "quality_report_fast.json"
    report_path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    print(f"[stats] Report saved to {report_path}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast pipeline: normalize + SHA-256 dedup")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel workers")
    parser.add_argument("--input", type=str, default=None, help="Specific input JSONL file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: data/processed/)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Docs per batch")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    process_all(
        num_workers=args.workers,
        target_file=args.input,
        output_dir=args.output_dir,
    )
