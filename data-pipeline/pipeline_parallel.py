"""Parallel processing pipeline using file-level chunking.

Splits each raw JSONL file into N chunks on disk, processes each chunk in a
separate process (normalize -> lang_detect -> quality), then merges results
and runs dedup serially. Zero per-doc IPC overhead.

Usage:
    python pipeline_parallel.py          # 4 workers (default)
    python pipeline_parallel.py 6        # 6 workers
"""

import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path

import orjson
from tqdm import tqdm

from config import PROCESSED_DIR, RAW_DIR, ensure_dirs
from processing.dedup import Deduplicator
from processing.stats import StatsCollector

DEFAULT_WORKERS = 4


def _process_chunk(args: tuple[Path, Path]) -> dict:
    """Process one chunk file through stages 1-3. Runs in a worker process.

    Each worker loads its own fasttext model and bnorm instance.
    Reads from chunk_path, writes passing docs to out_path.

    Returns:
        Stats dict with counts and rejection reasons.
    """
    chunk_path, out_path = args

    # Import inside worker so each process gets its own model instances
    from processing.lang_detect import filter_document as lang_filter
    from processing.normalize import normalize_document
    from processing.quality import filter_document as quality_filter

    local_stats = {
        "total": 0,
        "kept": 0,
        "input_chars": 0,
        "kept_chars": 0,
        "rejections": {},  # reason -> count
        "sources": {},  # source -> {total, kept, total_chars, kept_chars}
    }

    with open(chunk_path, "rb") as fin, open(out_path, "wb") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                doc = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue

            source = doc.get("source", "unknown")
            input_chars = doc.get("char_count", len(doc.get("text", "")))
            local_stats["total"] += 1
            local_stats["input_chars"] += input_chars

            if source not in local_stats["sources"]:
                local_stats["sources"][source] = {
                    "total": 0, "kept": 0, "total_chars": 0, "kept_chars": 0,
                }
            local_stats["sources"][source]["total"] += 1
            local_stats["sources"][source]["total_chars"] += input_chars

            # Stage 1: Normalize
            doc = normalize_document(doc)

            # Stage 2: Language detection + contamination check
            doc, lang_ok, lang_reason = lang_filter(doc)
            if not lang_ok:
                local_stats["rejections"][lang_reason] = (
                    local_stats["rejections"].get(lang_reason, 0) + 1
                )
                continue

            # Stage 3: Quality filter
            doc, quality_ok, quality_reason = quality_filter(doc)
            if not quality_ok:
                local_stats["rejections"][quality_reason] = (
                    local_stats["rejections"].get(quality_reason, 0) + 1
                )
                continue

            # Passed stages 1-3, write to temp output
            local_stats["kept"] += 1
            local_stats["kept_chars"] += doc["char_count"]
            local_stats["sources"][source]["kept"] += 1
            local_stats["sources"][source]["kept_chars"] += doc["char_count"]
            fout.write(orjson.dumps(doc) + b"\n")

    return local_stats


def _split_file(filepath: Path, num_chunks: int, tmp_dir: str) -> list[Path]:
    """Split a JSONL file into N roughly equal chunks."""
    # Count lines
    line_count = 0
    with open(filepath, "rb") as f:
        for line in f:
            if line.strip():
                line_count += 1

    if line_count == 0:
        return []

    lines_per_chunk = max(1, line_count // num_chunks)
    chunks = []
    current_chunk = None
    current_count = 0
    chunk_idx = 0

    with open(filepath, "rb") as f:
        for line in f:
            if not line.strip():
                continue

            if current_chunk is None or (
                current_count >= lines_per_chunk and chunk_idx < num_chunks
            ):
                if current_chunk is not None:
                    current_chunk.close()
                chunk_path = Path(tmp_dir) / f"chunk_{chunk_idx}.jsonl"
                chunks.append(chunk_path)
                current_chunk = open(chunk_path, "wb")
                current_count = 0
                chunk_idx += 1

            current_chunk.write(line)
            current_count += 1

    if current_chunk is not None:
        current_chunk.close()

    return chunks


def process_all(num_workers: int = DEFAULT_WORKERS):
    """Process all raw JSONL files with chunk-parallel stages 1-3, serial dedup."""
    ensure_dirs()
    stats = StatsCollector()
    deduplicator = Deduplicator("global")
    deduplicator.load_state()

    raw_files = []
    for subdir in sorted(RAW_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        for jsonl_file in sorted(subdir.glob("*.jsonl")):
            raw_files.append(jsonl_file)

    if not raw_files:
        print("[pipeline] No raw JSONL files found.")
        return stats

    print(f"[pipeline-parallel] {len(raw_files)} files, {num_workers} workers")
    total_kept = 0

    for raw_file in raw_files:
        print(f"\n--- {raw_file.name} ---")

        with tempfile.TemporaryDirectory(
            prefix="bangla_pipeline_", dir="/tmp"
        ) as tmp_dir:
            # Split input file into chunks
            print(f"[split] Splitting {raw_file.name} into {num_workers} chunks...")
            chunk_paths = _split_file(raw_file, num_workers, tmp_dir)

            if not chunk_paths:
                print(f"[pipeline] {raw_file.name}: empty, skipping")
                continue

            # Create output paths for each chunk
            out_paths = [
                Path(tmp_dir) / f"out_{i}.jsonl" for i in range(len(chunk_paths))
            ]
            worker_args = list(zip(chunk_paths, out_paths))

            # Process chunks in parallel
            print(f"[process] Processing {len(chunk_paths)} chunks in parallel...")
            with mp.Pool(num_workers) as pool:
                chunk_stats = pool.map(_process_chunk, worker_args)

            # Aggregate stats from workers
            total_in_file = sum(s["total"] for s in chunk_stats)
            kept_pre_dedup = sum(s["kept"] for s in chunk_stats)
            print(
                f"[stages 1-3] {kept_pre_dedup}/{total_in_file} passed "
                f"({kept_pre_dedup / max(total_in_file, 1):.1%})"
            )

            # Record per-source stats
            for cs in chunk_stats:
                for source, src_stats in cs["sources"].items():
                    for _ in range(src_stats["total"]):
                        stats.record_input(source, src_stats["total_chars"] // max(src_stats["total"], 1))
                for reason, count in cs["rejections"].items():
                    # Find which source this belongs to - use first source as approximation
                    source = next(iter(cs["sources"]), "unknown")
                    for _ in range(count):
                        stats.record_rejected(source, reason)

            # Merge + dedup (serial)
            print(f"[dedup] Merging and deduplicating...")
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            final_out = PROCESSED_DIR / raw_file.name
            kept = 0

            with open(final_out, "wb") as fout:
                for out_path in out_paths:
                    if not out_path.exists():
                        continue
                    with open(out_path, "rb") as fin:
                        for line in tqdm(
                            fin,
                            desc=f"Dedup {out_path.name}",
                            unit=" docs",
                        ):
                            line = line.strip()
                            if not line:
                                continue

                            doc = orjson.loads(line)
                            source = doc.get("source", "unknown")

                            is_dup, dup_reason = deduplicator.is_duplicate(
                                doc["text"], doc_id=doc.get("id", "")
                            )
                            if is_dup:
                                stats.record_rejected(source, dup_reason)
                                continue

                            stats.record_kept(source, doc["char_count"])
                            fout.write(line + b"\n")
                            kept += 1

            print(f"[pipeline] {raw_file.name}: {kept}/{total_in_file} docs kept")
            total_kept += kept

    deduplicator.save_state()
    print(f"\n[pipeline] Total: {total_kept} documents passed all filters")
    stats.print_summary()
    stats.save_report()
    return stats


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKERS
    process_all(num_workers=workers)
