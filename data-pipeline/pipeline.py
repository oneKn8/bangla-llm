"""Processing pipeline: normalize -> lang_detect -> quality -> dedup.

Reads raw JSONL files, applies all processing stages, writes to processed/.
"""

from pathlib import Path

import orjson
from tqdm import tqdm

from config import PROCESSED_DIR, RAW_DIR, ensure_dirs
from processing.dedup import Deduplicator
from processing.lang_detect import filter_document as lang_filter
from processing.normalize import normalize_document
from processing.quality import filter_document as quality_filter
from processing.stats import StatsCollector


def process_source(
    source_file: Path,
    stats: StatsCollector,
    deduplicator: Deduplicator,
    output_dir: Path | None = None,
) -> int:
    """Process a single raw JSONL file through the full pipeline.

    Args:
        source_file: Path to raw JSONL file
        stats: Stats collector instance
        deduplicator: Deduplicator instance (shared for cross-source dedup)
        output_dir: Override output directory

    Returns:
        Number of documents kept
    """
    out_dir = output_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / source_file.name

    kept = 0
    total = 0

    with open(source_file, "rb") as fin, open(out_path, "wb") as fout:
        for line in tqdm(fin, desc=f"Processing {source_file.name}", unit=" docs"):
            line = line.strip()
            if not line:
                continue

            try:
                doc = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue

            total += 1
            source = doc.get("source", "unknown")
            stats.record_input(source, doc.get("char_count", len(doc.get("text", ""))))

            # Stage 1: Normalize
            doc = normalize_document(doc)

            # Stage 2: Language detection + contamination check
            doc, lang_ok, lang_reason = lang_filter(doc)
            if not lang_ok:
                stats.record_rejected(source, lang_reason)
                continue

            # Stage 3: Quality filter
            doc, quality_ok, quality_reason = quality_filter(doc)
            if not quality_ok:
                stats.record_rejected(source, quality_reason)
                continue

            # Stage 4: Deduplication
            is_dup, dup_reason = deduplicator.is_duplicate(
                doc["text"], doc_id=doc.get("id", "")
            )
            if is_dup:
                stats.record_rejected(source, dup_reason)
                continue

            # Passed all filters
            stats.record_kept(source, doc["char_count"])
            fout.write(orjson.dumps(doc) + b"\n")
            kept += 1

    print(f"[pipeline] {source_file.name}: {kept}/{total} docs kept")
    return kept


def process_all(sources: list[str] | None = None) -> StatsCollector:
    """Process all raw JSONL files through the pipeline.

    Args:
        sources: List of source names to process (matches filenames).
                 None = process all files in raw/.

    Returns:
        StatsCollector with full statistics
    """
    ensure_dirs()
    stats = StatsCollector()
    deduplicator = Deduplicator("global")
    deduplicator.load_state()

    # Find all raw JSONL files
    raw_files = []
    for subdir in sorted(RAW_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        for jsonl_file in sorted(subdir.glob("*.jsonl")):
            if sources is None or any(s in jsonl_file.stem for s in sources):
                raw_files.append(jsonl_file)

    if not raw_files:
        print("[pipeline] No raw JSONL files found.")
        return stats

    print(f"[pipeline] Processing {len(raw_files)} files...")
    total_kept = 0

    for raw_file in raw_files:
        kept = process_source(raw_file, stats, deduplicator)
        total_kept += kept

    # Save dedup state for resume
    deduplicator.save_state()

    print(f"\n[pipeline] Total: {total_kept} documents passed all filters")
    stats.print_summary()
    stats.save_report()

    return stats


if __name__ == "__main__":
    process_all()
