"""Master CLI for the Bangla data collection pipeline.

Usage:
    python collect.py                     # Run full pipeline (collect + process + export)
    python collect.py --collect-only      # Only run collectors
    python collect.py --process-only      # Only run processing pipeline
    python collect.py --export-only       # Only export to compressed chunks
    python collect.py --sources wiki news # Only specific sources
    python collect.py --report            # Only generate stats report
    python collect.py --resume            # Resume interrupted collection
"""

import argparse
import sys
import time

from config import ensure_dirs

# Source name -> collector class mapping
SOURCE_MAP = {
    "wiki": ("collectors.wikipedia", "WikipediaCollector"),
    "newspaper": ("collectors.newspaper", "NewspaperCollector"),
    "literature": ("collectors.literature", "LiteratureCollector"),
    "web_crawl": ("collectors.web_crawl", "WebCrawlCollector"),
    "banglish": ("collectors.banglish", "BanglishCollector"),
    "hf_corpus": ("collectors.hf_corpus", "HFCorpusCollector"),
}

ALL_SOURCES = list(SOURCE_MAP.keys())


def get_collector(name: str):
    """Dynamically import and return a collector instance."""
    module_path, class_name = SOURCE_MAP[name]
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls()


def run_collection(sources: list[str]) -> None:
    """Run data collection for specified sources."""
    print(f"\n{'=' * 60}")
    print(f"COLLECTION PHASE - Sources: {', '.join(sources)}")
    print(f"{'=' * 60}\n")

    for source in sources:
        if source not in SOURCE_MAP:
            print(f"[collect] Unknown source: {source}. Skipping.")
            continue

        print(f"\n--- Collecting: {source} ---")
        start = time.time()
        try:
            collector = get_collector(source)
            collector.run()
        except Exception as e:
            print(f"[collect] ERROR in {source}: {e}")
            continue
        elapsed = time.time() - start
        print(f"--- {source} done in {elapsed:.1f}s ---\n")


def run_processing(sources: list[str] | None = None) -> None:
    """Run the processing pipeline."""
    from pipeline import process_all

    print(f"\n{'=' * 60}")
    print("PROCESSING PHASE")
    print(f"{'=' * 60}\n")

    process_all(sources=sources)


def run_export() -> None:
    """Run the export phase."""
    from export import export_chunks

    print(f"\n{'=' * 60}")
    print("EXPORT PHASE")
    print(f"{'=' * 60}\n")

    export_chunks()


def run_report() -> None:
    """Generate a stats report from existing processed data."""
    from config import PROCESSED_DIR
    from processing.stats import StatsCollector

    import orjson

    stats = StatsCollector()
    total = 0
    for jsonl_file in sorted(PROCESSED_DIR.glob("*.jsonl")):
        with open(jsonl_file, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                source = doc.get("source", "unknown")
                char_count = doc.get("char_count", len(doc.get("text", "")))
                stats.record_input(source, char_count)
                stats.record_kept(source, char_count)
                total += 1

    if total == 0:
        print("[report] No processed documents found.")
        return

    stats.print_summary()
    stats.save_report()


def main():
    parser = argparse.ArgumentParser(
        description="Bangla LLM Data Collection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=ALL_SOURCES,
        default=ALL_SOURCES,
        help="Sources to collect (default: all)",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only run collection, skip processing and export",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Only run processing pipeline on existing raw data",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export processed data to compressed chunks",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate statistics report from processed data",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted collection (uses saved state)",
    )

    args = parser.parse_args()
    ensure_dirs()

    start = time.time()

    if args.report:
        run_report()
    elif args.collect_only:
        run_collection(args.sources)
    elif args.process_only:
        run_processing(args.sources if args.sources != ALL_SOURCES else None)
    elif args.export_only:
        run_export()
    else:
        # Full pipeline
        run_collection(args.sources)
        run_processing()
        run_export()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
