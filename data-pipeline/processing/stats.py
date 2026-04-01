"""Generate per-source statistics and quality reports."""

import json
import time
from collections import defaultdict
from pathlib import Path

import orjson

from config import REPORTS_DIR


class StatsCollector:
    """Collect and report statistics across the pipeline."""

    def __init__(self):
        self.source_stats: dict[str, dict] = defaultdict(
            lambda: {
                "total_docs": 0,
                "total_chars": 0,
                "kept_docs": 0,
                "kept_chars": 0,
                "rejected": defaultdict(int),  # reason -> count
            }
        )
        self.global_stats = {
            "start_time": time.time(),
            "end_time": 0,
            "total_input": 0,
            "total_output": 0,
        }

    def record_input(self, source: str, char_count: int) -> None:
        """Record a document entering the pipeline."""
        self.source_stats[source]["total_docs"] += 1
        self.source_stats[source]["total_chars"] += char_count
        self.global_stats["total_input"] += 1

    def record_kept(self, source: str, char_count: int) -> None:
        """Record a document that passed all filters."""
        self.source_stats[source]["kept_docs"] += 1
        self.source_stats[source]["kept_chars"] += char_count
        self.global_stats["total_output"] += 1

    def record_rejected(self, source: str, reason: str) -> None:
        """Record a rejected document with reason."""
        self.source_stats[source]["rejected"][reason] += 1

    def generate_report(self) -> dict:
        """Generate a full statistics report."""
        self.global_stats["end_time"] = time.time()
        elapsed = self.global_stats["end_time"] - self.global_stats["start_time"]

        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(elapsed, 1),
            "global": {
                "total_input_docs": self.global_stats["total_input"],
                "total_output_docs": self.global_stats["total_output"],
                "retention_rate": (
                    self.global_stats["total_output"] / max(self.global_stats["total_input"], 1)
                ),
            },
            "per_source": {},
        }

        for source, stats in sorted(self.source_stats.items()):
            retention = stats["kept_docs"] / max(stats["total_docs"], 1)
            report["per_source"][source] = {
                "total_docs": stats["total_docs"],
                "kept_docs": stats["kept_docs"],
                "retention_rate": round(retention, 4),
                "total_chars": stats["total_chars"],
                "kept_chars": stats["kept_chars"],
                "avg_doc_length": (
                    stats["kept_chars"] // max(stats["kept_docs"], 1)
                ),
                "rejection_breakdown": dict(stats["rejected"]),
            }

        return report

    def save_report(self, filename: str = "quality_report.json") -> Path:
        """Save report to the reports directory."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report = self.generate_report()
        path = REPORTS_DIR / filename
        path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
        print(f"[stats] Report saved to {path}")
        return path

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        report = self.generate_report()

        print("\n" + "=" * 60)
        print("PIPELINE STATISTICS REPORT")
        print("=" * 60)
        print(f"Generated: {report['generated_at']}")
        print(f"Elapsed: {report['elapsed_seconds']}s")
        print(f"\nGlobal: {report['global']['total_input_docs']} in -> "
              f"{report['global']['total_output_docs']} out "
              f"({report['global']['retention_rate']:.1%} retention)")

        print(f"\n{'Source':<15} {'Input':>8} {'Output':>8} {'Rate':>8} {'Avg Len':>8}")
        print("-" * 55)
        for source, s in report["per_source"].items():
            print(
                f"{source:<15} {s['total_docs']:>8} {s['kept_docs']:>8} "
                f"{s['retention_rate']:>7.1%} {s['avg_doc_length']:>8}"
            )
            if s["rejection_breakdown"]:
                for reason, count in sorted(
                    s["rejection_breakdown"].items(), key=lambda x: -x[1]
                ):
                    print(f"  - {reason}: {count}")

        print("=" * 60)
