#!/usr/bin/env python3
"""Release-split policy for the Kotha corpus.

Every collected source is classified into one of three redistribution buckets,
derived from its declared `license` in corpus/sources.py (single source of truth):

  releasable  - open / public-domain text we may redistribute (with the license's
                attribution / share-alike obligations propagated).
  conditional - redistributable only after a per-source gate is cleared
                (per-item PD verification, or re-derivation from the underlying
                PD works rather than a non-commercial-packaged copy).
  train_only  - usable to train the model but NOT redistributed: copyrighted
                (news), scraped/synthetic literature, restrictive ToU (CulturaX),
                or unverified license ("see-card").

This module is deterministic and importable. As a CLI it joins the policy with a
process_report_full.json (see corpus/process_corpus.py) and emits per-bucket
doc/token totals plus a per-source manifest — the exact input for the dataset
card's composition table and the pod pass that writes the physical split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sources import SOURCES  # noqa: E402

BUCKETS = ("releasable", "conditional", "train_only")

# Per-source rationale overrides where the license string alone is too blunt.
# Keyed by source name; value is (bucket, human reason). Anything not listed here
# is classified by license_bucket() below and given a generated reason.
OVERRIDES: dict[str, tuple[str, str]] = {
    # NC packaging, but the underlying works are public domain: releasable only by
    # re-deriving text from the PD works, not by mirroring the NC-packaged copy.
    "bongboi": ("conditional",
                "package is CC-BY-NC; underlying works are PD — release requires "
                "re-derivation from the PD sources, not the NC package"),
    # archive.org proof set; a full run must pass per-item PD verification + an
    # OCR-garble gate before any of it is redistributable.
    "archive_bengali": ("conditional",
                         "PD only for a verified per-item subset; needs pd_verify "
                         "+ OCR-garble gate before release"),
    # Organic Banglish: the register is the research contribution, but the source
    # card does not state a redistribution license -> train-only until verified.
    "banglatlit_pt": ("train_only",
                      "organic Banglish, but source license unstated (see-card) — "
                      "verify before any release"),
}


def license_bucket(license_str: str) -> str:
    """Map a declared license string to a redistribution bucket (conservative)."""
    l = license_str.lower()
    if "train-only" in l:
        return "train_only"
    if "tou" in l or "see-card" in l:            # restrictive or unverified
        return "train_only"
    if "verify per-item" in l or "pd-subset" in l:
        return "conditional"
    if "cc-by-nc" in l:                          # non-commercial packaging
        return "conditional"
    if (l.startswith("pd") or "public domain" in l
            or "cc0" in l or "cc-by-sa" in l or "cc-by-4" in l
            or "cc-by " in l or l == "cc-by" or "odc-by" in l or l == "mit"):
        return "releasable"
    return "train_only"                          # default: do not redistribute


def classify(source: str) -> dict:
    """Return {source, bucket, license, reason} for a known source."""
    if source in OVERRIDES:
        bucket, reason = OVERRIDES[source]
        lic = SOURCES.get(source, {}).get("license", "?")
        return {"source": source, "bucket": bucket, "license": lic, "reason": reason}
    meta = SOURCES.get(source)
    if meta is None:
        return {"source": source, "bucket": "train_only", "license": "unknown",
                "reason": "source not in registry — conservative train-only"}
    lic = meta.get("license", "unknown")
    bucket = license_bucket(lic)
    return {"source": source, "bucket": bucket, "license": lic,
            "reason": f"license '{lic}' -> {bucket}"}


def build_manifest(report: dict) -> dict:
    """Join a process report's per_source counts with the release policy."""
    per_source = report.get("per_source", {})
    rows = []
    totals = {b: {"sources": 0, "read": 0, "kept": 0, "dup": 0,
                  "sadhu": 0, "tok": 0} for b in BUCKETS}
    for source, c in sorted(per_source.items()):
        info = classify(source)
        b = info["bucket"]
        read, kept = c.get("read", 0), c.get("kept", 0)
        dup, sadhu, tok = c.get("dup", 0), c.get("sadhu", 0), c.get("tok", 0)
        rows.append({**info, "read": read, "kept": kept, "dup": dup,
                     "sadhu": sadhu, "tok": tok})
        t = totals[b]
        t["sources"] += 1
        t["read"] += read
        t["kept"] += kept
        t["dup"] += dup
        t["sadhu"] += sadhu
        t["tok"] += tok
    grand_kept = sum(t["kept"] for t in totals.values()) or 1
    grand_tok = sum(t["tok"] for t in totals.values()) or 1
    for b in BUCKETS:
        totals[b]["pct_docs"] = round(100 * totals[b]["kept"] / grand_kept, 2)
        totals[b]["pct_tokens"] = round(100 * totals[b]["tok"] / grand_tok, 2)
        totals[b]["tok_B"] = round(totals[b]["tok"] / 1e9, 3)
    return {"buckets": totals, "sources": rows,
            "unknown_sources": [r["source"] for r in rows
                                if r["reason"].startswith("source not in registry")]}


def _fmt_table(manifest: dict) -> str:
    lines = []
    lines.append(f"{'source':24s} {'bucket':11s} {'license':22s} "
                 f"{'kept':>11s} {'~Btok':>7s} {'sadhu':>8s}")
    lines.append("-" * 90)
    order = {"releasable": 0, "conditional": 1, "train_only": 2}
    for r in sorted(manifest["sources"],
                    key=lambda r: (order[r["bucket"]], -r["kept"])):
        lines.append(f"{r['source']:24s} {r['bucket']:11s} {r['license'][:22]:22s} "
                     f"{r['kept']:>11,} {r['tok']/1e9:>7.2f} {r['sadhu']:>8,}")
    lines.append("-" * 90)
    for b in BUCKETS:
        t = manifest["buckets"][b]
        lines.append(f"{b:24s} {'':11s} {'':22s} {t['kept']:>11,} "
                     f"{t['tok_B']:>7.2f} {t['sadhu']:>8,}  "
                     f"({t['pct_docs']}% docs, {t['pct_tokens']}% tok)")
    if manifest["unknown_sources"]:
        lines.append(f"\n[WARN] sources not in registry (treated train_only): "
                     f"{manifest['unknown_sources']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Kotha release-split manifest")
    ap.add_argument("--report", help="process_report_full.json from process_corpus.py")
    ap.add_argument("--out", help="write manifest JSON here")
    ap.add_argument("--list-policy", action="store_true",
                    help="print the bucket for every registered source and exit")
    args = ap.parse_args()

    if args.list_policy:
        for name in sorted(SOURCES):
            info = classify(name)
            print(f"  {name:24s} -> {info['bucket']:11s}  ({info['license']})")
        return 0

    if not args.report:
        ap.error("--report is required (or use --list-policy)")
    report = json.loads(Path(args.report).read_text())
    manifest = build_manifest(report)
    print(_fmt_table(manifest))
    if args.out:
        Path(args.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"\nmanifest -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
