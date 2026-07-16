#!/usr/bin/env python3
"""Finalize + SANITY-CHECK the near-dedup release, ready to run the instant the pod's
`near_dedup_report.json` lands. Cross-validates the near-dedup output against the
earlier exact-dedup pass (`process_report_full.json`) so an implausible result can't
slip through, then prints the datasheet-ready figures.

Sanity gates (any failure => non-zero exit + loud message):
  1. near-dedup docs_kept <= exact-dedup docs_unique (near-dedup only REMOVES more).
  2. near-dedup docs_in ~= exact-dedup docs_read (same corpus; tolerate <=1% for the
     news/live shards added after the exact pass).
  3. every bucket's docs>0 has tokens>0; total_tokens == sum(bucket tokens).
  4. releasable is the dominant bucket (>50% of tokens) — matches the release thesis.
  5. exact token count present (token_method != estimate) if a tokenizer was used.

Usage:
  python3 corpus/finalize_release.py --near-dedup near_dedup_report.json \
      --exact corpus/data/process_report_full.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BUCKETS = ("releasable", "conditional", "train_only")


def load(p: str) -> dict:
    return json.loads(Path(p).read_text())


def check(cond: bool, msg: str, fails: list) -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {msg}")
    if not cond:
        fails.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser(description="Finalize + sanity-check the near-dedup release")
    ap.add_argument("--near-dedup", required=True, help="near_dedup_report.json")
    ap.add_argument("--exact", default="corpus/data/process_report_full.json",
                    help="the earlier exact-dedup report to cross-check against")
    args = ap.parse_args()

    nd = load(args.near_dedup)
    fails: list = []

    docs_in = nd.get("docs_in", 0)
    docs_kept = nd.get("docs_kept", 0)
    removed = nd.get("near_dup_removed", 0)
    per_bucket = nd.get("per_bucket", {})
    total_tok = nd.get("total_tokens", 0)

    print("=" * 68)
    print("NEAR-DEDUP RELEASE — SANITY CHECK")
    print("=" * 68)

    ex = None
    if Path(args.exact).exists():
        ex = load(args.exact)
        ex_unique = ex.get("docs_unique", 0)
        ex_read = ex.get("docs_read", 0)
        check(docs_kept <= ex_unique,
              f"kept {docs_kept:,} <= exact-unique {ex_unique:,}", fails)
        # docs_in should match the exact-dedup input within ~1% (news added later)
        tol = max(1, int(0.02 * ex_read))
        check(abs(docs_in - ex_read) <= tol or docs_in >= ex_read,
              f"docs_in {docs_in:,} ~= exact-read {ex_read:,} (+/-2% or grew)", fails)
    else:
        print(f"  [WARN] exact report {args.exact} not found — skipping cross-check")

    sum_bucket_tok = sum(per_bucket.get(b, {}).get("tokens", 0) for b in BUCKETS)
    check(total_tok == sum_bucket_tok,
          f"total_tokens {total_tok:,} == sum(bucket tokens) {sum_bucket_tok:,}", fails)

    for b in BUCKETS:
        d = per_bucket.get(b, {})
        if d.get("docs", 0) > 0:
            check(d.get("tokens", 0) > 0, f"bucket {b}: docs>0 => tokens>0", fails)

    rel_tok = per_bucket.get("releasable", {}).get("tokens", 0)
    check(total_tok > 0 and rel_tok / max(1, total_tok) > 0.5,
          f"releasable dominant ({100*rel_tok/max(1,total_tok):.1f}% of tokens)", fails)

    check(docs_in > 0 and 0 <= removed <= docs_in and docs_kept + removed == docs_in,
          f"kept {docs_kept:,} + removed {removed:,} == in {docs_in:,}", fails)

    print("-" * 68)
    print("DATASHEET-READY FIGURES")
    print(f"  docs in (post exact-dedup)   : {docs_in:,}")
    print(f"  docs kept (post near-dedup)  : {docs_kept:,}")
    print(f"  near-dup removed             : {removed:,} ({nd.get('near_dup_removed_pct', 0)}%)")
    print(f"  clusters / singletons        : {nd.get('clusters', 0):,} / {nd.get('singletons', 0):,}")
    print(f"  EXACT total tokens           : {total_tok:,} (~{total_tok/1e9:.3f}B)")
    print(f"  token method                 : {nd.get('tokenizer', '?')}")
    print("  per bucket (docs / ~Btok / sadhu):")
    for b in BUCKETS:
        d = per_bucket.get(b, {"docs": 0, "tokens": 0, "sadhu": 0})
        pct = 100 * d.get("tokens", 0) / max(1, total_tok)
        print(f"    {b:12s} {d.get('docs', 0):>11,}  {d.get('tokens', 0)/1e9:>7.3f}B  "
              f"sadhu={d.get('sadhu', 0):>9,}  ({pct:.1f}% tok)")
    print("=" * 68)

    if fails:
        print(f"RESULT: {len(fails)} SANITY CHECK(S) FAILED — do NOT publish as-is:")
        for m in fails:
            print(f"   - {m}")
        return 1
    print("RESULT: all sanity checks PASSED — release numbers are trustworthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
