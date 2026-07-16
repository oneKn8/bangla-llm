#!/usr/bin/env python3
"""Kotha corpus processor — ONE streaming pass over the collected shards:
exact cross-source dedup + সাধু-tagging + token recount + composition report.

We must read every shard once anyway, so we fuse the three jobs:
  1. NFC-normalize each doc's text (never morphological-fold before this).
  2. Exact cross-source dedup via xxhash64 of the normalized text, with a
     SOURCE-PRIORITY tie-break (the higher-priority source keeps the doc; later
     duplicates from lower-priority sources are dropped).
  3. সাধু tag via corpus/sadhu.py (per-doc {sadhu_per100, sadhu_cats, is_sadhu}).
  4. Token recount (char/4.45 estimate by default; --tokenizer for exact).

Reads shards from a local dir (after `rclone copy`) OR streams them straight from
an rclone remote (`--rclone-remote gdrive:kotha-corpus`) — the latter needs no
local disk. Writes an enriched+deduped JSONL(.gz) per track if --out given, always
prints a composition report.

MinHash/LSH NEAR-dedup is intentionally NOT here (RAM-heavy) — run it as a second
datatrove stage on the pod over this exact-deduped output. This pass is the cheap,
must-do first stage that already yields the real unique-token count + the সাধু slice.

Run on a high-RAM CPU pod (fast Drive I/O). Memory ≈ 8 bytes * unique_docs for the
hash set (~40M docs ≈ a few GB).

Examples:
  # stream from Drive, report only
  python3 corpus/process_corpus.py --rclone-remote gdrive:kotha-corpus --tracks native codeswitch banglish
  # local dir (after rclone copy), write enriched deduped shards
  python3 corpus/process_corpus.py --shards-dir corpus/data --out corpus/processed --sadhu-only-report
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xxhash                                   # noqa: E402
from sadhu import classify as sadhu_classify    # noqa: E402

# Higher priority = kept on a dedup collision. Curated/literary/unique first,
# Common-Crawl-derived last (they overlap the most). Unknown sources -> 0.
SOURCE_PRIORITY = {
    "sangraha_bn": 100, "wikisource_bn": 95, "wikipedia_bn": 90,
    "bangla_textbook": 85, "tagore_gitanjali": 84, "banglatlit_pt": 80,
    "banglish_kawsar_v3": 70, "banglish_kawsar_v1": 69, "banglish_kawsar_80k": 68,
    "fineweb2_bn": 50, "hplt_bn": 45, "culturax_bn": 40, "indiccorp2_bn": 35,
}


def shard_source(name: str) -> str:
    # shard filename: "<source>-<track>-<idx>.jsonl.gz"
    base = name.split("/")[-1]
    return base.split("-")[0] if "-" in base else base


def iter_shards(remote: str | None, shards_dir: str | None, tracks: list[str]):
    """Yield (source, track, shard_name, open_text_fh) for every shard, ordered by
    source priority so the highest-priority copy of a duplicate is seen first."""
    entries = []  # (priority, source, track, locator)
    for track in tracks:
        if remote:
            r = subprocess.run(["rclone", "lsf", f"{remote}/{track}/"],
                               capture_output=True, text=True, timeout=300)
            names = [ln.strip() for ln in r.stdout.splitlines()
                     if ln.strip().endswith(".jsonl.gz")]
            for nm in names:
                src = shard_source(nm)
                entries.append((-SOURCE_PRIORITY.get(src, 0), src, track,
                                ("remote", f"{remote}/{track}/{nm}", nm)))
        elif shards_dir:
            d = Path(shards_dir) / track
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.jsonl.gz")):
                src = shard_source(p.name)
                entries.append((-SOURCE_PRIORITY.get(src, 0), src, track,
                                ("local", str(p), p.name)))
    entries.sort(key=lambda e: (e[0], e[1], e[3][2]))   # priority, source, name
    for _pri, src, track, (kind, locator, nm) in entries:
        if kind == "local":
            with gzip.open(locator, "rt", encoding="utf-8") as fh:
                yield src, track, nm, fh
        else:
            proc = subprocess.Popen(["rclone", "cat", locator], stdout=subprocess.PIPE)
            with gzip.GzipFile(fileobj=proc.stdout) as gz:
                yield src, track, nm, (line.decode("utf-8") for line in gz)
            proc.wait()


def _safe_lines(it, shard):
    """Yield lines, tolerating a truncated/corrupt/mid-write gzip shard (use partial, skip rest)."""
    try:
        for line in it:
            yield line
    except (EOFError, OSError) as e:
        print(f"  [WARN] truncated/corrupt shard {shard}: {e} — used partial", file=sys.stderr)


class BloomDedup:
    """Fixed-memory approximate dedup: m bits + k hashes; RAM = m/8 bytes, CONSTANT
    regardless of corpus size (OOM-proof). Default 1<<31 bits = 256 MB, ~0.1% false-
    positive at 100M docs (a FP drops a unique doc as 'dup' — negligible for stats;
    the decisive dedup is the pod MinHash stage anyway)."""
    __slots__ = ("m", "k", "buf")

    def __init__(self, nbits: int = 1 << 31, k: int = 10):
        self.m, self.k, self.buf = nbits, k, bytearray(nbits >> 3)

    def check_add(self, data: bytes) -> bool:
        h1 = xxhash.xxh64(data, seed=0).intdigest()
        h2 = xxhash.xxh64(data, seed=0x9E3779B1).intdigest() | 1
        buf, m, seen = self.buf, self.m, True
        for i in range(self.k):
            b = (h1 + i * h2) % m
            byte, mask = b >> 3, 1 << (b & 7)
            if not (buf[byte] & mask):
                seen = False
                buf[byte] |= mask
        return seen


class ExactDedup:
    """Exact dedup via a hash set — RAM GROWS with unique docs (~50-90 B/doc). Use
    only where RAM is ample (pod), never unbounded on a small laptop."""
    __slots__ = ("seen",)

    def __init__(self):
        self.seen: set[int] = set()

    def check_add(self, data: bytes) -> bool:
        h = xxhash.xxh64(data).intdigest()
        if h in self.seen:
            return True
        self.seen.add(h)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Kotha corpus processor (dedup + sadhu + recount)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rclone-remote", help="e.g. gdrive:kotha-corpus (stream, no local disk)")
    src.add_argument("--shards-dir", help="local dir containing <track>/ subdirs of .jsonl.gz")
    ap.add_argument("--tracks", nargs="+", default=["native", "codeswitch", "banglish"])
    ap.add_argument("--out", help="write enriched+deduped shards here (per track)")
    ap.add_argument("--chars-per-tok", type=float, default=4.45, help="token estimate divisor (fallback when --tokenizer absent)")
    ap.add_argument("--tokenizer", default=None,
                    help="path to a SentencePiece .model for EXACT token counts "
                         "(default: char/--chars-per-tok estimate)")
    ap.add_argument("--dedup", choices=["bloom", "exact", "none"], default="bloom",
                    help="bloom=fixed 256MB, OOM-proof (~0.1%% FP); exact=set, RAM grows; none")
    ap.add_argument("--report", default=None, help="write JSON report here")
    ap.add_argument("--progress-every", type=int, default=200_000)
    args = ap.parse_args()

    sp = None
    if args.tokenizer:
        import sentencepiece as _spm
        sp = _spm.SentencePieceProcessor(model_file=args.tokenizer)
        print(f"exact token count via {args.tokenizer} (vocab {sp.vocab_size()})", flush=True)

    dedup = BloomDedup() if args.dedup == "bloom" else (ExactDedup() if args.dedup == "exact" else None)
    t0 = time.time()
    n_total = n_unique = 0
    tok_unique = 0
    sadhu_docs = 0
    sadhu_tok = 0
    per_src = defaultdict(lambda: Counter())   # source -> {read,kept,dup,sadhu,tok}
    out_fh = {}
    outdir = Path(args.out) if args.out else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    def get_out(track):
        if not outdir:
            return None
        if track not in out_fh:
            out_fh[track] = gzip.open(outdir / f"{track}.dedup.jsonl.gz", "wt", encoding="utf-8")
        return out_fh[track]

    for source, track, shard, fh in iter_shards(args.rclone_remote, args.shards_dir, args.tracks):
        ps = per_src[source]
        for line in _safe_lines(fh, shard):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = d.get("text", "") or ""
            n_total += 1
            ps["read"] += 1
            nfc = unicodedata.normalize("NFC", text)
            if dedup is not None and dedup.check_add(nfc.encode("utf-8")):
                ps["dup"] += 1
                continue
            n_unique += 1
            ps["kept"] += 1
            toks = len(sp.encode(nfc)) if sp is not None else int(len(text) / args.chars_per_tok)
            tok_unique += toks
            ps["tok"] += toks
            sc = sadhu_classify(text)
            if sc.is_sadhu:
                sadhu_docs += 1
                sadhu_tok += toks
                ps["sadhu"] += 1
            ofh = get_out(track)
            if ofh is not None:
                d["sadhu_per100"] = sc.per100
                d["sadhu_cats"] = sc.cats
                d["is_sadhu"] = sc.is_sadhu
                ofh.write(json.dumps(d, ensure_ascii=False) + "\n")
            if n_total % args.progress_every == 0:
                el = time.time() - t0
                print(f"  [{el:6.0f}s] read={n_total:,} unique={n_unique:,} "
                      f"dup={100*(n_total-n_unique)/n_total:.1f}% "
                      f"sadhu={sadhu_docs:,} ~{tok_unique/1e9:.2f}B uniq tok", flush=True)

    for f in out_fh.values():
        f.close()

    dt = time.time() - t0
    report = {
        "token_method": args.tokenizer or f"estimate char/{args.chars_per_tok}",
        "docs_read": n_total, "docs_unique": n_unique,
        "dup_rate_pct": round(100 * (n_total - n_unique) / max(1, n_total), 2),
        "unique_tokens_est": tok_unique, "unique_tokens_B": round(tok_unique / 1e9, 3),
        "sadhu_docs": sadhu_docs, "sadhu_tokens_est": sadhu_tok,
        "sadhu_tokens_B": round(sadhu_tok / 1e9, 3),
        "sadhu_pct_of_unique": round(100 * sadhu_docs / max(1, n_unique), 2),
        "seconds": round(dt, 1),
        "per_source": {s: dict(c) for s, c in per_src.items()},
    }
    print("\n" + "=" * 72)
    print(f"docs read={n_total:,}  unique={n_unique:,}  dup={report['dup_rate_pct']}%  "
          f"unique tokens≈{report['unique_tokens_B']}B  in {dt:.0f}s")
    print(f"সাধু: {sadhu_docs:,} docs ({report['sadhu_pct_of_unique']}% of unique) "
          f"≈ {report['sadhu_tokens_B']}B tokens")
    print("per-source (read / kept / dup / sadhu / ~Btok):")
    for s, c in sorted(per_src.items(), key=lambda kv: -kv[1]["kept"]):
        print(f"  {s:22s} read={c['read']:>10,} kept={c['kept']:>10,} "
              f"dup={c['dup']:>9,} sadhu={c['sadhu']:>8,} ~{c['tok']/1e9:.2f}B")
    print("=" * 72)
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
