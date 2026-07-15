#!/usr/bin/env python3
"""Harvest the full Bengali Wikisource dump → plain-text JSONL for the Kotha collector.

Bengali Wikisource is the richest CLEAN সাধু source: human-proofread Unicode (no OCR)
of the 19th/early-20th-c canon — Bankim, Vidyasagar (বর্ণপরিচয়/বেতাল), Michael Madhusudan
(মেঘনাদবধ), early Tagore — in genuine সাধু (করিয়া/হইয়াছে/তাহার). Text CC-BY-SA, works PD.

We take BOTH namespaces:
  ns=0   (mainspace)  — works with inline text; transclusion-only pages strip to ~nothing
                        and are dropped by --min-chars.
  ns=104 (Page:)      — the ProofreadPage per-scan text; this is where the BULK of the
                        proofread classical text actually lives.
Strips wikitext (templates/markup/refs) with mwparserfromhell.strip_code, writes
{id, title, ns, text}. Feed the output through the normal collector (a `wikisource_dump`
source with a `local_jsonl` path) so it gets purity routing + সাধু tagging + Drive upload.

Usage:
  python3 corpus/scrape_wikisource.py \
    --dump corpus/data/wikisource/bnwikisource.xml.bz2 \
    --out  corpus/data/wikisource/wikisource_pages.jsonl
"""
from __future__ import annotations

import argparse
import bz2
import json
import re
import sys

import mwparserfromhell
import mwxml

_WS = re.compile(r"[ \t]*\n[ \t]*\n[ \t]*(\n[ \t]*)*")   # collapse blank-line runs


def strip_wikitext(raw: str) -> str:
    try:
        plain = mwparserfromhell.parse(raw).strip_code(normalize=True, collapse=True)
    except Exception:
        # last-ditch: drop the most common markup so a single bad page can't abort the run
        plain = re.sub(r"\{\{[^{}]*\}\}|\[\[|\]\]|'''?|<[^>]+>", "", raw)
    plain = _WS.sub("\n\n", plain).strip()
    return plain


def main() -> int:
    ap = argparse.ArgumentParser(description="Bengali Wikisource dump -> plain-text JSONL")
    ap.add_argument("--dump", required=True, help="bnwikisource-*-pages-articles.xml.bz2")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--namespaces", type=int, nargs="+", default=[0, 104],
                    help="0=mainspace works, 104=Page: proofread text (default both)")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--progress-every", type=int, default=2000)
    args = ap.parse_args()

    ns_ok = set(args.namespaces)
    n_pages = n_kept = n_short = n_redir = 0
    with bz2.open(args.dump, "rt", encoding="utf-8") as fh, \
            open(args.out, "w", encoding="utf-8") as out:
        for page in mwxml.Dump.from_file(fh):
            if page.namespace not in ns_ok:
                continue
            if getattr(page, "redirect", None):
                n_redir += 1
                continue
            n_pages += 1
            text = ""
            for rev in page:                 # pages-articles holds the current revision
                text = rev.text or ""
            plain = strip_wikitext(text)
            if len(plain) < args.min_chars:
                n_short += 1
                continue
            out.write(json.dumps(
                {"id": page.id, "title": page.title, "ns": page.namespace, "text": plain},
                ensure_ascii=False) + "\n")
            n_kept += 1
            if n_kept % args.progress_every == 0:
                print(f"  kept={n_kept:,}  seen={n_pages:,}  short={n_short:,}", flush=True)
    print(f"done: kept={n_kept:,} / seen={n_pages:,} pages "
          f"(short={n_short:,}, redirects={n_redir:,}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
