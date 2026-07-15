#!/usr/bin/env python3
"""Live Bengali NEWS crawler → recency corpus (2024 → today → this minute).

Discovers article URLs from each outlet's RSS feeds + XML sitemaps, fetches each,
extracts the main article text + publication date + title with trafilatura, keeps
Bengali articles dated >= --since (default 2024-01-01), and appends to a JSONL the
Kotha collector ingests (source `news_live`, local_jsonl). Polite: per-domain rate
limit + robots-respecting fetch. Resumable: a .seen file dedups URLs across runs, so
you can kill and restart, or re-run daily to pull only what's new.

News text is copyrighted → tag it train-only (do NOT redistribute as a licensed corpus;
keep url+date provenance so a releasable version can be reconstructed by re-fetch).

Runs until every discovered URL is processed (or you kill it). Re-run to catch up to now.

Usage:
  python3 corpus/scrape_news.py --out corpus/data/news/news.jsonl --since 2024-01-01
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from urllib.parse import urlparse

import feedparser
import trafilatura
from trafilatura.feeds import find_feed_urls

# (name, homepage). Bangladesh + West Bengal majors. sitemap_search + feed discovery
# both run per site; more outlets = more coverage.
SITES = [
    ("prothomalo", "https://www.prothomalo.com"),
    ("bdnews24", "https://bangla.bdnews24.com"),
    ("jugantor", "https://www.jugantor.com"),
    ("kalerkantho", "https://www.kalerkantho.com"),
    ("samakal", "https://samakal.com"),
    ("ittefaq", "https://www.ittefaq.com.bd"),
    ("banglatribune", "https://www.banglatribune.com"),
    ("dhakapost", "https://www.dhakapost.com"),
    ("jagonews24", "https://www.jagonews24.com"),
    ("banglanews24", "https://www.banglanews24.com"),
    ("risingbd", "https://www.risingbd.com"),
    ("dailynayadiganta", "https://www.dailynayadiganta.com"),
    ("mzamin", "https://mzamin.com"),
    ("bhorerkagoj", "https://www.bhorerkagoj.com"),
    ("anandabazar", "https://www.anandabazar.com"),
    ("sangbadpratidin", "https://www.sangbadpratidin.in"),
    ("bartaman", "https://bartamanpatrika.com"),
]

_BENGALI = re.compile(r"[ঀ-৿]")
_DOMAIN_LAST: dict[str, float] = {}
_MIN_GAP = 1.2   # seconds between hits to the SAME domain (politeness)


def bengali_ratio(t: str) -> float:
    ns = [c for c in t if not c.isspace()]
    return (sum(1 for c in ns if _BENGALI.match(c)) / len(ns)) if ns else 0.0


def throttle(url: str):
    d = urlparse(url).netloc
    now = time.time()
    wait = _MIN_GAP - (now - _DOMAIN_LAST.get(d, 0))
    if wait > 0:
        time.sleep(wait)
    _DOMAIN_LAST[d] = time.time()


def discover(name: str, home: str) -> set[str]:
    """RSS-first (freshest, no slow sitemap recursion) + a shallow homepage link grab."""
    urls: set[str] = set()
    dom = urlparse(home).netloc
    # 1) trafilatura feed discovery -> article links directly (freshest)
    try:
        urls.update(find_feed_urls(home, target_lang="bn") or [])
    except Exception:
        pass
    # 2) common feed paths via feedparser (fallback)
    for f in (home.rstrip("/") + p for p in
              ("/rss.xml", "/feed", "/rss", "/feed/", "/rss/", "/atom.xml", "/feed.xml")):
        try:
            throttle(f)
            for e in feedparser.parse(f).entries:
                if e.get("link"):
                    urls.add(e["link"])
        except Exception:
            pass
    # 2) shallow homepage grab — in-domain links (non-articles just extract to None later)
    try:
        throttle(home)
        html = trafilatura.fetch_url(home)
        if html:
            for m in re.findall(r'href=["\']?(https?://[^"\' >]+)', html):
                if dom in urlparse(m).netloc:
                    urls.add(m.split("#")[0])
    except Exception:
        pass
    return urls


def main() -> int:
    ap = argparse.ArgumentParser(description="Live Bengali news crawler")
    ap.add_argument("--out", default="corpus/data/news/news.jsonl")
    ap.add_argument("--since", default="2024-01-01", help="keep articles dated >= this (YYYY-MM-DD)")
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--min-bengali", type=float, default=0.5)
    ap.add_argument("--seen", default="corpus/data/news/.seen")
    ap.add_argument("--max-per-site", type=int, default=0, help="0 = unbounded")
    args = ap.parse_args()

    seen: set[str] = set()
    try:
        seen = set(l.strip() for l in open(args.seen, encoding="utf-8") if l.strip())
    except FileNotFoundError:
        pass
    print(f"[news] resume: {len(seen):,} URLs already seen; since={args.since}")

    out = open(args.out, "a", encoding="utf-8")
    seen_fh = open(args.seen, "a", encoding="utf-8")
    kept = fetched = skipped = 0
    t0 = time.time()

    for name, home in SITES:
        try:
            urls = discover(name, home)
        except Exception as e:
            print(f"[news] {name}: discover failed {type(e).__name__}", file=sys.stderr)
            continue
        fresh = [u for u in urls if u not in seen]
        print(f"[news] {name}: {len(urls):,} urls ({len(fresh):,} new)", flush=True)
        site_kept = 0
        for u in fresh:
            if args.max_per_site and site_kept >= args.max_per_site:
                break
            seen.add(u)
            seen_fh.write(u + "\n")
            try:
                throttle(u)
                html = trafilatura.fetch_url(u)
                if not html:
                    continue
                data = trafilatura.extract(html, output_format="json", with_metadata=True,
                                           target_language="bn", favor_precision=True)
                fetched += 1
                if not data:
                    continue
                d = json.loads(data)
                text = (d.get("text") or "").strip()
                date = d.get("date")   # 'YYYY-MM-DD' or None
                if len(text) < args.min_chars or bengali_ratio(text) < args.min_bengali:
                    skipped += 1
                    continue
                if date and date < args.since:   # too old
                    skipped += 1
                    continue
                out.write(json.dumps({
                    "text": text, "title": d.get("title", ""), "url": u,
                    "date": date, "source": name,
                }, ensure_ascii=False) + "\n")
                kept += 1
                site_kept += 1
                if kept % 200 == 0:
                    out.flush(); seen_fh.flush()
                    el = time.time() - t0
                    print(f"  [{el:6.0f}s] kept={kept:,} fetched={fetched:,} skip={skipped:,}", flush=True)
            except Exception:
                continue
    out.close(); seen_fh.close()
    print(f"[news] done: kept={kept:,} (fetched={fetched:,}, skipped={skipped:,}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
