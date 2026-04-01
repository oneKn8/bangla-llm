# Bangla Data Collection Pipeline - Implementation Plan

## Context
Building a pure Bangla language model from scratch. Need clean, verified Bangla text data with zero Hindi/Assamese contamination. User has OCD about data quality - controls every source. Storage on Google Drive (2 accounts, 4TB total).

## Project Location
`/home/oneknight/projects/bangla-llm/data-pipeline/`

## Architecture

```
data-pipeline/
├── collect.py                 # Master CLI orchestrator
├── config.py                  # Paths, rate limits, thresholds
├── pipeline.py                # Processing: normalize -> filter -> dedup -> stats
├── export.py                  # Export compressed chunks to Drive
├── requirements.txt
├── collectors/
│   ├── base.py                # BaseCollector with download resume, state, JSONL write
│   ├── wikipedia.py           # bnwiki + bnwikisource dumps
│   ├── newspaper.py           # Async: Prothom Alo, Kaler Kantho, Ittefaq (NOT Daily Star - ai-train=no)
│   ├── literature.py          # Internet Archive public domain supplements
│   ├── web_crawl.py           # Seed list of Bangla sites + language filter
│   ├── banglish.py            # Reddit r/bangladesh, r/bangla via PRAW
│   └── hf_corpus.py           # OSCAR, Sangraha, CC-100 Bengali subsets (streaming)
├── processing/
│   ├── normalize.py           # NFKC + bnUnicodeNormalizer + csebuetnlp normalizer
│   ├── lang_detect.py         # fasttext lid.176.bin, Hindi/Assamese contamination check
│   ├── dedup.py               # SHA-256 exact + MinHash LSH near-dedup
│   ├── quality.py             # Min length, Bengali char ratio, boilerplate removal
│   └── stats.py               # Per-source stats + quality report
├── seeds/
│   ├── bangla_websites.txt    # ~20 seed URLs for web crawling
│   └── subreddits.txt         # Reddit targets
├── data/
│   ├── raw/                   # Per-source raw JSONL
│   ├── processed/             # After normalize + filter
│   ├── final/                 # Deduplicated, chunked, compressed
│   ├── state/                 # Resume state files + newspaper SQLite
│   └── reports/               # Quality reports
└── tests/
```

## Output Schema (every JSONL line)
```json
{
  "id": "sha256-first-128-chars",
  "text": "...",
  "source": "wikipedia|wikisource|prothomalo|kalerkantho|ittefaq|literature|web_crawl|banglish|oscar|sangraha",
  "url": "https://...",
  "title": "...",
  "timestamp": "2026-02-09T12:00:00Z",
  "lang": "bn",
  "lang_score": 0.98,
  "char_count": 4523,
  "metadata": {}
}
```

## Storage: Google Drive
- Raw data -> Drive Account 1: `My Drive/bangla-llm/raw/`
- Final processed data -> Drive Account 2: `My Drive/bangla-llm/final/`
- Compressed JSONL chunks (100MB each): `bangla_001.jsonl.gz`, `bangla_002.jsonl.gz`, ...
- manifest.json with chunk index, doc counts, hashes
- Local `data/` is working directory, synced to Drive after processing

## Data Sources & Estimates

| Source | Strategy | Raw Est. | Clean Est. | Docs |
|--------|----------|----------|------------|------|
| Wikipedia (bnwiki dump) | Download XML dump, wikiextractor | 1.2 GB | 800 MB | 130K |
| Wikisource (bnwikisource) | Download XML dump, wikiextractor | 600 MB | 450 MB | 15K |
| Prothom Alo | Async sitemap crawl, 1 req/sec | 800 MB | 500 MB | 150K |
| Kaler Kantho | Async sitemap crawl, 1 req/sec | 600 MB | 400 MB | 100K |
| Ittefaq | Async sitemap crawl, 1 req/sec | 400 MB | 250 MB | 80K |
| Literature (Archive.org) | API search + text download | 50 MB | 40 MB | 100 |
| OSCAR Bengali | HF streaming | 5 GB | 2 GB | 2M |
| Sangraha Bengali | HF streaming | 2 GB | 1.5 GB | 500K |
| CC-100 Bengali | Direct download | 3 GB | 1.5 GB | 1M |
| Web Crawl (seeds) | Async BFS crawl, lang filter | 1.5 GB | 800 MB | 200K |
| Banglish (Reddit) | PRAW API | 200 MB | 100 MB | 50K |
| **Total** | | **~15 GB** | **~8 GB** | **~4.2M** |

After cross-source dedup: **~5-6 GB, ~3-3.5M docs**

## Implementation Order (8 scripts)

### Script 1: config.py + base collector
- Central config (paths, rate limits, thresholds)
- BaseCollector ABC: download_with_resume(), save_documents(), load/save_state()

### Script 2: wikipedia.py
- Download bnwiki + bnwikisource dumps
- Run wikiextractor, convert to unified JSONL
- Filter stubs < 200 chars, disambiguation pages

### Script 3: newspaper.py
- Async with aiohttp + aiolimiter (1 req/sec/domain)
- Sitemap-driven collection for each newspaper
- SQLite state DB for resumability
- Per-newspaper HTML parsing (article body extraction)
- Exclude Daily Star Bangla (ai-train=no in robots.txt)

### Script 4: literature.py + web_crawl.py + banglish.py + hf_corpus.py
- Literature: Internet Archive API, supplement to Wikisource
- Web crawl: seed list, readability-lxml extraction, depth limit 3
- Banglish: PRAW, heuristic detection (keyword list + negative EN score)
- HF corpus: streaming download of OSCAR/Sangraha/CC-100

### Script 5: normalize.py
- Stage 1: Unicode NFKC
- Stage 2: bnUnicodeNormalizer (Bengali-specific fixes)
- Stage 3: csebuetnlp normalizer (URL/emoji strip)
- Skip stages 2-3 for Banglish (bn-Latn)

### Script 6: lang_detect.py + quality.py
- fasttext lid.176.bin, threshold >= 0.65
- Flag docs with Hindi/Assamese secondary scores > 0.2
- Devanagari character detection (U+0900-U+097F = contamination)
- Quality: min 100 chars, Bengali char ratio >= 50%, repeat line ratio < 30%

### Script 7: dedup.py
- Pass 1: SHA-256 exact dedup
- Pass 2: MinHash LSH (5-char shingles, threshold 0.8)
- Per-source first, then cross-source (disk-backed for RAM constraint)

### Script 8: collect.py (master) + pipeline.py + export.py + stats.py
- CLI: `python collect.py`, `--sources`, `--process-only`, `--resume`, `--report`
- Pipeline chains: normalize -> lang_detect -> quality -> dedup
- Export: split into 100MB gzipped JSONL chunks
- Stats: per-source counts, filter breakdown, sample docs

## Key Dependencies
```
aiohttp, aiolimiter, beautifulsoup4, lxml, readability-lxml
praw, requests, wikiextractor, robotexclusionrulesparser
bnunicodenormalizer, normalizer (csebuetnlp), fasttext-wheel
datasketch, datasets, tqdm, orjson
```

## Verification
1. Run each collector individually, check raw JSONL output
2. Sample 50 docs per source, manually verify they're clean Bangla
3. Run lang_detect on samples, confirm no Hindi/Assamese leaks
4. Check dedup removes known duplicate Wikipedia/Sangraha overlap
5. Final stats report shows per-source breakdown
