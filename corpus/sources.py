#!/usr/bin/env python3
"""Source registry for Kotha corpus collection.

`track` is an advisory hint; the purity classifier makes the final per-document
routing decision. Static source-partition across machines: assign different keys
to laptop / VPS / pod so they collect in parallel without overlap, then one global
cross-dedup pass merges them. Gated sources need `huggingface-cli login`.
"""
from __future__ import annotations

SOURCES: dict[str, dict] = {
    # --- Track A: native Bengali script (open) ---
    "fineweb2_bn": dict(hf_id="HuggingFaceFW/fineweb-2", config="ben_Beng",
                        split="train", text_field="text", track="native",
                        license="ODC-By", gated=False),
    "wikipedia_bn": dict(hf_id="wikimedia/wikipedia", config="20231101.bn",
                         split="train", text_field="text", track="native",
                         license="CC-BY-SA", gated=False),
    "hplt_bn": dict(hf_id="HPLT/HPLT2.0_cleaned", config="ben_Beng",
                    split="train", text_field="text", track="native",
                    license="CC0", gated=False),
    "sangraha_bn": dict(hf_id="ai4bharat/sangraha", data_dir="verified/ben",
                        split="train", text_field="text", track="native",
                        license="CC-BY-4.0", gated=False),
    # --- DIVERSITY adds: low Common-Crawl overlap (encyclopedic/literary), preflighted 2026-07-14 ---
    "wikisource_bn": dict(hf_id="wikimedia/wikisource", config="20231201.bn",
                          split="train", text_field="text", track="native",
                          license="CC-BY-SA", gated=False),
    # IndicCorpV2 (ben_Beng) preflighted OK 2026-07-14 — LARGE web corpus, heavy CC/sangraha
    # overlap (volume, not diversity); registered but NOT auto-collected. Launch only to scale raw.
    "indiccorp2_bn": dict(hf_id="ai4bharat/IndicCorpV2", config="indiccorp_v2",
                          split="ben_Beng", text_field="text", track="native",
                          license="see-card", gated=False),
    # Literary/textbook diversity (preflighted 07-14; permissive/PD only for a releasable corpus):
    "bangla_textbook": dict(hf_id="md-nishat-008/Bangla-TextBook",
                            split="train", text_field="text", track="native",
                            license="MIT", gated=False),
    # NOTE: gitanjali 'poems' field = ENGLISH translation; Bengali is 'bangla_poem'.
    "tagore_gitanjali": dict(hf_id="Shakil2448868/rabindranath-gitanjali",
                             split="train", text_field="bangla_poem", track="native",
                             license="PD (Tagore d.1941)", gated=False),
    # Bengali Wikisource FULL dump (richest CLEAN সাধু: proofread classics, no OCR).
    # Fed via local_jsonl produced by corpus/scrape_wikisource.py (not an HF stream).
    "wikisource_dump": dict(local_jsonl="corpus/data/wikisource/wikisource_pages.jsonl",
                            text_field="text", track="native",
                            license="CC-BY-SA (works PD)", gated=False),
    # --- RECENCY track: current data 2024 -> today (post-Hasina-flee coverage the corpus lacked) ---
    # Live bn.wikipedia dump (current, not the stale Nov-2023 HF snapshot) via scrape_wikisource.py --namespaces 0.
    "wikipedia_live": dict(local_jsonl="corpus/data/wikipedia_live/wikipedia_live_pages.jsonl",
                           text_field="text", track="native",
                           license="CC-BY-SA", gated=False),
    # Live Bengali news 2024->now via corpus/scrape_news.py (RSS+feeds). COPYRIGHTED -> train-only, keep url+date.
    "news_live": dict(local_jsonl="corpus/data/news/news.jsonl",
                      text_field="text", track="native",
                      license="news-train-only", gated=False),
    # --- PD/permissive CLASSICS (releasable core), wave-1 ingest 2026-07-14 ---
    "bongboi": dict(local_jsonl="corpus/data/bongboi/bongboi.jsonl",
                    text_field="text", track="native",
                    license="CC-BY-NC pkg (works PD)", gated=False),
    "eboipotro": dict(local_jsonl="corpus/data/eboipotro/eboipotro.jsonl",
                      text_field="text", track="native",
                      license="PD (works PD; repos unlicensed)", gated=False),
    "kaggle_tagore": dict(local_jsonl="corpus/data/kaggle_tagore/kaggle_tagore.jsonl",
                          text_field="text", track="native",
                          license="CC0", gated=False),
    # archive.org proof (68 docs); FULL pre-1930 run needs pd_verified + OCR-garble gate before release.
    "archive_bengali": dict(local_jsonl="corpus/data/archive_bengali/archive_bengali.jsonl",
                            text_field="text", track="native",
                            license="PD-subset (verify per-item)", gated=False),
    # --- TRAIN-ONLY literary (scraped/synthetic; do NOT redistribute; model-only) ---
    "sayurio_ekpatagolpo": dict(hf_id="sayurio/ekpatagolpo-scrape-bangla-literature", split="train",
                                text_field="content", track="native", license="scraped-train-only", gated=False),
    "sayurio_kobita": dict(hf_id="sayurio/bangla-kobita-scrape-bangla-literature", split="train",
                           text_field="content", track="native", license="scraped-train-only", gated=False),
    "sayurio_storymirror": dict(hf_id="sayurio/storymirror.com-web-scrape", split="train",
                                text_field="content", track="native", license="scraped-train-only", gated=False),
    "shakil_bangla_stories": dict(hf_id="Shakil2448868/Bangla-Stories", split="train",
                                  text_field="trans_bangla", track="native", license="synthetic-train-only", gated=False),
    # --- Track A: native (gated — need HF login + accept terms) ---
    "culturax_bn": dict(hf_id="uonlp/CulturaX", config="bn",
                        split="train", text_field="text", track="native",
                        license="mC4+OSCAR ToU", gated=True),
    # --- Track B: organic Banglish (romanized). Confirm exact ids before prod. ---
    "banglatlit_pt": dict(hf_id="aplycaebous/BanglaTLit",
                          split="train", text_field="text_transliterated", track="banglish",
                          force_track="banglish", license="see-card", gated=False),
    # additional dedicated Banglish sources (preflighted 2026-07-09: {id,text}, raw romanized)
    "banglish_kawsar_v3": dict(hf_id="kawsarahmd/banglish_dataset_v3",
                               split="train", text_field="text", track="banglish",
                               force_track="banglish", license="see-card", gated=False),
    "banglish_kawsar_v1": dict(hf_id="kawsarahmd/banglish_dataset_v1",
                               split="train", text_field="text", track="banglish",
                               force_track="banglish", license="see-card", gated=False),
    # gated — need HF terms acceptance (account midnightKernel), then re-preflight text_field before launch
    # NOTE: parallel/transliterated (Bengali<->Banglish<->English, emotion-labeled) -> NOT organic;
    # romanized column is "Banglish". Source-tag keeps it separable from organic banglatlit_pt.
    "banglish_kawsar_80k": dict(hf_id="kawsarahmd/banglish_80K_dataset_v1",
                                split="train", text_field="Banglish", track="banglish",
                                force_track="banglish", license="see-card", gated=True),
    "banglish_tensorlab_225k": dict(hf_id="tensorlabco/banglish_dataset_pretrained_225k",
                                    split="train", text_field="text", track="banglish",
                                    force_track="banglish", license="see-card", gated=True),
    # audit-holdout sources are the same corpora; provenance is tagged per-doc.
}


def get(name: str) -> dict:
    if name not in SOURCES:
        raise KeyError(f"unknown source '{name}'. known: {sorted(SOURCES)}")
    return SOURCES[name]
