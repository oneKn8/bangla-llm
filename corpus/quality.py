#!/usr/bin/env python3
"""Heuristic quality + safety tagging for corpus curation.

The curation filter that separates gold from junk (SEO keyword-spam, adult "choti"
content, PII, boilerplate, gibberish) so a fine-tune sees clean, coherent text.
All CPU, fast. Returns per-document flags + a 0..1 quality score; downstream selects
high-quality subsets by thresholding. Model-based fluency/topic tags come separately.
"""
from __future__ import annotations
import re

# --- adult / NSFW markers (Bengali erotic "choti" content is endemic in bn web crawl) ---
NSFW_MARKERS_BN = ["চটি", "চোদাচুদি", "যৌন গল্প", "সেক্স", "কামুক", "চুদা", "চোদা", "গুদ"]
NSFW_MARKERS_ROMAN = ["choti", "chodachudi", "chudir golpo", "choda chudi", "sex story",
                      "chudlam", "chudar", "sexgolpo", "banglachoti", "chodar"]
_NSFW_ROMAN_RE = re.compile(r"\b(" + "|".join(map(re.escape, NSFW_MARKERS_ROMAN)) + r")\b")

# --- PII ---
BD_PHONE_RE = re.compile(r"(?<!\d)(?:\+?880|0)1[3-9]\d{8}(?!\d)")   # Bangladeshi mobile
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

# --- spam / boilerplate signals ---
_SENT_SPLIT = re.compile(r"[।!?\n]")


def _repetition_ratio(text: str) -> float:
    """Fraction of lines that are duplicates (boilerplate/loops)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return 0.0
    return 1.0 - (len(set(lines)) / len(lines))


def _comma_list_ratio(text: str) -> float:
    """SEO keyword-stuffing looks like long comma-separated short tokens."""
    segs = [s.strip() for s in text.split(",")]
    if len(segs) < 6:
        return 0.0
    short = sum(1 for s in segs if 0 < len(s.split()) <= 3)
    return short / len(segs)


def quality_tags(text: str) -> dict:
    n = len(text)
    low = text.lower()
    # safety flags
    nsfw = any(m in text for m in NSFW_MARKERS_BN) or bool(_NSFW_ROMAN_RE.search(low))
    phones = BD_PHONE_RE.findall(text)
    emails = EMAIL_RE.findall(text)
    has_pii = bool(phones or emails)
    # junk signals
    rep = _repetition_ratio(text)
    comma_spam = _comma_list_ratio(text)
    n_urls = len(URL_RE.findall(text))
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    words = text.split()
    mean_sent_words = (len(words) / len(sents)) if sents else 0.0
    # symbol density (non-letter, non-space, non-Bengali/Latin punctuation)
    letters = sum(1 for c in text if c.isalpha())
    letter_ratio = letters / n if n else 0.0

    is_spam = comma_spam >= 0.6 or rep >= 0.5 or (mean_sent_words < 3 and len(words) > 20)
    is_short = len(words) < 15
    is_low_letter = letter_ratio < 0.45   # lots of digits/symbols -> tables/junk

    # composite 0..1 quality score (1 = clean prose). Deductive, clamped.
    score = 1.0
    if is_spam:        score -= 0.5
    if nsfw:           score -= 0.4
    score -= min(0.3, rep * 0.6)
    score -= min(0.2, comma_spam * 0.3)
    if is_low_letter:  score -= 0.2
    if is_short:       score -= 0.15
    score = round(max(0.0, min(1.0, score)), 3)

    return {
        "quality_score": score,
        "flags": {
            "nsfw": nsfw,
            "pii": has_pii,
            "spam": bool(is_spam),
            "short": bool(is_short),
            "repetitive": rep >= 0.3,
            "low_letter": bool(is_low_letter),
        },
        "signals": {
            "rep_ratio": round(rep, 3),
            "comma_list_ratio": round(comma_spam, 3),
            "mean_sent_words": round(mean_sent_words, 1),
            "n_urls": n_urls, "n_phones": len(phones), "n_emails": len(emails),
        },
    }


if __name__ == "__main__":
    tests = [
        "আজকে সকালে বাজারে গিয়ে দেখি সব সবজির দাম অনেক বেড়ে গেছে। মানুষ খুব কষ্টে আছে।",
        "Gp,Grameenphone,Robi,Airtel,Banglalink,2016,2017,2018,2019,2020,Welcome Tune Code List",
        "Bangla Choti Golpo chodachudir golpo আমার বন্ধু ফোন 01712345678 email x@y.com",
    ]
    import json
    for t in tests:
        print(t[:50], "->", json.dumps(quality_tags(t)["flags"]), "q=", quality_tags(t)["quality_score"])
