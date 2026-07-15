#!/usr/bin/env python3
"""সাধু (sadhu-bhasha) detector for the Kotha corpus.

Marker-frequency scorer that pulls the classical literary register (সাধু) out of
already-collected text with zero new scraping. Sadhu is distinguished from চলিত
(cholito) mainly by finite/nonfinite verb morphology (করিয়া/করিল/হইয়াছে) and
archaic pronouns/particles (তাহার/যাহা/হইতে). Marker set verified against prior
work (BanglaBlend, Sadhu2Colito) and tuned to avoid the known cholit leaks.

Score = 100 * marker_tokens / bengali_tokens.  Classify সাধু if per100 >= 5.0 AND
>= 2 distinct marker categories fire (≈95% precision, eyeball-verified). A lower
per100 >= 2.5 gate gives higher recall (~55-60% precision) and pairs with a
Sanskrit-śloka reject (Devanagari double-danda ॥ density + no Bengali finite verb).

Substring markers are naturally OCR-robust (they survive garble like হইপের/নাহৰে).
Runs pre-normalization on NFC text (never after morphological folding).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Bengali letters block (letters only; excludes digits/danda)
_BENGALI_LETTER = re.compile(r"[ঀ-৿]")
# strip surrounding punctuation/quotes/danda so suffix tests hit the bare word
_STRIP = "।॥,.;:!?\"'`()[]{}<>–—-…«»“”‘’*_/\\|@#৷৶৸৹‌‍ "

# --- exclusion sets (known cholit / neutral leaks the markers must NOT count) ---
_HOI_OK = None  # হই-startswith handled inline
_PARTICIPLE_STOP = {"দুনিয়া", "দরিয়া", "মিডিয়া", "হালুয়া", "জিনিয়া"}   # -িয়া non-verbal
_FINITE_STOP = {"ছিলেন", "ছিলাম"}                                         # copula (register-neutral)
_FUTURE_STOP = {"সচিব"}                                                    # noun, not -িবে verb
_PRONOUNS = {
    "তাহা", "তাহার", "তাহাকে", "তাহাদের", "তাহাদিগ", "তাহাদিগের", "তাঁহার", "তাঁহাকে",
    "ইহা", "ইহার", "ইহাকে", "ইহাদের", "উহা", "উহার", "উহাকে",
    "যাহা", "যাহার", "যাহাকে", "যাহাদের", "যাঁহারা", "যাঁহার",
}
_PARTICLES = {"হইতে", "সহিত", "নহে", "নাই", "বটে", "নিমিত্ত", "যথায়", "তথায়", "কদাচ"}

_CATS = "abcdefghi"


@dataclass
class SadhuScore:
    per100: float
    cats: str            # distinct categories that fired, e.g. "aeh"
    hits: int
    bn_tokens: int
    is_sadhu: bool       # high-precision gate
    is_sadhu_hr: bool    # high-recall gate (with śloka reject)
    is_shloka: bool


def _match_cats(t: str) -> str:
    """Return the set (as a string) of sadhu marker categories a token matches."""
    c = ""
    # (a) হই-copula: হইয়া/হইল/হইবে/হইতে/হইয়াছে
    if t.startswith("হই") and len(t) >= 4:
        c += "a"
    # (b) perfect -িয়াছ : করিয়াছে
    if "িয়াছ" in t:
        c += "b"
    # (c) continuous -িতেছ : করিতেছে
    if "িতেছ" in t:
        c += "c"
    # (d) verbal-noun -িবার : করিবার
    if "িবার" in t:
        c += "d"
    # (e) participle -িয়া (len>=5, not the noun stoplist) : করিয়া/বলিয়া
    if t.endswith("িয়া") and len(t) >= 5 and t not in _PARTICIPLE_STOP:
        c += "e"
    # (f) finite -িলেন/-িলাম excluding copula ছিলেন/ছিলাম : বলিলেন
    if (t.endswith("িলেন") or t.endswith("িলাম")) and t not in _FINITE_STOP:
        c += "f"
    # (g) future -িবে/-িবেন (len>=5, excl সচিব) : করিবে/করিবেন
    if (t.endswith("িবেন") or t.endswith("িবে")) and len(t) >= 5 and t not in _FUTURE_STOP:
        c += "g"
    # (h) archaic pronouns
    if t in _PRONOUNS:
        c += "h"
    # (i) archaic particles
    if t in _PARTICLES:
        c += "i"
    return c


def classify(text: str) -> SadhuScore:
    if not text:
        return SadhuScore(0.0, "", 0, 0, False, False, False)
    text = unicodedata.normalize("NFC", text)   # য়/ড়/ঢ় etc. have decomposed forms that break substring markers
    bn_tokens = 0
    hits = 0
    cats: set[str] = set()
    for raw in text.split():
        t = raw.strip(_STRIP)
        if not t or not _BENGALI_LETTER.search(t):
            continue
        bn_tokens += 1
        m = _match_cats(t)
        if m:
            hits += 1
            cats.update(m)
    per100 = 100.0 * hits / bn_tokens if bn_tokens else 0.0
    catstr = "".join(sorted(cats))
    # Sanskrit-śloka / Devanagari-danda reject: dense ॥ but no Bengali finite verb
    danda2 = text.count("॥")           # ॥ double danda (śloka boundary)
    has_finite = bool(cats & set("abcfg"))  # any finite/copula verb category
    is_shloka = danda2 >= 2 and not has_finite
    ncats = len(cats)
    is_sadhu = per100 >= 5.0 and ncats >= 2
    is_sadhu_hr = per100 >= 2.5 and ncats >= 2 and not is_shloka
    return SadhuScore(round(per100, 2), catstr, hits, bn_tokens, is_sadhu, is_sadhu_hr, is_shloka)


if __name__ == "__main__":
    import json
    import statistics
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "corpus/data/audit_v2_gold_all_n20000.jsonl"
    field = sys.argv[2] if len(sys.argv) > 2 else "text"
    scores, by_src, sadhu_hits = [], {}, []
    n = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get(field, "") or ""
        r = classify(t)
        scores.append(r.per100)
        n += 1
        src = d.get("source", d.get("route", "?"))
        by_src.setdefault(src, []).append(r)
        if r.is_sadhu:
            sadhu_hits.append((r.per100, r.cats, t[:160].replace("\n", " ")))
    scores.sort()
    p = lambda q: scores[min(len(scores) - 1, int(q * len(scores)))] if scores else 0
    print(f"n={n}  p50={statistics.median(scores):.2f}  p90={p(0.9):.2f}  p99={p(0.99):.2f}  max={max(scores):.2f}")
    print(f"is_sadhu (hi-prec per100>=5 & >=2 cats): {sum(1 for r in (x for v in by_src.values() for x in v) if r.is_sadhu)}"
          f"   is_sadhu_hr (recall): {sum(1 for r in (x for v in by_src.values() for x in v) if r.is_sadhu_hr)}")
    print("--- per-source median per100 (discrimination check) ---")
    for src, v in sorted(by_src.items(), key=lambda kv: -statistics.median([r.per100 for r in kv[1]])):
        meds = statistics.median([r.per100 for r in v])
        frac = 100 * sum(1 for r in v if r.is_sadhu) / len(v)
        print(f"  {src:28s} n={len(v):5d}  median={meds:5.2f}  %sadhu={frac:5.1f}")
    print("--- top sadhu docs (eyeball) ---")
    for s, cats, t in sorted(sadhu_hits, reverse=True)[:8]:
        print(f"  [{s:5.1f} {cats:>6s}] {t}")
