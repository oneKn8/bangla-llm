#!/usr/bin/env python3
"""Bengali + Banglish profanity / slur seed lexicon for corpus toxicity tagging.

Purpose: DOCUMENTATION and FILTERING of a research corpus (flag toxic spans so they
can be reported honestly, down-weighted, or held out) - not generation. This is a
SEED; Santo (native speaker) expands it, same as the dialect shibboleth lexicon.

Matching is word-level and lowercased for romanized forms, and substring-with-guards
for Bengali script (Bengali words are not always whitespace-delimited). Keep entries
lowercase. Group by severity so downstream can threshold.
"""
from __future__ import annotations
import re

# --- Bengali-script terms (common vulgar/abusive; seed set) ---
PROFANITY_BN = {
    "high": [
        "খানকি", "মাগি", "বেশ্যা", "চুদ", "চোদা", "চুদি", "চোদন",
        "বাল", "গুদ", "ধন", "খানকির", "মাদারচোদ", "বোকাচোদা",
    ],
    "medium": [
        "শালা", "শালি", "হারামি", "হারামজাদা", "কুত্তা", "কুত্তার বাচ্চা",
        "বাটপার", "শুয়োর", "শুওরের বাচ্চা", "জারজ", "বেজন্মা",
    ],
    "mild": [
        "গাধা", "বলদ", "পাগল", "ছাগল", "চোর", "মিথ্যুক",
    ],
}

# --- Romanized (Banglish) variants; many spellings per lexeme by design ---
PROFANITY_ROMAN = {
    "high": [
        "khanki", "khankir", "magi", "magir", "beshya", "chud", "choda", "chudi",
        "chodon", "gud", "gudmara", "madarchod", "madarchod", "bokachoda", "bokacoda",
    ],
    "medium": [
        "shala", "salaa", "shali", "harami", "haramjada", "kutta", "kuttar baccha",
        "batpar", "batpaar", "shuor", "shuorer baccha", "jaroj", "bejonma",
    ],
    "mild": [
        "gadha", "gadhaa", "bolod", "pagol", "chagol", "chor", "mittuk",
    ],
}

SEVERITY = ("high", "medium", "mild")

# Precompiled romanized word matchers (word-boundary, lowercased input)
_ROMAN_RE = {
    sev: re.compile(r"\b(" + "|".join(sorted({re.escape(w) for w in words}, key=len, reverse=True)) + r")\b")
    for sev, words in PROFANITY_ROMAN.items()
}
# Bengali-script terms sorted longest-first so multiword/compound matches win
_BN_TERMS = {sev: sorted(set(words), key=len, reverse=True) for sev, words in PROFANITY_BN.items()}


def find_profanity(text: str) -> dict:
    """Return {'hits': [term...], 'severity': 'high'|'medium'|'mild'|None, 'count': int}.
    Scans both Bengali-script (substring) and romanized (word-boundary) forms."""
    hits, sev_found = [], set()
    low = text.lower()
    for sev in SEVERITY:
        for term in _BN_TERMS[sev]:
            if term in text:
                hits.append(term); sev_found.add(sev)
        for m in _ROMAN_RE[sev].finditer(low):
            hits.append(m.group(1)); sev_found.add(sev)
    severity = next((s for s in SEVERITY if s in sev_found), None)  # worst present
    return {"hits": sorted(set(hits)), "severity": severity, "count": len(hits)}


if __name__ == "__main__":
    for t in ["tui ekta batpar", "সে একটা খানকির পোলা", "আজকে আবহাওয়া ভালো", "darun khobor"]:
        print(repr(t), "->", find_profanity(t))
