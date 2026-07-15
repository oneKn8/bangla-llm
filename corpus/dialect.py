#!/usr/bin/env python3
"""Shibboleth dialect tagger for Bangladeshi Bengali (v1 seed lexicon).

Low-recall BY DESIGN: assigns a regional dialect only on a strong shibboleth,
else 'standard'. Regional dialects are barely written online, so this tags the
rare marked docs and leaves the rest 'standard' — coverage is reported honestly,
never balanced. Santo (Rangpur native) extends these lexicons. Markers cover
both native Bengali script and romanized (Banglish) forms.

Note: আঁই is shared by Sylheti and Chittagonian (ambiguous); ties resolve by
total hit count then dict order. Flag ambiguous tags downstream if needed.
"""
from __future__ import annotations

import re

# dialect -> {"strong": {...}, "weak": {...}}  (lowercased; native + romanized)
MARKERS: dict[str, dict[str, set[str]]] = {
    "rangpur": {
        "strong": {"মুই", "বাহে", "হামরা", "হামার", "মুক", "ছাওয়া",
                   "mui", "bahe", "hamra", "hamar", "muk"},
        "weak": {"তোই", "গেসোস", "করিস", "toi", "gesos", "korchos", "koris", "koi"},
    },
    "sylheti": {
        "strong": {"আঁই", "খিতা", "আমরার", "তুমরার", "বদ্দা",
                   "ai", "aai", "kita", "beda", "baddha"},
        "weak": {"অখল", "কিতাব", "oto", "okhol"},
    },
    "chittagong": {
        "strong": {"গুইন্না", "আঁর", "নঅ", "guinna", "aar", "noa"},
        "weak": {"aiso", "aiyer", "zaigoi", "honde"},
    },
    "noakhali": {
        "strong": {"হাঁচা", "আঁর", "hasa", "aar"},
        "weak": {"হেতে", "hete"},
    },
    "barishal": {
        "strong": {"হ্যায়", "হালার", "hyay"},
        "weak": {"বাদাইম্মা"},
    },
}

_WORD = re.compile(r"[\wঀ-৿]+", re.UNICODE)


def tag(text: str) -> tuple[str, int]:
    """Return (dialect, score). Confident regional tag only with 2+ strong
    shibboleths, OR 1 strong that is a real fraction of a short doc — so a standard
    article that merely *mentions* মুই once (e.g. Wikipedia's dialect section) stays
    'standard' instead of being mislabeled Rangpur. Conservative by design."""
    toks = [w.lower() for w in _WORD.findall(text)]
    n = max(1, len(toks))
    tokset = set(toks)
    best, best_strong, best_score = "standard", 0, 0
    for dia, m in MARKERS.items():
        strong = len(tokset & m["strong"])
        score = 2 * strong + len(tokset & m["weak"])
        if score > best_score:
            best, best_strong, best_score = dia, strong, score
    if best_strong >= 2 or (best_strong >= 1 and best_score / n >= 0.03):
        return best, best_score
    return "standard", 0


if __name__ == "__main__":
    cases = [
        ("মুই ভাত খাম, বাহে", "rangpur"),
        ("mui jam, tui koi gesos", "rangpur"),
        ("আঁই খিতা করমু", "sylheti"),
        ("guinna aiso, aar naam ki", "chittagong"),
        ("আমি ভাত খাব, তুমি কেমন আছ", "standard"),
        ("ami valo achi tui kemon achis", "standard"),
    ]
    ok = 0
    for txt, want in cases:
        d, s = tag(txt)
        good = d == want
        ok += good
        print(f"{'OK ' if good else 'XX '}{txt[:32]:34s} -> {d:11s}(score {s}) want {want}")
    print(f"\n{ok}/{len(cases)} correct")
