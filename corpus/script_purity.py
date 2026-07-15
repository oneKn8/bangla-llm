#!/usr/bin/env python3
"""Script-purity + Banglish router for the Kotha v2 corpus.

Two-track corpus design:
  - Track A "native": pure Bengali-script text (the big clean pretraining pool).
  - Track B "banglish": romanized Bengali (Latin script) as people type on social
    media, plus light Bangla<->English code-switch. Deliberately KEPT, not filtered.
  - reject: other South Asian scripts (Hindi/Devanagari, Assamese, Odia, Tamil...),
    pure English, and low-quality junk.

This is the decision engine only (pure stdlib, unit-testable offline). It is meant
to be wrapped by a datatrove pipeline for the HF-streaming sources and by the news/
book/Banglish crawlers. GlotLID (fastText) is a PLUGGABLE second-stage LID gate:
pass a callable `lid=` to classify() to require top-label == 'ben_Beng'; without it
the Unicode gates still do the bulk of the work.

References: GlotLID (EMNLP 2023 findings), FineWeb-2 (arXiv 2506.20920).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

# --- Unicode ranges -------------------------------------------------------
BENGALI = (0x0980, 0x09FF)
ASSAMESE_ONLY = frozenset({0x09F0, 0x09F1})  # ৰ, ৱ — Bengali uses র (0x09B0), never these
BENGALI_DIGITS = (0x09E6, 0x09EF)
# Shared Indic punctuation that is legitimate in Bengali but encoded in the
# Devanagari block: । (dari, U+0964) ends nearly every Bengali sentence, ॥ (U+0965).
# Must be exempted from the Devanagari hard-reject or ALL punctuated Bangla dies.
SHARED_PUNCT = frozenset({0x0964, 0x0965})

# Other South-Asian / foreign scripts -> hard reject for a Bengali corpus.
# All ranges verified against unicode.org Blocks.txt + Unicode Standard v17.0 code
# charts (not from model memory). Bishnupriya Manipuri is written in the SAME
# Bengali-Assamese block, so it cannot be caught here — it needs the LID/marker layer.
BLOCKED_BLOCKS = (
    (0x0900, 0x097F),  # Devanagari (Hindi/Marathi/Sanskrit)
    (0xA8E0, 0xA8FF),  # Devanagari Extended
    (0x1CD0, 0x1CFF),  # Vedic Extensions
    (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0D80, 0x0DFF),  # Sinhala
    (0x0600, 0x06FF),  # Arabic (Urdu, base)
    (0x0750, 0x077F),  # Arabic Supplement
    (0x0870, 0x089F),  # Arabic Extended-B
    (0x08A0, 0x08FF),  # Arabic Extended-A (Urdu)
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A (common in Urdu)
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    (0xA800, 0xA82F),  # Syloti Nagri (Sylheti — own script, out of scope v1)
    (0xAAE0, 0xAAFF),  # Meetei Mayek Extensions (Manipuri/Meitei)
    (0xABC0, 0xABFF),  # Meetei Mayek (Manipuri/Meitei)
)

# High-frequency romanized-Bengali markers (social-media Banglish). Not exhaustive;
# a trained Banglish LID replaces this in production. Lowercased, ASCII.
BANGLISH_MARKERS = frozenset({
    "ami", "tumi", "tui", "apni", "amar", "tomar", "tor", "apnar", "amader", "tomader",
    "kemon", "kmn", "kemn", "acho", "acho", "achis", "achi", "ache", "asen", "asi",
    "valo", "bhalo", "vlo", "khub", "onek", "ektu", "kicu", "kichu", "kisu",
    "kore", "korbo", "korchi", "korish", "korte", "hobe", "hoise", "hoye", "hoyeche",
    "na", "nai", "ki", "ki", "kire", "keno", "kn", "ken", "kotha", "kota", "bolo", "bol",
    "bhai", "vai", "apu", "mama", "dost", "bondhu", "vabe", "kbe", "kobe", "ajke", "aj",
    "kal", "ekhon", "ekn", "tarpor", "thik", "thik", "accha", "achcha", "hae", "hmm",
    "koi", "kই", "dekho", "dekhi", "janina", "jani", "bujhi", "bujlam", "shob", "sob",
    "dhonnobad", "dhonnnobad", "onek", "besh", "beshi", "kharap", "mn", "mon", "hoy",
})

# Common English stopwords -> if Latin text is mostly these, it's English, not Banglish.
ENGLISH_MARKERS = frozenset({
    "the", "and", "is", "are", "was", "were", "of", "to", "in", "on", "for", "with",
    "this", "that", "have", "has", "will", "would", "you", "your", "they", "their",
    "from", "which", "there", "about", "what", "when", "where", "been", "can", "not",
})

DEFAULT = dict(
    min_bengali_chars=200,       # Track A: raise from repo's 100
    native_ratio=0.88,           # Bengali-letter ratio to keep as native
    latin_cap=0.05,              # max Latin fraction for native track
    blocked_abs=5,               # hard-reject if >= this many other-SA-script chars
    blocked_frac=0.005,          # or >= this fraction of letters
    # Assamese ROUTING reject is density-primary: a long Bengali doc that merely
    # mentions Assam or a name with a few ৰ/ৱ (e.g. the মৃদঙ্গ/khol article, a shared
    # Bengali-Assamese-Manipuri instrument) must be KEPT, not dropped. Genuine
    # Assamese-language text has DENSE ৰ/ৱ. The audit stream separately captures any
    # doc with >=3 ৰ/ৱ as a candidate for human gold review (high recall there,
    # high precision here). Final thresholds are tuned against the gold labels.
    assamese_abs=15,             # reject only with many ৰ/ৱ (whole Assamese passage)
    assamese_frac=0.005,         # or >= 0.5% of Bengali letters (dense = real Assamese)
    banglish_min_markers=2,      # min Banglish markers to accept a Latin doc
    banglish_min_words=4,        # min word count for Banglish track
)


@dataclass
class Verdict:
    track: str                    # 'native' | 'banglish' | 'reject'
    reason: str
    bengali_ratio: float = 0.0
    latin_ratio: float = 0.0
    bengali_chars: int = 0
    blocked_chars: int = 0
    assamese_chars: int = 0
    stats: dict = field(default_factory=dict)


def _in(cp: int, block: tuple[int, int]) -> bool:
    return block[0] <= cp <= block[1]


def normalize_text(s: str) -> str:
    """NFC only (safe) — used for detection AND as the canonical stored form.

    NFC does NOT fold Assamese ৰ (U+09F0) / ৱ (U+09F1), so same-script
    contamination stays visible. Do NOT run a Bengali morphological normalizer
    here: bnunicodenormalizer folds ৰ->র (U+09B0) and ৱ->ব (U+09AC), which ERASES
    the Assamese signal the contamination audit depends on. (This folding is
    itself a finding: the field's standard normalizer hides Assamese leakage.)
    """
    return unicodedata.normalize("NFC", s)


def bengali_morph_normalize(s: str) -> str:
    """OPTIONAL Bengali morphological normalization (bnunicodenormalizer): folds
    nukta forms (য়/ড়/ঢ়) and reorders vowel signs. WARNING: it also folds Assamese
    ৰ->র and ৱ->ব, so it MUST run only AFTER a doc is routed+audited, never before
    detection, and never on the audit stream. Apply only to final native training text."""
    try:
        _bn = _get_bn_normalizer()
        return " ".join(_bn(w)["normalized"] or w for w in s.split())
    except Exception:
        return s


_BN = None
def _get_bn_normalizer():
    global _BN
    if _BN is None:
        from bnunicodenormalizer import Normalizer  # type: ignore
        _BN = Normalizer()
    return _BN


def _counts(s: str) -> dict:
    bengali = latin = other_letter = assamese = blocked = 0
    for ch in s:
        cp = ord(ch)
        if cp in SHARED_PUNCT:
            continue  # । ॥ — neutral punctuation, not contamination
        if cp in ASSAMESE_ONLY:
            assamese += 1
            continue
        if _in(cp, BENGALI):
            cat = unicodedata.category(ch)
            if cat[0] == "L" or cat in ("Mn", "Mc"):  # letters + vowel signs
                bengali += 1
            continue
        if any(_in(cp, b) for b in BLOCKED_BLOCKS):
            blocked += 1
            continue
        cat = unicodedata.category(ch)
        if cat[0] == "L":
            if (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A) or (0x00C0 <= cp <= 0x024F):
                latin += 1
            else:
                other_letter += 1
    letters = bengali + latin + other_letter + assamese
    return dict(bengali=bengali, latin=latin, other=other_letter,
                assamese=assamese, blocked=blocked, letters=letters)


def _lc_words(s: str):
    return [w for w in "".join(c if (c.isalnum()) else " " for c in s.lower()).split() if w]


def classify(text: str, cfg: dict = DEFAULT, lid: Optional[Callable[[str], tuple[str, float]]] = None) -> Verdict:
    """Route a document to native / banglish / reject.

    lid: optional GlotLID-style callable returning (label, prob); when given, the
         native track additionally requires label == 'ben_Beng' with prob >= 0.70.
    """
    s = normalize_text(text)
    c = _counts(s)
    letters = max(1, c["letters"])
    bengali_ratio = c["bengali"] / letters
    latin_ratio = c["latin"] / letters

    # 1. Hard-reject other South Asian / foreign scripts (a single Hindi sentence -> drop).
    if c["blocked"] >= cfg["blocked_abs"] or c["blocked"] / letters >= cfg["blocked_frac"]:
        return Verdict("reject", "blocked_script", bengali_ratio, latin_ratio,
                       c["bengali"], c["blocked"], c["assamese"], c)

    # 1b. Reject significant non-Bengali, non-Latin SCRIPT (CJK, Cyrillic, Greek…) —
    #     e.g. a Bengali news doc quoting Chinese/French. Latin is NOT counted here; it
    #     is handled below as code-switch / romanized Banglish.
    if c["other"] / letters >= 0.05:
        return Verdict("reject", "foreign_script", bengali_ratio, latin_ratio,
                       c["bengali"], c["blocked"], c["assamese"], c)

    # 2. Hard-reject Assamese same-script contamination (the audit signal). Checked
    #    BEFORE routing so a heavily-Assamese doc cannot slip past the native ratio
    #    gate and leak into the Banglish track.
    if c["assamese"] >= cfg["assamese_abs"] or c["assamese"] / max(1, c["bengali"]) >= cfg["assamese_frac"]:
        return Verdict("reject", "assamese", bengali_ratio, latin_ratio,
                       c["bengali"], c["blocked"], c["assamese"], c)

    # 3. Native (pure Bengali script) track.
    if bengali_ratio >= cfg["native_ratio"] and latin_ratio <= cfg["latin_cap"]:
        if c["bengali"] < cfg["min_bengali_chars"]:
            return Verdict("reject", "too_short", bengali_ratio, latin_ratio,
                           c["bengali"], c["blocked"], c["assamese"], c)
        if lid is not None:
            label, prob = lid(s)
            if not (label == "ben_Beng" and prob >= 0.70):
                return Verdict("reject", f"lid:{label}:{prob:.2f}", bengali_ratio, latin_ratio,
                               c["bengali"], c["blocked"], c["assamese"], c)
        return Verdict("native", "ok", bengali_ratio, latin_ratio,
                       c["bengali"], c["blocked"], c["assamese"], c)

    # 4. Register B, split two ways:
    #    - "banglish"  = romanized Bengali (Latin-dominant), the register the paper claims.
    #    - "codeswitch" = Bengali-script-dominant text with real embedded Latin/English.
    #    A native-script source yields ~no true romanized Banglish; that comes from the
    #    dedicated Banglish sources (BanglaTLit-PT / social). Keeping them distinct so the
    #    ablation is about romanized Banglish, not native-with-Latin.
    words = _lc_words(s)
    bn_hits = sum(1 for w in words if w in BANGLISH_MARKERS)
    en_hits = sum(1 for w in words if w in ENGLISH_MARKERS)
    if latin_ratio > 0.5:
        if len(words) >= cfg["banglish_min_words"] and bn_hits >= cfg["banglish_min_markers"] and bn_hits >= en_hits:
            return Verdict("banglish", f"romanized(bn={bn_hits},en={en_hits})", bengali_ratio,
                           latin_ratio, c["bengali"], c["blocked"], c["assamese"], c)
        return Verdict("reject", f"english_or_other(bn={bn_hits},en={en_hits})", bengali_ratio,
                       latin_ratio, c["bengali"], c["blocked"], c["assamese"], c)
    if bengali_ratio >= 0.5 and latin_ratio >= cfg["latin_cap"]:
        return Verdict("codeswitch", f"codeswitch(lat={latin_ratio:.2f},bn={bn_hits})", bengali_ratio,
                       latin_ratio, c["bengali"], c["blocked"], c["assamese"], c)

    return Verdict("reject", "low_signal", bengali_ratio, latin_ratio,
                   c["bengali"], c["blocked"], c["assamese"], c)


# --- offline self-test ----------------------------------------------------
if __name__ == "__main__":
    long_bn = ("বাংলাদেশ দক্ষিণ এশিয়ার একটি সার্বভৌম রাষ্ট্র। এর রাজধানী ঢাকা। "
               "বাংলা এই দেশের রাষ্ট্রভাষা এবং সংস্কৃতির প্রাণকেন্দ্র। ") * 6
    cases = [
        ("pure native (long)", long_bn, "native"),
        ("native w/ tiny english", long_bn + " (GDP)", "native"),
        ("hindi contamination", "বাংলা ভাষা সুন্দর। यह हिंदी वाक्य है और यह दूषण है।", "reject"),
        ("assamese (dense)", "মোৰ নাম ৰাম, মই ভাল আছোঁ ৰৱ ৰৰ ৱৱ এই সমস্ত অসমীয়া বৰ্ণ।", "reject"),
        ("bengali w/ few ৰ (kept)", long_bn * 4 + " ৰ ৱ ৰ", "native"),  # মৃদঙ্গ-type: 3 stray Assamese in a long Bengali doc -> KEEP
        ("romanized banglish", "tomar ki khobor? ami valo achi. tui kemon achis bhai onek din por", "banglish"),
        ("bengali-script codeswitch", "আজকে office এ গিয়ে দেখি সব meeting cancel হয়ে গেছে, assignment টা submit করতে হবে আজ রাতেই।", "codeswitch"),
        ("pure english", "The weather is nice today and we are going out for a walk with friends.", "reject"),
        ("short bangla", "ঢাকা।", "reject"),
    ]
    ok = 0
    for name, txt, want in cases:
        v = classify(txt)
        good = v.track == want
        ok += good
        flag = "OK " if good else "XX "
        print(f"{flag}{name:26s} -> {v.track:9s} (want {want:9s})  "
              f"bn={v.bengali_ratio:.2f} lat={v.latin_ratio:.2f} blk={v.blocked_chars} asm={v.assamese_chars} [{v.reason}]")
    print(f"\n{ok}/{len(cases)} correct")
