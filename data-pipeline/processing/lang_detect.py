"""Language detection and contamination checking using fasttext.

Uses fasttext lid.176.bin model to:
- Verify text is Bengali (bn)
- Flag Hindi/Assamese contamination
- Detect Devanagari characters (hard contamination signal)
"""

import os
from pathlib import Path

from config import (
    BLOCKED_SCRIPT_RANGES,
    CONTAMINATION_THRESHOLD,
    DEVANAGARI_RANGE,
    LANG_DETECT_THRESHOLD,
    ASSAMESE_ONLY_CHARS,
    BASE_DIR,
)

_model = None
MODEL_PATH = BASE_DIR / "lid.176.bin"
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


def _load_model():
    """Load the fasttext language identification model."""
    global _model
    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        print(f"[lang_detect] Downloading fasttext model to {MODEL_PATH}...")
        import requests
        resp = requests.get(MODEL_URL, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print("[lang_detect] Model downloaded.")

    import fasttext
    # Suppress fasttext warnings about loading model
    _model = fasttext.load_model(str(MODEL_PATH))
    return _model


def detect_language(text: str) -> dict:
    """Detect the language of text using fasttext.

    Returns:
        Dict with keys:
        - lang: predicted language code (e.g., "bn")
        - lang_score: confidence score (0-1)
        - secondary_langs: list of (lang, score) for top 5 predictions
        - is_bengali: True if passes threshold
        - contamination: dict of contamination signals
    """
    model = _load_model()

    # fasttext needs single line
    clean_text = text.replace("\n", " ")[:5000]
    predictions = model.predict(clean_text, k=5)

    labels = [l.replace("__label__", "") for l in predictions[0]]
    scores = [float(s) for s in predictions[1]]

    primary_lang = labels[0] if labels else "unknown"
    primary_score = scores[0] if scores else 0.0

    secondary_langs = list(zip(labels, scores))

    # Check for Hindi/Assamese in secondary predictions
    hindi_score = 0.0
    assamese_score = 0.0
    for lang, score in secondary_langs:
        if lang == "hi":
            hindi_score = score
        elif lang == "as":
            assamese_score = score

    # Hard contamination: Devanagari character detection
    # IMPORTANT: Exclude U+0964 (danda ।) and U+0965 (double danda ॥) from the count.
    # These live in the Devanagari block but are shared across 20+ Indic scripts
    # including Bengali. Every Bengali sentence ends with ।, so counting them
    # as "Devanagari contamination" would reject virtually all Bengali text.
    devanagari_count = _count_chars_in_range(text, DEVANAGARI_RANGE)
    danda_count = sum(1 for c in text if c in ('\u0964', '\u0965'))
    devanagari_count -= danda_count
    devanagari_count = max(devanagari_count, 0)  # Safety: never negative
    devanagari_ratio = devanagari_count / max(len(text), 1)

    # Assamese-only character detection
    assamese_char_count = sum(1 for c in text if c in ASSAMESE_ONLY_CHARS)

    # Other South Asian scripts
    other_script_count = sum(
        _count_chars_in_range(text, r) for r in BLOCKED_SCRIPT_RANGES
        if r != DEVANAGARI_RANGE  # already counted
    )

    contamination = {
        "hindi_score": hindi_score,
        "assamese_score": assamese_score,
        "devanagari_chars": devanagari_count,
        "devanagari_ratio": devanagari_ratio,
        "assamese_only_chars": assamese_char_count,
        "other_script_chars": other_script_count,
        "has_contamination": (
            hindi_score > CONTAMINATION_THRESHOLD
            or assamese_score > CONTAMINATION_THRESHOLD
            or devanagari_count > 0
            or assamese_char_count > 0
            or other_script_count > 0
        ),
    }

    return {
        "lang": primary_lang,
        "lang_score": primary_score,
        "secondary_langs": secondary_langs,
        "is_bengali": primary_lang == "bn" and primary_score >= LANG_DETECT_THRESHOLD,
        "contamination": contamination,
    }


def _count_chars_in_range(text: str, char_range: tuple[int, int]) -> int:
    """Count characters that fall within a Unicode range."""
    low, high = char_range
    return sum(1 for c in text if low <= ord(c) <= high)


def filter_document(doc: dict) -> tuple[dict, bool, str]:
    """Run language detection on a document and decide keep/reject.

    Args:
        doc: Document dict with 'text' field

    Returns:
        Tuple of (updated_doc, keep, reject_reason)
        - keep: True if document passes language filter
        - reject_reason: empty string if kept, otherwise reason for rejection
    """
    # Skip lang detection for Banglish - it's expected to be Latin script
    if doc.get("source") == "banglish":
        doc["lang"] = "bn-Latn"
        doc["lang_score"] = 1.0
        return doc, True, ""

    result = detect_language(doc["text"])

    doc["lang"] = result["lang"]
    doc["lang_score"] = result["lang_score"]
    doc["metadata"]["lang_detail"] = {
        "secondary_langs": result["secondary_langs"][:3],
        "contamination": result["contamination"],
    }

    if not result["is_bengali"]:
        return doc, False, f"not_bengali: {result['lang']}={result['lang_score']:.3f}"

    if result["contamination"]["has_contamination"]:
        reasons = []
        c = result["contamination"]
        if c["hindi_score"] > CONTAMINATION_THRESHOLD:
            reasons.append(f"hindi={c['hindi_score']:.3f}")
        if c["assamese_score"] > CONTAMINATION_THRESHOLD:
            reasons.append(f"assamese={c['assamese_score']:.3f}")
        if c["devanagari_chars"] > 0:
            reasons.append(f"devanagari={c['devanagari_chars']}")
        if c["assamese_only_chars"] > 0:
            reasons.append(f"assamese_chars={c['assamese_only_chars']}")
        if c["other_script_chars"] > 0:
            reasons.append(f"other_scripts={c['other_script_chars']}")
        return doc, False, "contamination: " + ", ".join(reasons)

    return doc, True, ""
