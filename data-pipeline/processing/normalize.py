"""Text normalization pipeline for Bangla text.

Four stages:
0. HTML entity decoding
1. Unicode NFKC normalization
2. bnUnicodeNormalizer (Bengali-specific: nukta, hasanta, diacritics, ZWJ/ZWNJ)
   - MUST run after NFKC, and NFKC must NOT run again after this
   - bnorm recomposes nukta chars (U+09DC, U+09DD, U+09DF) which are
     composition-excluded in Unicode. Running NFKC after would decompose them.
3. Cleanup (URLs, emoji, pipe-to-danda, double hasanta, orphaned marks)

Stages 2-3 are skipped for Banglish (bn-Latn) text.
"""

import html
import re
import unicodedata

# Cache the bnUnicodeNormalizer instance (not thread-safe, but we're single-threaded)
_bnorm = None
_bnorm_available = None

# Regex to match contiguous Bengali script spans within a token
_BN_SPAN_RE = re.compile(r"([\u0980-\u09FF]+)")


def _get_bnorm():
    """Lazy-load and cache the bnUnicodeNormalizer."""
    global _bnorm, _bnorm_available
    if _bnorm_available is None:
        try:
            from bnunicodenormalizer import Normalizer
            _bnorm = Normalizer()
            _bnorm_available = True
        except ImportError:
            print("[normalize] WARNING: bnunicodenormalizer not installed, skipping stage 2")
            _bnorm_available = False
    return _bnorm


def normalize_text(text: str, is_banglish: bool = False) -> str:
    """Run the full normalization pipeline on a text.

    Args:
        text: Raw text to normalize
        is_banglish: If True, skip Bengali-specific normalization (stages 2-3)

    Returns:
        Normalized text
    """
    # Stage 0: Decode HTML entities (before anything else)
    text = _decode_html_entities(text)

    # Stage 1: Unicode NFKC (always applied)
    # No Bengali char has a compatibility decomposition, so NFKC and NFC produce
    # identical output for the Bengali block. NFKC additionally normalizes non-Bengali
    # chars in the text (e.g., NBSP -> space). Note: both NFC and NFKC will DECOMPOSE
    # the three nukta precomposed chars (U+09DC, U+09DD, U+09DF) because they are
    # composition-excluded. bnorm in stage 2 recomposes them afterward.
    text = unicodedata.normalize("NFKC", text)
    text = _clean_whitespace(text)

    if is_banglish:
        text = text.lower()  # Banglish: lowercase to reduce vocab ("Ami" == "ami")
        text = _cleanup_lite(text)  # URLs, emails, whitespace (skip Bengali-specific rules)
        return text

    # Stage 2: bnUnicodeNormalizer (word-by-word, Bengali-specific)
    # Handles: nukta recomposition, broken diacritics, hasanta validation,
    # conjunct validation, ZWJ/ZWNJ cleanup, Assamese char mapping.
    # IMPORTANT: Do NOT apply NFKC after this - it would undo nukta recomposition.
    text = _bn_unicode_normalize(text)

    # Stage 3: Cleanup (URLs, emoji, punctuation, remaining edge cases)
    text = _cleanup(text)

    return text


def _decode_html_entities(text: str) -> str:
    """Decode HTML entities like &nbsp; &lt; &#2453; etc."""
    return html.unescape(text)


def _clean_whitespace(text: str) -> str:
    """Normalize whitespace: collapse runs, strip lines."""
    # Replace various Unicode spaces with regular space
    text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000\uFEFF]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Clean up line endings
    text = re.sub(r"\r\n?", "\n", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _bn_unicode_normalize(text: str) -> str:
    """Bengali-specific Unicode normalization using bnUnicodeNormalizer.

    Applies bnorm only to contiguous Bengali script spans within each token,
    preserving punctuation and non-Bengali characters. This prevents bnorm
    from receiving mixed-script input (e.g., "আমি," or "বাংলা।") which
    can cause it to return None and silently drop words.

    Handles:
    - Nukta recomposition (ড+় -> ড়, ঢ+় -> ঢ়, য+় -> য়)
    - Broken diacritics (separate ে+া -> ো)
    - Invalid hasanta positions
    - Conjunct validation (184 known valid Bangla conjuncts)
    - ZWJ/ZWNJ cleanup (only র‍্য pattern preserved)
    - Assamese char mapping (ৰ->র, ৱ->ব)
    - Legacy symbol mapping
    """
    normalizer = _get_bnorm()
    if normalizer is None:
        return text

    normalized_lines = []
    for line in text.split("\n"):
        tokens = line.split()
        out_tokens = []
        for tok in tokens:
            if not tok:
                continue
            # Split token into Bengali spans and non-Bengali separators
            # e.g., "আমি," -> ["", "আমি", ","] or "hello" -> ["hello"]
            parts = _BN_SPAN_RE.split(tok)
            # Bengali spans are at odd indices after re.split with capture group
            for i in range(1, len(parts), 2):
                bengali_span = parts[i]
                try:
                    result = normalizer(bengali_span)
                    normalized = result.get("normalized")
                    if normalized is not None:
                        parts[i] = normalized
                    # If None, keep original Bengali span (don't drop content)
                except Exception:
                    pass  # Keep original on error
            out_tokens.append("".join(parts))
        normalized_lines.append(" ".join(out_tokens))

    return "\n".join(normalized_lines)


def _cleanup_lite(text: str) -> str:
    """Lightweight cleanup for Banglish text. URLs, emails, whitespace only."""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = _clean_whitespace(text)
    return text


def _cleanup(text: str) -> str:
    """Post-normalization cleanup for Bangla text.

    Handles:
    - URL removal
    - Email removal
    - Emoji removal
    - Pipe to danda conversion (| -> ।)
    - Double hasanta cleanup
    - Orphaned combining marks
    - Excessive punctuation
    """
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove emails
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # NOTE: Emojis are intentionally KEPT for general-purpose/chatbot training.
    # Real Bangla chat uses emojis heavily. Stripping them loses conversational signal.
    # Only the _remove_symbols() call below strips truly useless chars (dingbats, box drawing, etc).
    text = _remove_symbols(text)

    # Pipe to danda: | -> । unless BOTH sides are digits (don't break "3|4")
    # Convert if either side is not a digit: "a|b" -> "a।b", "3|b" -> "3।b"
    text = re.sub(r"(?<!\d)\||\|(?!\d)", "\u0964", text)

    # Double hasanta cleanup: consecutive ্্ -> single ্
    text = re.sub(r"\u09CD{2,}", "\u09CD", text)

    # Orphaned combining marks at word/line start
    # Bengali combining marks (vowel signs, nukta, chandrabindu, etc) need a base char
    text = re.sub(r"(?:^|(?<=\s))[\u09BE-\u09CD\u09D7\u09BC]+", "", text)

    # Clean residual orphaned ZWJ/ZWNJ not caught by bnorm
    # (bnorm handles most, this catches inter-word or sentence-level orphans)
    text = _clean_zwj(text)

    # Normalize excessive punctuation (3+ of same -> 1)
    text = re.sub(r"([।!?.]){3,}", r"\1", text)

    # Clean up whitespace again after removals
    text = _clean_whitespace(text)

    return text


def _clean_zwj(text: str) -> str:
    """Remove orphaned ZWJ/ZWNJ characters.

    Valid ZWJ/ZWNJ usage in Bengali requires adjacency to hasanta (্, U+09CD).
    bnUnicodeNormalizer handles most cases, but this catches sentence-level orphans.

    Valid patterns (preserved):
    - hasanta + ZWJ/ZWNJ + consonant (e.g., ্‌য in উদ্‌যাপন)
    - consonant + ZWJ + hasanta (e.g., র‍্য)

    Everything else is orphaned and gets removed.
    """
    hasanta = "\u09CD"
    bn_consonant = r"[\u0995-\u09B9\u09DC-\u09DF]"
    # Use Private Use Area char as sentinel (U+E000, never in real text)
    sentinel = "\uE000"

    # Protect valid: hasanta + ZWJ/ZWNJ + consonant
    text = re.sub(
        f"({hasanta})([\u200c\u200d])({bn_consonant})",
        lambda m: m.group(1) + sentinel + m.group(2) + m.group(3),
        text,
    )
    # Protect valid: consonant + ZWJ + hasanta
    text = re.sub(
        f"({bn_consonant})([\u200d])({hasanta})",
        lambda m: m.group(1) + sentinel + m.group(2) + m.group(3),
        text,
    )

    # Remove all remaining orphaned ZWJ/ZWNJ
    text = re.sub(r"[\u200c\u200d]", "", text)

    # Restore protected ones (remove sentinel, ZWJ/ZWNJ already in place)
    text = text.replace(sentinel, "")

    return text


def _remove_symbols(text: str) -> str:
    """Remove useless symbol characters while keeping emojis.

    Keeps: Bengali script, ASCII, emojis (U+1F000+), Devanagari danda/double-danda,
    and common punctuation. Only strips truly useless chars like box drawing,
    dingbats, technical symbols, etc.
    """
    cleaned = []
    for char in text:
        cp = ord(char)
        if (
            cp < 0x2500                    # Below box-drawing (keeps all common chars)
            or (0x0980 <= cp <= 0x09FF)    # Bengali block
            or (0x0964 <= cp <= 0x0965)    # Danda and double danda
            or (cp >= 0x1F000)             # Emojis (supplementary planes)
            or (0x2600 <= cp <= 0x27BF)    # Misc symbols + dingbats (includes common emojis)
            or (0xFE00 <= cp <= 0xFE0F)    # Variation selectors (emoji modifiers)
            or (0x200D == cp)              # ZWJ (emoji sequences like family emoji)
            or char.isascii()              # ASCII fallback
        ):
            cleaned.append(char)
    return "".join(cleaned)


def normalize_document(doc: dict) -> dict:
    """Normalize a document dict in-place.

    Detects Banglish from metadata and applies appropriate normalization.
    """
    is_banglish = (
        doc.get("source") == "banglish"
        or doc.get("metadata", {}).get("script") == "bn-Latn"
    )
    doc["text"] = normalize_text(doc["text"], is_banglish=is_banglish)
    doc["char_count"] = len(doc["text"])
    return doc
