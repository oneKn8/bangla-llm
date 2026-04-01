#!/usr/bin/env python3
"""Audit a SentencePiece vocabulary for non-Bangla Indic script contamination.

Scans every token in the vocabulary and flags any that contain characters from
blocked Unicode ranges (Devanagari, Odia, Tamil, Telugu, Kannada, Malayalam,
Gurmukhi, Gujarati, and Assamese-only characters outside shared Bengali range).

Usage:
    python audit_vocab.py --model /path/to/bangla_bpe_32k.model
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from dataclasses import dataclass, field

import sentencepiece as spm

# ---------------------------------------------------------------------------
# Blocked Unicode ranges
# ---------------------------------------------------------------------------
# Each entry: (name, start_codepoint, end_codepoint)
# These are inclusive ranges.
BLOCKED_RANGES: list[tuple[str, int, int]] = [
    ("Devanagari", 0x0900, 0x097F),
    ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Odia", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
]

# Devanagari codepoints that are legitimately shared with Bengali/Bangla.
# U+0964 DEVANAGARI DANDA (।) -- standard sentence-ending punctuation in Bangla
# U+0965 DEVANAGARI DOUBLE DANDA (॥) -- used in Bangla verse/religious text
# These fall in the Devanagari range but are NOT contamination.
DEVANAGARI_SHARED_WITH_BENGALI: set[int] = {0x0964, 0x0965}

# Assamese-only characters that are NOT shared with Bengali.
# U+09F0 BENGALI LETTER RA WITH LOWER DIAGONAL (Assamese ৰ)
# U+09F1 BENGALI LETTER RA WITH MIDDLE DIAGONAL (Assamese ৱ)
ASSAMESE_ONLY_CODEPOINTS: set[int] = {0x09F0, 0x09F1}


@dataclass
class AuditResult:
    """Container for vocabulary audit results."""

    total_tokens: int = 0
    contaminated_tokens: list[dict[str, str]] = field(default_factory=list)
    passed: bool = True


def is_blocked_char(ch: str) -> str | None:
    """Return the blocking reason if *ch* falls in a blocked range, else None."""
    cp = ord(ch)

    # Allow Devanagari punctuation shared with Bengali (danda, double danda).
    if cp in DEVANAGARI_SHARED_WITH_BENGALI:
        return None

    if cp in ASSAMESE_ONLY_CODEPOINTS:
        return f"Assamese-only U+{cp:04X}"

    for name, lo, hi in BLOCKED_RANGES:
        if lo <= cp <= hi:
            return name

    return None


def audit_vocabulary(model_path: str) -> AuditResult:
    """Load a SentencePiece model and check every token for contamination.

    Returns an ``AuditResult`` with a list of contaminated tokens.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    result = AuditResult(total_tokens=sp.get_piece_size())
    seen_scripts: set[str] = set()

    for token_id in range(result.total_tokens):
        piece = sp.id_to_piece(token_id)

        for ch in piece:
            reason = is_blocked_char(ch)
            if reason is not None:
                char_name = unicodedata.name(ch, f"U+{ord(ch):04X}")
                result.contaminated_tokens.append(
                    {
                        "id": str(token_id),
                        "piece": piece,
                        "char": ch,
                        "codepoint": f"U+{ord(ch):04X}",
                        "char_name": char_name,
                        "script": reason,
                    }
                )
                seen_scripts.add(reason)
                break  # One bad char is enough to flag this token.

    result.passed = len(result.contaminated_tokens) == 0

    # --- Print report ---
    print("=" * 68)
    print("VOCABULARY AUDIT REPORT")
    print("=" * 68)
    print(f"Model:            {model_path}")
    print(f"Total tokens:     {result.total_tokens}")
    print(f"Contaminated:     {len(result.contaminated_tokens)}")
    print(f"Blocked scripts:  {', '.join(sorted(seen_scripts)) or 'none'}")
    print()

    if result.passed:
        print("RESULT: PASS -- No non-Bangla Indic script contamination found.")
    else:
        print("RESULT: FAIL -- Contaminated tokens detected:")
        print()
        print(f"  {'ID':>6}  {'Piece':<30}  {'Codepoint':<10}  {'Script'}")
        print(f"  {'--':>6}  {'-----':<30}  {'---------':<10}  {'------'}")
        for entry in result.contaminated_tokens[:50]:
            print(
                f"  {entry['id']:>6}  {entry['piece']:<30}  "
                f"{entry['codepoint']:<10}  {entry['script']}"
            )
        if len(result.contaminated_tokens) > 50:
            remaining = len(result.contaminated_tokens) - 50
            print(f"  ... and {remaining} more.")

    print("=" * 68)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit SentencePiece vocabulary for script contamination.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the .model file",
    )
    args = parser.parse_args()

    result = audit_vocabulary(args.model)
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
