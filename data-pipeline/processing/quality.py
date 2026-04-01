"""Quality filtering for Bangla text documents.

Filters:
- Minimum document length
- Bengali character ratio (script check)
- Repeat line ratio (boilerplate detection)
- Boilerplate pattern removal
"""

import re
from collections import Counter

from config import (
    BENGALI_RANGE,
    MAX_REPEAT_LINE_RATIO,
    MIN_BENGALI_CHAR_RATIO,
    MIN_DOC_LENGTH,
)

# Common boilerplate patterns in Bangla web content
BOILERPLATE_PATTERNS = [
    re.compile(r"সর্বস্বত্ব\s+সংরক্ষিত"),  # All rights reserved
    re.compile(r"মন্তব্য\s+করুন"),  # Leave a comment
    re.compile(r"শেয়ার\s+করুন"),  # Share this
    re.compile(r"আরও\s+পড়ুন"),  # Read more
    re.compile(r"বিজ্ঞাপন"),  # Advertisement
    re.compile(r"কুকি\s*পলিসি|গোপনীয়তা\s*নীতি"),  # Cookie/privacy policy
    re.compile(r"নিউজলেটার"),  # Newsletter
    re.compile(r"সাবস্ক্রাইব"),  # Subscribe
]


def check_quality(text: str, is_banglish: bool = False) -> tuple[bool, str]:
    """Check if text passes quality filters.

    Args:
        text: Document text
        is_banglish: If True, skip Bengali char ratio check

    Returns:
        Tuple of (passes, reject_reason)
    """
    # Min length
    if len(text) < MIN_DOC_LENGTH:
        return False, f"too_short: {len(text)} < {MIN_DOC_LENGTH}"

    # Bengali character ratio (skip for Banglish)
    if not is_banglish:
        bengali_ratio = _bengali_char_ratio(text)
        if bengali_ratio < MIN_BENGALI_CHAR_RATIO:
            return False, f"low_bengali_ratio: {bengali_ratio:.3f} < {MIN_BENGALI_CHAR_RATIO}"

    # Repeat line ratio
    repeat_ratio = _repeat_line_ratio(text)
    if repeat_ratio > MAX_REPEAT_LINE_RATIO:
        return False, f"high_repeat_ratio: {repeat_ratio:.3f} > {MAX_REPEAT_LINE_RATIO}"

    return True, ""


def clean_boilerplate(text: str) -> str:
    """Remove common boilerplate patterns from text."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        if any(p.search(stripped) for p in BOILERPLATE_PATTERNS):
            continue
        cleaned.append(line)

    result = "\n".join(cleaned)
    # Collapse excessive blank lines after removal
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _bengali_char_ratio(text: str) -> float:
    """Calculate the ratio of Bengali script characters in text."""
    if not text:
        return 0.0

    low, high = BENGALI_RANGE
    bengali_count = sum(1 for c in text if low <= ord(c) <= high)
    # Count all alphabetic/script characters (not spaces/punctuation/digits)
    alpha_count = sum(1 for c in text if c.isalpha())

    if alpha_count == 0:
        return 0.0

    return bengali_count / alpha_count


def _repeat_line_ratio(text: str) -> float:
    """Calculate the ratio of repeated lines in text.

    Used to detect boilerplate-heavy pages (nav menus, footers repeated).
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) <= 1:
        return 0.0

    counts = Counter(lines)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(lines)


def filter_document(doc: dict) -> tuple[dict, bool, str]:
    """Run quality checks on a document.

    Args:
        doc: Document dict

    Returns:
        Tuple of (doc, keep, reject_reason)
    """
    is_banglish = (
        doc.get("source") == "banglish"
        or doc.get("metadata", {}).get("script") == "bn-Latn"
    )

    # Clean boilerplate first
    doc["text"] = clean_boilerplate(doc["text"])
    doc["char_count"] = len(doc["text"])

    # Run quality checks
    passes, reason = check_quality(doc["text"], is_banglish=is_banglish)
    return doc, passes, reason
