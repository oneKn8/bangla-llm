#!/usr/bin/env python3
"""Build a higher-quality assistant-style SFT dataset.

This script is intentionally stricter than the earlier curation pass:
- removes textbook / exercise contamination
- removes step-by-step / markdown-tutor style answers
- penalizes vague generic assistant filler
- prefers creative, structured, and information-dense answers
- mixes in gold sets as a high-quality core

By default it writes:
  finetune/data/kotha_final_train.jsonl
  finetune/data/kotha_final_val.jsonl
  finetune/data/kotha_final_report.json
  finetune/data/REVIEW_FINAL_DATASET.txt
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BANGLA_RANGE = re.compile(r"[\u0980-\u09FF]")
WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?।]+")

TEXTBOOK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"###",
        r"ধাপে ধাপে",
        r"ধাপ\s*[০-৯0-9]+",
        r"কাজ[-–]?\s*[০-৯0-9]+",
        r"পাঠ\s*[০-৯0-9]+",
        r"অধ্যা[য়য]",
        r"শিখনফল",
        r"অনুশীলনী",
        r"সমাধান করার জন্য",
        r"নিম্নলিখিতভাবে",
        r"উদ্দীপক",
        r"প্রশ্নের উত্তর",
        r"ফাঁকা স্থানে",
        r"বহুনির্বাচনী",
    ]
]

GENERIC_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"করতে পারেন",
        r"করতে পারো",
        r"চেষ্টা করুন",
        r"চেষ্টা করো",
        r"মনোযোগ দিন",
        r"ভালো হবে",
        r"সাহায্য করবে",
        r"ধীরে ধীরে",
        r"প্রথমে নিশ্চিত",
        r"তারপরও সমস্যা হলে",
        r"নিয়মিত অনুশীলন করুন",
        r"নিয়মিত অনুশীলন করো",
        r"ইতিবাচক থাকুন",
        r"নিজের উন্নতির দিকে",
    ]
]

META_RESPONSE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^অবশ্যই",
        r"^নিশ্চয়ই",
        r"^নিশ্চয়ই",
        r"^নিচে",
        r"^এখানে",
        r"^প্রথমে",
        r"^চলুন",
        r"^আমি .*সাহায্য করতে পারি",
        r"\*\*",
        r"---",
    ]
]

TIME_SENSITIVE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"আজকের",
        r"এখন",
        r"আজ\b",
        r"খবর কী",
        r"দাম কত",
        r"কত টাকা কেজি",
        r"আজ.*খেলা",
    ]
]

TASK_RATIOS = {
    "factual": 0.18,
    "advice": 0.18,
    "creative": 0.24,
    "dialogue": 0.12,
    "translation": 0.04,
    "math": 0.12,
    "structured": 0.12,
}

CREATIVE_NOUNS = [
    "কবিতা",
    "ছড়া",
    "গল্প",
    "ছোটগল্প",
    "চিঠি",
    "ডায়েরি",
    "ডায়েরি",
    "হাইকু",
    "গান",
    "দৃশ্য",
    "বার্তা",
    "মেসেজ",
    "ক্যাপশন",
    "স্ট্যাটাস",
    "শুভেচ্ছা",
    "উইশ",
    "পোস্ট",
    "টুইট",
    "রিপ্লাই",
]

CREATIVE_PATTERN = "|".join(re.escape(noun) for noun in CREATIVE_NOUNS)

GENERATION_REQUEST_RE = re.compile(
    r"(লিখ(?:ে)?(?:ুন|ো)?|রচনা\s+কর(?:ো|ুন)|তৈরি\s+কর(?:ো|ুন)|শোনাও|দাও|বল(?:ো|ুন))"
)

CREATIVE_REQUEST_RE = re.compile(
    rf"({CREATIVE_PATTERN}).{{0,40}}"
    r"(লিখ(?:ে)?(?:ুন|ো)?|রচনা\s+কর(?:ো|ুন)|তৈরি\s+কর(?:ো|ুন)|শোনাও|দাও|বল(?:ো|ুন))"
    r"|"
    r"(লিখ(?:ে)?(?:ুন|ো)?|রচনা\s+কর(?:ো|ুন)|তৈরি\s+কর(?:ো|ুন)|শোনাও|দাও|বল(?:ো|ুন)).{0,40}"
    rf"({CREATIVE_PATTERN})"
)

DIALOGUE_REQUEST_RE = re.compile(
    r"(সংলাপ|কথোপকথন|চ্যাট).{0,40}"
    r"(লিখ(?:ে)?(?:ুন|ো)?|রচনা\s+কর(?:ো|ুন)|তৈরি\s+কর(?:ো|ুন)|দাও|বল(?:ো|ুন))"
    r"|"
    r"(লিখ(?:ে)?(?:ুন|ো)?|রচনা\s+কর(?:ো|ুন)|তৈরি\s+কর(?:ো|ুন)|দাও|বল(?:ো|ুন)).{0,40}"
    r"(সংলাপ|কথোপকথন|চ্যাট)"
)

TRANSLATION_REQUEST_RE = re.compile(
    r"অনুবাদ|translate|বাংলায়\s+কর|বাংলায়\s+কর|ইংরেজিতে\s+কর|হিন্দিতে\s+কর|উর্দুতে\s+কর"
)

LANGUAGE_HINT_RE = re.compile(r"বাংলায়|বাংলায়|ইংরেজিতে|হিন্দিতে|উর্দুতে|in english|in bangla", re.IGNORECASE)

GEOMETRY_TRANSLATION_HINTS = [
    "ট্র্যাপিজয়েড",
    "ট্র্যাপিজয়েড",
    "জ্যামিতিক",
    "প্রিজম",
    "পিরামিড",
    "চতুর্ভুজ",
    "সরানো",
    "স্থানান্তর",
]

ADVICE_HINTS = [
    "কিভাবে",
    "কি করব",
    "কী করব",
    "উচিত",
    "পারি",
    "পরামর্শ",
    "সমস্যা",
    "সাহায্য",
    "উদ্বিগ্ন",
    "উদ্বেগ",
    "চাপ",
    "মন খারাপ",
    "কষ্ট",
    "ভয়",
    "ভয়",
    "একাকী",
    "হতাশ",
    "বিষণ্ন",
    "আসক্ত",
    "সম্পর্ক",
    "দ্বন্দ্ব",
    "ভেঙে",
    "কান্না",
    "রাগ",
    "অভিমান",
]

EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
EMOTION_WORDS = [
    "খুশি",
    "আনন্দ",
    "দুঃখ",
    "বেদনা",
    "ভালোবাসা",
    "রাগ",
    "অভিমান",
    "হাসি",
    "কান্না",
    "স্বপ্ন",
    "মায়া",
    "মায়া",
    "ভয়",
    "ভয়",
    "উচ্ছ্বাস",
    "একাকী",
    "সুখ",
    "আশা",
]

PRIORITY_SOURCE_PREFIXES = ("sft_style_",)
PRIORITY_SOURCE_MIN_COUNTS = {
    "sft_style_": 180,
}
GENERATED_SOURCE_NAMES = {
    "sft_gemini.jsonl",
}


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())


def bangla_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    bangla_count = len(BANGLA_RANGE.findall(text))
    alpha_count = sum(
        1
        for c in text
        if not c.isspace()
        and c not in ".,;:!?-()[]{}\"'`/\\|@#$%^&*+=~<>0123456789"
    )
    if alpha_count == 0:
        return 0.0
    return bangla_count / alpha_count


def sentence_count(text: str) -> int:
    return len([p for p in SENTENCE_SPLIT_RE.split(text) if p.strip()])


def word_count(text: str) -> int:
    return len(normalize_text(text).split())


def repeated_ngram_ratio(text: str, n: int = 3) -> float:
    tokens = normalize_text(text).split()
    if len(tokens) < n * 3:
        return 0.0

    counts: Counter[tuple[str, ...]] = Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )
    if not counts:
        return 0.0

    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / max(len(tokens) - n + 1, 1)


def has_repeated_lines(text: str) -> bool:
    lines = [normalize_text(line) for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    return len(set(lines)) != len(lines)


def extract_content(message: Any) -> str:
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content

    parts = message.get("parts")
    if isinstance(parts, list):
        extracted: list[str] = []
        for part in parts:
            if isinstance(part, str):
                extracted.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                extracted.append(part["text"])
        return "\n".join(extracted)

    return ""


def extract_pair(obj: dict[str, Any]) -> tuple[str, str] | None:
    messages = obj.get("messages")
    if isinstance(messages, list):
        user = ""
        assistant = ""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = extract_content(msg)
            if role == "user" and not user:
                user = content
            elif role == "assistant" and not assistant:
                assistant = content
        if user and assistant:
            return user, assistant

    for user_key in ("question", "prompt", "instruction", "input"):
        for assistant_key in ("answer", "response", "output"):
            user = obj.get(user_key)
            assistant = obj.get(assistant_key)
            if isinstance(user, str) and isinstance(assistant, str):
                return user, assistant

    return None


def classify_task(user_text: str) -> str:
    text = user_text.lower()

    if CREATIVE_REQUEST_RE.search(text):
        return "creative"
    if DIALOGUE_REQUEST_RE.search(text) and "সংক্ষিপ্ত" not in text and "বিশ্লেষণ" not in text:
        return "dialogue"
    if (
        TRANSLATION_REQUEST_RE.search(text)
        and (LANGUAGE_HINT_RE.search(text) or '"' in user_text or "'" in user_text)
        and not any(hint in text for hint in GEOMETRY_TRANSLATION_HINTS)
    ):
        return "translation"
    if any(token in text for token in ["+", "-", "×", "÷", "%", "শতকরা", "সমীকরণ", "যোগ", "বিয়োগ", "বিয়োগ", "ভাগ", "গুণ", "ভগ্নাংশ", "গণিত"]):
        return "math"
    if any(token in text for token in ["তালিকা", "উল্লেখ", "পার্থক্য", "ধরন", "বৈশিষ্ট্য", "ধাপ", "পয়েন্ট"]):
        return "structured"
    if any(token in text for token in ADVICE_HINTS):
        return "advice"
    return "factual"


def count_pattern_hits(patterns: list[re.Pattern[str]], text: str) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def quality_score(user_text: str, assistant_text: str, task: str, source_name: str) -> float:
    ratio = bangla_char_ratio(assistant_text)
    words = word_count(assistant_text)
    sentences = sentence_count(assistant_text)

    score = 0.0
    score += ratio * 3.0

    if 18 <= words <= 160:
        score += 2.0
    elif 12 <= words < 18:
        score += 0.8
    elif 160 < words <= 260:
        score += 1.0
    else:
        score -= 1.0

    if 2 <= sentences <= 6:
        score += 1.0
    elif sentences == 1:
        score -= 0.4
    else:
        score -= 0.2

    if task == "creative":
        if "\n" in assistant_text:
            score += 0.8
        if words >= 24:
            score += 0.8
    elif task == "dialogue":
        if assistant_text.count(":") >= 2 or assistant_text.count("ঃ") >= 2:
            score += 1.0
    elif task == "translation":
        if words <= 35:
            score += 0.6
    elif task == "math":
        if re.search(r"[০-৯0-9]", assistant_text):
            score += 0.7
    elif task == "structured":
        if re.search(r"(^|\n)\s*[-*১-৯0-9]", assistant_text):
            score += 0.8

    if "gold" in source_name:
        score += 0.8
    if source_name.startswith("sft_style_"):
        score += 1.5
    if task in {"creative", "dialogue", "advice"}:
        if EMOJI_RE.search(assistant_text):
            score += 0.3
        if any(word in assistant_text for word in EMOTION_WORDS):
            score += 0.3

    score -= count_pattern_hits(TEXTBOOK_PATTERNS, assistant_text) * 2.5
    score -= count_pattern_hits(GENERIC_PATTERNS, assistant_text) * 0.7
    if task in {"creative", "dialogue"}:
        score -= count_pattern_hits(META_RESPONSE_PATTERNS, assistant_text) * 1.5

    if any(pattern.search(user_text) for pattern in TIME_SENSITIVE_PATTERNS):
        score -= 2.0

    if has_repeated_lines(assistant_text):
        score -= 1.5

    score -= repeated_ngram_ratio(assistant_text) * 3.0
    return score


def reject_reason(user_text: str, assistant_text: str, task: str) -> str | None:
    user_text = normalize_text(user_text)
    assistant_text = normalize_text(assistant_text)

    if not user_text or not assistant_text:
        return "empty"

    words = word_count(assistant_text)
    if words < 8:
        return "too_short"

    if len(assistant_text) > 4096:
        return "too_long"

    ratio = bangla_char_ratio(assistant_text)
    if ratio < 0.35:
        return "low_bangla"

    if count_pattern_hits(TEXTBOOK_PATTERNS, assistant_text) > 0:
        return "textbook_style"

    if any(pattern.search(user_text) for pattern in TIME_SENSITIVE_PATTERNS):
        return "time_sensitive"

    if has_repeated_lines(assistant_text) or repeated_ngram_ratio(assistant_text) > 0.20:
        return "repetitive"

    if task == "creative" and words < 24:
        return "weak_creative"

    if task == "dialogue" and assistant_text.count(":") + assistant_text.count("ঃ") < 2:
        return "weak_dialogue"

    if task in {"creative", "dialogue"} and count_pattern_hits(META_RESPONSE_PATTERNS, assistant_text) > 0:
        return "meta_response"

    if task == "advice" and words < 16:
        return "shallow_advice"

    if task == "structured" and not re.search(r"(^|\n)\s*[-*১-৯0-9]", assistant_text):
        return "unstructured_answer"

    if count_pattern_hits(GENERIC_PATTERNS, assistant_text) >= 3 and words < 28:
        return "generic_filler"

    return None


def iter_source_paths(data_dir: Path, include_gold: bool, include_generated: bool) -> list[Path]:
    paths = sorted(data_dir.glob("sft_*.jsonl"))
    if not include_generated:
        paths = [path for path in paths if path.name not in GENERATED_SOURCE_NAMES]
    if include_gold:
        paths.extend(sorted(data_dir.glob("kotha*_gold.jsonl")))
    return paths


def build_examples(
    data_dir: Path,
    include_gold: bool,
    include_generated: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = Counter()
    best_by_question: dict[str, dict[str, Any]] = {}

    for path in iter_source_paths(data_dir, include_gold, include_generated):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    continue

                pair = extract_pair(obj)
                if not pair:
                    stats["missing_pair"] += 1
                    continue

                user_text, assistant_text = pair
                task = classify_task(user_text)
                reason = reject_reason(user_text, assistant_text, task)
                if reason:
                    stats[reason] += 1
                    continue

                score = quality_score(user_text, assistant_text, task, path.name)
                if score < 2.5:
                    stats["low_score"] += 1
                    continue

                record = {
                    "messages": [
                        {"role": "user", "content": normalize_text(user_text)},
                        {"role": "assistant", "content": assistant_text.strip()},
                    ],
                    "_meta": {
                        "task": task,
                        "score": round(score, 4),
                        "source": path.name,
                    },
                }

                key = normalize_text(user_text).lower()
                existing = best_by_question.get(key)
                if existing is None or record["_meta"]["score"] > existing["_meta"]["score"]:
                    if existing is not None:
                        stats["dedup_replaced"] += 1
                    best_by_question[key] = record
                else:
                    stats["dedup_dropped"] += 1

    return list(best_by_question.values()), dict(stats)


def is_priority_source(example: dict[str, Any]) -> bool:
    source = example["_meta"]["source"]
    return any(source.startswith(prefix) for prefix in PRIORITY_SOURCE_PREFIXES)


def priority_source_target(source_name: str) -> int:
    for prefix, target in PRIORITY_SOURCE_MIN_COUNTS.items():
        if source_name.startswith(prefix):
            return target
    return 0


def select_balanced_examples(
    examples: list[dict[str, Any]],
    target_size: int,
    rng: random.Random,
    allow_oversample: bool = False,
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_task[example["_meta"]["task"]].append(example)

    for task_examples in by_task.values():
        task_examples.sort(key=lambda ex: ex["_meta"]["score"], reverse=True)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    task_selected = Counter()

    priority_examples = [
        example
        for example in sorted(examples, key=lambda ex: ex["_meta"]["score"], reverse=True)
        if is_priority_source(example)
    ]
    for example in priority_examples:
        if len(selected) >= target_size:
            break
        selected.append(example)
        selected_ids.add(id(example))
        task_selected[example["_meta"]["task"]] += 1

    if allow_oversample and priority_examples:
        priority_counts = Counter(example["_meta"]["source"] for example in selected)
        grouped_priority: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for example in priority_examples:
            grouped_priority[example["_meta"]["source"]].append(example)

        for source_name, source_examples in grouped_priority.items():
            target = priority_source_target(source_name)
            if target <= len(source_examples):
                continue
            idx = 0
            pool = source_examples[:]
            rng.shuffle(pool)
            while priority_counts[source_name] < min(target, target_size):
                base = pool[idx % len(pool)]
                dup = copy.deepcopy(base)
                dup["_meta"]["oversampled"] = True
                dup["_meta"]["oversample_index"] = idx
                selected.append(dup)
                priority_counts[source_name] += 1
                task_selected[dup["_meta"]["task"]] += 1
                idx += 1
                if idx % len(pool) == 0:
                    rng.shuffle(pool)
                if len(selected) >= target_size:
                    break

    for task, ratio in TASK_RATIOS.items():
        quota = max(int(target_size * ratio) - task_selected[task], 0)
        task_examples = by_task.get(task, [])
        remaining_examples = [example for example in task_examples if id(example) not in selected_ids]
        for example in remaining_examples[:quota]:
            selected.append(example)
            selected_ids.add(id(example))
            task_selected[task] += 1

        if allow_oversample and remaining_examples and quota > len(remaining_examples):
            pool = remaining_examples[:]
            rng.shuffle(pool)
            idx = 0
            while task_selected[task] < int(target_size * ratio):
                base = pool[idx % len(pool)]
                dup = copy.deepcopy(base)
                dup["_meta"]["oversampled"] = True
                dup["_meta"]["oversample_index"] = idx
                selected.append(dup)
                task_selected[task] += 1
                idx += 1
                if idx % len(pool) == 0:
                    rng.shuffle(pool)

    if len(selected) < target_size:
        remainder = sorted(
            (example for example in examples if id(example) not in selected_ids),
            key=lambda ex: ex["_meta"]["score"],
            reverse=True,
        )
        selected.extend(remainder[: target_size - len(selected)])

    return selected[:target_size]


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for example in examples:
            output = {"messages": example["messages"]}
            fh.write(json.dumps(output, ensure_ascii=False) + "\n")


def write_review_file(path: Path, examples: list[dict[str, Any]], sample_size: int, rng: random.Random) -> None:
    sample = rng.sample(examples, min(sample_size, len(examples)))
    with open(path, "w", encoding="utf-8") as fh:
        for idx, example in enumerate(sample, 1):
            user_text = example["messages"][0]["content"]
            assistant_text = example["messages"][1]["content"]
            meta = example["_meta"]
            fh.write(f"[{idx}] task={meta['task']} score={meta['score']:.2f} source={meta['source']}\n")
            fh.write(f"    Q: {user_text}\n")
            fh.write(f"    A: {assistant_text}\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a higher-quality Bangla SFT dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("finetune/data"))
    parser.add_argument("--target-size", type=int, default=50_000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", type=str, default="kotha_final")
    parser.add_argument("--review-sample-size", type=int, default=100)
    parser.add_argument(
        "--include-gold",
        action="store_true",
        help="Also include kotha*_gold.jsonl sources. Disabled by default for pure-curation runs.",
    )
    parser.add_argument(
        "--allow-oversample",
        action="store_true",
        help="Oversample rare tasks to hit the task-balance ratios.",
    )
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Include locally generated sources such as sft_gemini.jsonl.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    examples, filter_stats = build_examples(
        args.data_dir,
        include_gold=args.include_gold,
        include_generated=args.include_generated,
    )
    if not examples:
        raise SystemExit("No examples survived filtering.")

    selected = select_balanced_examples(
        sorted(examples, key=lambda ex: ex["_meta"]["score"], reverse=True),
        target_size=min(args.target_size, len(examples)),
        rng=rng,
        allow_oversample=args.allow_oversample,
    )
    rng.shuffle(selected)

    split_idx = int(len(selected) * (1 - args.val_ratio))
    train_examples = selected[:split_idx]
    val_examples = selected[split_idx:]

    train_path = args.data_dir / f"{args.output_prefix}_train.jsonl"
    val_path = args.data_dir / f"{args.output_prefix}_val.jsonl"
    report_path = args.data_dir / f"{args.output_prefix}_report.json"
    review_path = args.data_dir / "REVIEW_FINAL_DATASET.txt"

    write_jsonl(train_path, train_examples)
    write_jsonl(val_path, val_examples)
    write_review_file(review_path, train_examples, args.review_sample_size, rng)

    task_counts = Counter(example["_meta"]["task"] for example in selected)
    source_counts = Counter(example["_meta"]["source"] for example in selected)
    oversampled_total = sum(1 for example in selected if example["_meta"].get("oversampled"))
    emoji_rows = sum(
        1 for example in selected if EMOJI_RE.search(example["messages"][1]["content"])
    )
    emotion_rows = sum(
        1
        for example in selected
        if any(word in example["messages"][1]["content"] for word in EMOTION_WORDS)
    )
    report = {
        "selected_total": len(selected),
        "train_total": len(train_examples),
        "val_total": len(val_examples),
        "oversampled_total": oversampled_total,
        "emoji_rows": emoji_rows,
        "emotion_rows": emotion_rows,
        "task_counts": dict(sorted(task_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "filter_stats": filter_stats,
    }

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"Selected: {len(selected):,}")
    print(f"Train:    {len(train_examples):,} -> {train_path}")
    print(f"Val:      {len(val_examples):,} -> {val_path}")
    print(f"Report:   {report_path}")
    print(f"Review:   {review_path}")
    print(f"Emoji rows:   {emoji_rows:,}")
    print(f"Emotion rows: {emotion_rows:,}")
    print(f"Oversampled:  {oversampled_total:,}")
    print("\nTask mix:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task:12s} {count:>7,}")
    print("\nTop sources:")
    for source, count in source_counts.most_common(10):
        print(f"  {source:24s} {count:>7,}")


if __name__ == "__main__":
    main()
