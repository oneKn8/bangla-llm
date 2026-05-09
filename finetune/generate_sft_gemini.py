#!/usr/bin/env python3
"""Generate native Bengali SFT data using Gemini 3 Flash.

Generates high-quality Bengali instruction-response pairs across diverse
categories. Each batch request asks Gemini to produce 25 pairs in JSONL format.

Usage:
    export GEMINI_API_KEY="your-key-here"
    python generate_sft_gemini.py --output finetune/data/sft_gemini.jsonl --count 10000

Requires: google-genai
    pip install google-genai
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

CATEGORIES = [
    {
        "name": "বাংলাদেশ জ্ঞান",
        "prompt": """বাংলাদেশ সম্পর্কিত প্রশ্ন-উত্তর তৈরি করুন। বিষয়: ইতিহাস (মুক্তিযুদ্ধ, ভাষা আন্দোলন, প্রাচীন বাংলা),
ভূগোল (নদী, জেলা, বিভাগ, প্রাকৃতিক সম্পদ), সংস্কৃতি (উৎসব, খাবার, সংগীত, সাহিত্য),
অর্থনীতি (গার্মেন্টস, কৃষি, রেমিট্যান্স), বিখ্যাত ব্যক্তিত্ব, রাজনীতি, শিক্ষা ব্যবস্থা।""",
    },
    {
        "name": "বিজ্ঞান ও প্রযুক্তি",
        "prompt": """বিজ্ঞান ও প্রযুক্তি বিষয়ে প্রশ্ন-উত্তর তৈরি করুন। বিষয়: পদার্থবিজ্ঞান, রসায়ন, জীববিজ্ঞান,
মহাকাশবিজ্ঞান, পরিবেশবিজ্ঞান, কম্পিউটার বিজ্ঞান, কৃত্রিম বুদ্ধিমত্তা, ইন্টারনেট, মোবাইল প্রযুক্তি,
চিকিৎসাবিজ্ঞান, জেনেটিক্স, ন্যানোটেকনোলজি।""",
    },
    {
        "name": "গণিত ও যুক্তি",
        "prompt": """গণিত ও যুক্তি বিষয়ক প্রশ্ন-উত্তর তৈরি করুন। বিষয়: পাটিগণিত, বীজগণিত, জ্যামিতি,
শব্দসমস্যা (বাস্তব জীবনের প্রেক্ষাপটে -- বাজার, চাষ, স্কুল), ধাঁধা, লজিক পাজল,
প্যাটার্ন চিনুন, তুলনা ও বিশ্লেষণ। ধাপে ধাপে সমাধান দিন। বাংলা সংখ্যা (১, ২, ৩) ব্যবহার করুন।""",
    },
    {
        "name": "সৃজনশীল লেখা",
        "prompt": """সৃজনশীল লেখার নির্দেশনা ও উত্তর তৈরি করুন। বিষয়: ছোটগল্প লেখা, কবিতা রচনা (প্রকৃতি, দেশপ্রেম,
ভালোবাসা, ঋতু), চিঠি লেখা (আবেদন, অভিযোগ, ব্যক্তিগত), রচনা/প্রবন্ধ, সংলাপ রচনা
(শিক্ষক-ছাত্র, বন্ধু-বন্ধু, ডাক্তার-রোগী), প্রবাদ-প্রবচনের অর্থ ও ব্যবহার, বাংলা ব্যাকরণ।""",
    },
    {
        "name": "দৈনন্দিন জীবন",
        "prompt": """দৈনন্দিন জীবনের ব্যবহারিক প্রশ্ন-উত্তর তৈরি করুন। বিষয়: রান্নার রেসিপি (বাংলাদেশি খাবার --
বিরিয়ানি, ইলিশ, পিঠা, ভর্তা, ডাল), স্বাস্থ্য পরামর্শ, ব্যায়াম, মানসিক স্বাস্থ্য,
চাকরি ও ক্যারিয়ার, ভ্রমণ গাইড (কক্সবাজার, সুন্দরবন, সিলেট, রাঙামাটি),
বাগান করা, কৃষি, আইনি জ্ঞান, আর্থিক পরামর্শ।""",
    },
    {
        "name": "বিশ্ব জ্ঞান",
        "prompt": """বিশ্ব সম্পর্কিত সাধারণ জ্ঞানের প্রশ্ন-উত্তর তৈরি করুন। বিষয়: বিশ্ব ইতিহাস, ভূগোল (মহাদেশ, দেশ,
মহাসাগর), বিশ্ব সংস্কৃতি, আন্তর্জাতিক সংস্থা (জাতিসংঘ, WHO, IMF), বিশ্ব অর্থনীতি,
খেলাধুলা (ক্রিকেট, ফুটবল, অলিম্পিক), বিখ্যাত বিজ্ঞানী ও আবিষ্কার, দর্শন ও নৈতিকতা।""",
    },
]

SYSTEM_PROMPT = """আপনি একজন বাংলা ভাষার বিশেষজ্ঞ। আপনার কাজ হলো উচ্চমানের বাংলা প্রশ্ন-উত্তর জোড়া তৈরি করা
যা একটি AI মডেলকে প্রশিক্ষণ দিতে ব্যবহার করা হবে।

নিয়ম:
- সম্পূর্ণ বাংলায় লিখুন (বাংলা লিপি ব্যবহার করুন)
- উত্তর বিস্তারিত হবে (কমপক্ষে ২-৫ বাক্য)
- সঠিক তথ্য, সংখ্যা ও উদাহরণ দিন
- বাংলা সংখ্যা ব্যবহার করুন (১, ২, ৩)
- প্রশ্নের ধরন বৈচিত্র্যময় রাখুন (কী?, কেন?, কীভাবে?, ব্যাখ্যা করুন, পার্থক্য কী?, তুলনা করুন)
- প্রতিটি জোড়া অনন্য হবে -- পুনরাবৃত্তি করবেন না
- অনুবাদের মতো শোনাবে না, স্বাভাবিক বাংলা হবে"""


def generate_batch(
    client,
    category: dict,
    batch_size: int = 25,
    existing_questions: set | None = None,
) -> list[dict]:
    """Generate a batch of SFT pairs using Gemini."""

    avoid_text = ""
    if existing_questions and len(existing_questions) > 5:
        samples = random.sample(list(existing_questions), min(10, len(existing_questions)))
        avoid_text = "\n\nএই প্রশ্নগুলো আগেই তৈরি হয়েছে, এগুলো এড়িয়ে নতুন প্রশ্ন করুন:\n" + "\n".join(
            f"- {q}" for q in samples
        )

    user_prompt = f"""{category['prompt']}

ঠিক {batch_size}টি প্রশ্ন-উত্তর জোড়া তৈরি করুন। প্রতিটি জোড়া একটি আলাদা লাইনে JSON ফরম্যাটে দিন:
{{"messages": [{{"role": "user", "content": "প্রশ্ন"}}, {{"role": "assistant", "content": "উত্তর"}}]}}

শুধুমাত্র JSON লাইনগুলো দিন, অন্য কিছু নয়।{avoid_text}"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.9,
            "max_output_tokens": 8192,
        },
    )

    pairs = []
    for line in response.text.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if "messages" in obj and len(obj["messages"]) >= 2:
                pairs.append(obj)
        except json.JSONDecodeError:
            continue

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Bengali SFT data with Gemini")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--count", type=int, default=10000, help="Target number of pairs")
    parser.add_argument("--batch-size", type=int, default=25, help="Pairs per API call")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API key (or set GEMINI_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    existing_questions: set[str] = set()
    pairs_per_category = args.count // len(CATEGORIES)

    print(f"Target: {args.count} pairs ({pairs_per_category} per category)")
    print(f"Categories: {len(CATEGORIES)}")
    print(f"Batch size: {args.batch_size}")
    print()

    with open(args.output, "w", encoding="utf-8") as fh:
        for cat in CATEGORIES:
            cat_count = 0
            cat_name = cat["name"]
            print(f"[{cat_name}] Generating {pairs_per_category} pairs...")

            while cat_count < pairs_per_category:
                try:
                    pairs = generate_batch(
                        client, cat, args.batch_size, existing_questions
                    )
                except Exception as e:
                    print(f"  API error: {e}. Retrying in 5s...", file=sys.stderr)
                    time.sleep(5)
                    continue

                for pair in pairs:
                    q = pair["messages"][0]["content"]
                    if q in existing_questions:
                        continue
                    existing_questions.add(q)
                    fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    cat_count += 1
                    total += 1

                print(f"  [{cat_name}] {cat_count}/{pairs_per_category} (+{len(pairs)} this batch)")
                fh.flush()

                # Rate limit: Gemini Flash free tier = 15 RPM
                time.sleep(4)

    print(f"\nDone. Wrote {total} pairs to {args.output}")


if __name__ == "__main__":
    main()
