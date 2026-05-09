#!/usr/bin/env python3
"""Generate high-quality Bangla creative SFT data with Gemini.

This focuses on the skills the current model is weak at:
- poem generation
- short story generation
- dialogue / scene writing
- letters / diary entries
- descriptive writing

Output format:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
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
        "name": "poem",
        "prompt": """শুধু কবিতা-কেন্দ্রিক জোড়া তৈরি করুন। ব্যবহারকারীর অনুরোধগুলো বৈচিত্র্যময় হবে:
- প্রেম, বৃষ্টি, শহর, গ্রাম, নদী, একাকিত্ব, আনন্দ, দেশপ্রেম, শিশুতোষ, হাস্যরস
- কখনও ৪ লাইনের, কখনও ৮-১২ লাইনের, কখনও মুক্তছন্দ, কখনও ছড়া
- উত্তর অবশ্যই কবিতাটি নিজে হবে; কবিতা লেখার নির্দেশনা বা ব্যাখ্যা নয়""",
    },
    {
        "name": "short_story",
        "prompt": """শুধু ছোটগল্প-কেন্দ্রিক জোড়া তৈরি করুন। ব্যবহারকারীর অনুরোধগুলো বৈচিত্র্যময় হবে:
- গ্রামবাংলা, শহুরে জীবন, রহস্য, নস্টালজিয়া, শিশুকেন্দ্রিক, বন্ধুত্ব, পারিবারিক সম্পর্ক, হালকা জাদুবাস্তবতা
- গল্পগুলো সংক্ষিপ্ত কিন্তু পূর্ণাঙ্গ হবে; শুরু, মোড়, এবং শেষ থাকবে
- উত্তর অবশ্যই গল্পটি নিজে হবে; গল্প কীভাবে লিখতে হয় তা নয়""",
    },
    {
        "name": "dialogue_scene",
        "prompt": """শুধু সংলাপ বা দৃশ্য-রচনা ভিত্তিক জোড়া তৈরি করুন। বৈচিত্র্যের জন্য:
- বন্ধু-বন্ধু, মা-মেয়ে, শিক্ষক-ছাত্র, দোকানদার-ক্রেতা, ডাক্তার-রোগী, দুই ভাইবোন
- কখনও নাট্যধর্মী দৃশ্য, কখনও বাস্তব কথোপকথন
- উত্তরে একাধিক টার্ন থাকবে এবং বক্তার নাম বা পরিচয় স্পষ্ট থাকবে""",
    },
    {
        "name": "letter_diary",
        "prompt": """শুধু চিঠি, ডায়েরি, ব্যক্তিগত নোট বা স্মৃতিচারণধর্মী জোড়া তৈরি করুন। বৈচিত্র্যের জন্য:
- বন্ধুকে চিঠি, মাকে চিঠি, চাকরির আবেদন নয় বরং ব্যক্তিগত আবেগপূর্ণ লেখা
- ডায়েরিতে প্রথম পুরুষের কণ্ঠ থাকবে
- স্কুল ভ্রমণ, বইমেলা, প্রথম ঢাকা-দর্শন, বর্ষার দিন, হারানো বন্ধুর স্মৃতি ইত্যাদি বিষয় রাখুন""",
    },
    {
        "name": "descriptive",
        "prompt": """শুধু বর্ণনামূলক ও সৃজনশীল অনুচ্ছেদ ভিত্তিক জোড়া তৈরি করুন। ব্যবহারকারীর অনুরোধে থাকবে:
- কোনো দৃশ্য, ঋতু, শহর, বাজার, নদীর ঘাট, রেলস্টেশন, গ্রামের ভোর, মেঘলা বিকেল
- চরিত্রের মনস্তত্ত্ব বা পরিবেশের অনুভূতি
- উত্তরে জীবন্ত, চিত্রধর্মী, সাহিত্যিক বাংলা থাকবে""",
    },
    {
        "name": "children_fable",
        "prompt": """শুধু শিশুতোষ গল্প, রূপকথা, নীতিগল্প বা কল্পনাপ্রবণ সৃজনশীল জোড়া তৈরি করুন। বৈচিত্র্যের জন্য:
- পশুপাখির গল্প, কথা বলা গাছ, ছোট্ট অভিযাত্রা, হারানো তারা, বুদ্ধিমান শিশু
- ভাষা সহজ কিন্তু রুচিসম্পন্ন হবে
- উত্তরে উপদেশধর্মী বক্তৃতা নয়, বরং গল্প বা কল্পনার লেখা নিজে থাকবে""",
    },
]

SYSTEM_PROMPT = """আপনি একজন উচ্চমানের বাংলা সৃজনশীল লেখক এবং ডেটা-কিউরেটর।

আপনার কাজ হলো এমন বাংলা instruction-response জোড়া তৈরি করা যা একটি বাংলা AI মডেলকে কবিতা, ছোটগল্প, সংলাপ, চিঠি, ডায়েরি এবং সৃজনশীল অনুচ্ছেদ লিখতে শেখাবে।

কঠোর নিয়ম:
- সম্পূর্ণ বাংলা লিপি ব্যবহার করুন
- উত্তর হবে আসল সৃজনশীল লেখা; কোনো ব্যাখ্যা, টিপস, ধাপে ধাপে নির্দেশনা, বা মেটা মন্তব্য নয়
- "ধাপে ধাপে", "চলুন লিখি", "নিচে দেওয়া হলো", "###", "প্রথমে" ইত্যাদি ব্যবহার করবেন না
- কবিতায় প্রয়োজনমতো লাইন ব্রেক থাকবে
- গল্পে চরিত্র, পরিবেশ, এবং একটি পূর্ণাঙ্গ ছোট আর্ক থাকবে
- সংলাপে একাধিক টার্ন থাকবে
- ভাষা যেন জীবন্ত, প্রাকৃতিক, সাহিত্যিক কিন্তু কৃত্রিম না শোনায়
- প্রশ্নগুলো একে অপরের থেকে ভিন্ন হবে
- একই টোন বা বাক্যরীতি বারবার ব্যবহার করবেন না
- শুধু JSON line দিন, অন্য কিছু নয়"""

BAD_PATTERNS = ["ধাপে ধাপে", "###", "নিচে দেওয়া হলো", "প্রথমে", "এই কবিতাটি", "এই গল্পটি"]


def word_count(text: str) -> int:
    return len(text.strip().split())


def valid_pair(category_name: str, pair: dict) -> bool:
    messages = pair.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False

    user = messages[0].get("content", "")
    assistant = messages[1].get("content", "")
    if not isinstance(user, str) or not isinstance(assistant, str):
        return False

    if any(pattern in assistant for pattern in BAD_PATTERNS):
        return False

    words = word_count(assistant)
    if category_name == "poem":
        return words >= 16 and assistant.count("\n") >= 2
    if category_name == "short_story":
        return words >= 70
    if category_name == "dialogue_scene":
        return assistant.count(":") + assistant.count("ঃ") >= 4
    if category_name == "letter_diary":
        return words >= 50
    if category_name == "descriptive":
        return words >= 45
    if category_name == "children_fable":
        return words >= 55
    return words >= 20


def generate_batch(client, category: dict, batch_size: int, existing_questions: set[str]) -> list[dict]:
    avoid = ""
    if existing_questions:
        sampled = random.sample(list(existing_questions), min(12, len(existing_questions)))
        avoid = "\n\nএই প্রশ্নগুলোর মতো প্রশ্ন এড়িয়ে নতুন ও আলাদা অনুরোধ তৈরি করুন:\n" + "\n".join(
            f"- {question}" for question in sampled
        )

    prompt = f"""{category['prompt']}

ঠিক {batch_size}টি JSON line তৈরি করুন।
প্রতিটি line-এর format হবে:
{{"messages": [{{"role": "user", "content": "ব্যবহারকারীর অনুরোধ"}}, {{"role": "assistant", "content": "উচ্চমানের সৃজনশীল উত্তর"}}]}}

গুরুত্বপূর্ণ:
- উত্তর হবে সরাসরি সৃজনশীল লেখা
- ব্যবহারকারীর অনুরোধে বৈচিত্র্য রাখুন
- প্রতিটি জোড়া স্বয়ংসম্পূর্ণ এবং training-ready হবে
- শুধু JSON lines দিন{avoid}"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 1.0,
            "max_output_tokens": 8192,
        },
    )

    pairs: list[dict] = []
    for line in response.text.strip().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if valid_pair(category["name"], obj):
            pairs.append(obj)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Bangla creative gold SFT data with Gemini")
    parser.add_argument("--output", type=Path, default=Path("finetune/data/kotha_creative_gold.jsonl"))
    parser.add_argument("--count", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sleep", type=float, default=4.0)
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    from google import genai

    client = genai.Client(api_key=api_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing_questions: set[str] = set()
    total_written = 0
    per_category_target = max(1, args.count // len(CATEGORIES))

    print(f"Target total: {args.count}")
    print(f"Per category target: {per_category_target}")
    print(f"Output: {args.output}")
    print()

    with open(args.output, "w", encoding="utf-8") as fh:
        for category in CATEGORIES:
            category_written = 0
            print(f"[{category['name']}] generating...")

            while category_written < per_category_target:
                try:
                    pairs = generate_batch(client, category, args.batch_size, existing_questions)
                except Exception as exc:
                    print(f"  API error: {exc}. Retrying in 8s...", file=sys.stderr)
                    time.sleep(8)
                    continue

                new_pairs = 0
                for pair in pairs:
                    user_text = pair["messages"][0]["content"].strip()
                    if user_text in existing_questions:
                        continue
                    existing_questions.add(user_text)
                    fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    total_written += 1
                    category_written += 1
                    new_pairs += 1
                    if category_written >= per_category_target:
                        break

                fh.flush()
                print(
                    f"  [{category['name']}] {category_written}/{per_category_target} "
                    f"(accepted +{new_pairs} from {len(pairs)})"
                )
                time.sleep(args.sleep)

    print(f"\nDone. Wrote {total_written} pairs to {args.output}")


if __name__ == "__main__":
    main()
