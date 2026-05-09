#!/usr/bin/env python3
"""Generate gold Bengali SFT data using parallel Gemini sessions."""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY environment variable is required")
client = genai.Client(api_key=API_KEY)

SYSTEM = """আপনি একজন বাংলা ভাষার বিশেষজ্ঞ। উচ্চমানের বাংলা প্রশ্ন-উত্তর জোড়া তৈরি করুন।

গুরুত্বপূর্ণ নিয়ম:
- সরাসরি উত্তর দিন। "ধাপে ধাপে" বা "চলুন বিশ্লেষণ করি" বলবেন না
- "ধাপ ১", "ধাপ ২", "### ধাপ" ব্যবহার করবেন না
- মার্কডাউন হেডিং (###) ব্যবহার করবেন না
- উত্তর ২-৫ বাক্যের হবে, সংক্ষিপ্ত কিন্তু তথ্যপূর্ণ
- স্বাভাবিক কথোপকথনের মতো উত্তর দিন
- বাংলা সংখ্যা ব্যবহার করুন
- প্রতিটি উত্তরের শেষে একটি সম্পূর্ণ বাক্য থাকবে"""

TASKS = [
    ("bd_history", "বাংলাদেশের ইতিহাস: মুক্তিযুদ্ধ ১৯৭১, ভাষা আন্দোলন ১৯৫২, বঙ্গবন্ধু, জিয়াউর রহমান, স্বাধীনতা যুদ্ধ, পাকিস্তান আমল, ব্রিটিশ আমল"),
    ("bd_geo", "বাংলাদেশের ভূগোল ও প্রকৃতি: পদ্মা, মেঘনা, যমুনা, সুন্দরবন, কক্সবাজার, সিলেট, চট্টগ্রাম, রাঙামাটি, বিভাগ, জেলা"),
    ("bd_culture", "বাংলাদেশের সংস্কৃতি: পহেলা বৈশাখ, ঈদ, দুর্গাপূজা, বাংলা খাবার, ইলিশ, পিঠা, বাউল গান, রবীন্দ্রসংগীত, নজরুলগীতি"),
    ("science", "বিজ্ঞান: পদার্থবিজ্ঞান, রসায়ন, জীববিজ্ঞান, মহাকাশ, পরিবেশ, জলবায়ু পরিবর্তন, প্রাণী, উদ্ভিদ"),
    ("tech", "প্রযুক্তি: কম্পিউটার, ইন্টারনেট, মোবাইল, কৃত্রিম বুদ্ধিমত্তা, সফটওয়্যার, হার্ডওয়্যার, সামাজিক মাধ্যম"),
    ("math", "গণিত: যোগ, বিয়োগ, গুণ, ভাগ, শতকরা, ভগ্নাংশ, জ্যামিতি -- বাস্তব জীবনের উদাহরণ দিয়ে সমাধান"),
    ("literature", "সাহিত্য: রবীন্দ্রনাথ, নজরুল, জীবনানন্দ, শরৎচন্দ্র, হুমায়ূন আহমেদ, বাংলা কবিতা, উপন্যাস, ছোটগল্প"),
    ("daily", "দৈনন্দিন: রান্না, স্বাস্থ্য, ব্যায়াম, ভ্রমণ, শিক্ষা, চাকরি, কৃষি, পরিবার, সমাজ"),
    ("world", "বিশ্ব: বিশ্ব ইতিহাস, ভূগোল, মহাদেশ, দেশ, জাতিসংঘ, অলিম্পিক, বিখ্যাত বিজ্ঞানী, আবিষ্কার"),
    ("creative", "সৃজনশীল: কবিতা লিখুন, গল্প লিখুন, চিঠি লিখুন, সংলাপ লিখুন, প্রবাদের অর্থ বলুন -- আসল সৃজনশীল বিষয়বস্তু তৈরি করুন"),
]


def generate_batch(task_name, topic, batch_num):
    prompt = f"""বিষয়: {topic}

২৫টি বাংলা প্রশ্ন-উত্তর জোড়া তৈরি করুন। প্রতিটি আলাদা লাইনে JSON:
{{"messages": [{{"role": "user", "content": "প্রশ্ন"}}, {{"role": "assistant", "content": "উত্তর"}}]}}

মনে রাখুন: সরাসরি উত্তর দিন। কোনো "ধাপে ধাপে" বা ধাপ নম্বর নয়। ২-৫ বাক্য। শুধু JSON দিন।"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"system_instruction": SYSTEM, "temperature": 0.9, "max_output_tokens": 8192},
        )
        pairs = []
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    if "messages" in obj and len(obj["messages"]) >= 2:
                        r = obj["messages"][1]["content"]
                        if "ধাপে ধাপে" not in r and "ধাপ ১" not in r and "### " not in r:
                            pairs.append(obj)
                except json.JSONDecodeError:
                    pass
        return task_name, batch_num, pairs
    except Exception as e:
        return task_name, batch_num, []


def main():
    all_pairs = []
    total_batches = 0

    # 6 batches per task, 10 tasks = 60 batches, 5 workers
    jobs = []
    for task_name, topic in TASKS:
        for batch in range(15):
            jobs.append((task_name, topic, batch))

    print(f"Launching {len(jobs)} batches across {len(TASKS)} topics...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for task_name, topic, batch in jobs:
            f = executor.submit(generate_batch, task_name, topic, batch)
            futures[f] = (task_name, batch)
            time.sleep(1)  # stagger to avoid rate limit

        for future in as_completed(futures):
            task_name, batch = futures[future]
            _, _, pairs = future.result()
            all_pairs.extend(pairs)
            total_batches += 1
            print(f"  [{total_batches}/{len(jobs)}] {task_name} batch {batch}: +{len(pairs)} (total: {len(all_pairs)})")

    # Write output
    out = "finetune/data/kotha_gold.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(all_pairs)} gold pairs to {out}")


if __name__ == "__main__":
    main()
