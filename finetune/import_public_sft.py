#!/usr/bin/env python3
"""Import selected public Bengali datasets into local SFT JSONL files.

This keeps the main curation pipeline simple: once converted, the new files are
picked up automatically by build_final_sft.py because they use the sft_*.jsonl
prefix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_bengalichat(output_path: Path) -> int:
    dataset = load_dataset("rishiraj/bengalichat")["train"]
    allowed_categories = {"Generation", "Brainstorm", "Chat", "Rewrite"}
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for record in dataset:
        if record.get("category") not in allowed_categories:
            continue

        messages = record.get("messages") or []
        user = ""
        assistant = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = normalize_text(message.get("content") or "")
            if role == "user" and not user:
                user = content
            elif role == "assistant" and not assistant:
                assistant = content

        if not user or not assistant:
            continue

        key = (user, assistant)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }
        )

    write_jsonl(output_path, rows)
    return len(rows)


def export_empathetic(output_path: Path) -> int:
    csv_path = hf_hub_download(
        "rasel-harun/BengaliEmpatheticConversationsCorpus",
        "BengaliEmpatheticConversationsCorpus.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(csv_path)

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for record in df.to_dict(orient="records"):
        title = normalize_text(str(record.get("Question-Title") or ""))
        question = normalize_text(str(record.get("Questions") or ""))
        answer = normalize_text(str(record.get("Answers") or ""))
        topic = normalize_text(str(record.get("Topics") or ""))

        if not question or not answer or question.lower() == "nan" or answer.lower() == "nan":
            continue

        prompt_parts = []
        if topic and topic.lower() != "nan":
            prompt_parts.append(f"বিষয়: {topic}")
        if title and title.lower() != "nan" and title not in question:
            prompt_parts.append(title)
        prompt_parts.append(question)
        prompt = "\n\n".join(prompt_parts)

        key = (prompt, answer)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            }
        )

    write_jsonl(output_path, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import public Bengali SFT datasets")
    parser.add_argument("--data-dir", type=Path, default=Path("finetune/data"))
    args = parser.parse_args()

    bengalichat_path = args.data_dir / "sft_public_bengalichat.jsonl"
    empathetic_path = args.data_dir / "sft_public_empathetic.jsonl"

    bengalichat_count = export_bengalichat(bengalichat_path)
    empathetic_count = export_empathetic(empathetic_path)

    print(f"Wrote {bengalichat_count:,} rows -> {bengalichat_path}")
    print(f"Wrote {empathetic_count:,} rows -> {empathetic_path}")


if __name__ == "__main__":
    main()
