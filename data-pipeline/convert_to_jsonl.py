"""Convert downloaded datasets (Parquet, CC-100 raw text) to pipeline JSONL format.

Handles:
  - HuggingFace Parquet files (CulturaX, Sangraha) -> JSONL
  - CC-100 raw text (.txt.xz, blank-line separated docs) -> JSONL

Usage:
    python convert_to_jsonl.py parquet <dir> <output.jsonl> <source_name>
    python convert_to_jsonl.py cc100 <input.txt.xz> <output.jsonl>
"""

import argparse
import hashlib
import time
from pathlib import Path

import orjson


def _make_doc(text: str, source: str, url: str = "", title: str = "") -> dict:
    """Create a document dict matching the pipeline JSONL schema."""
    text = text.strip()
    return {
        "id": hashlib.sha256(text[:128].encode()).hexdigest(),
        "text": text,
        "source": source,
        "url": url,
        "title": title,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lang": "bn",
        "lang_score": 0.0,
        "char_count": len(text),
        "metadata": {},
    }


def convert_parquet(parquet_dir: str, output_path: str, source: str, text_field: str = "text"):
    """Convert all Parquet files in a directory tree to a single JSONL file.

    Reads in batches of 10K rows for memory efficiency.
    Auto-detects text field name if the specified one isn't found.
    """
    import pyarrow.parquet as pq

    parquet_files = sorted(Path(parquet_dir).rglob("*.parquet"))
    if not parquet_files:
        print(f"[convert] No .parquet files found in {parquet_dir}")
        return

    print(f"[convert] {len(parquet_files)} Parquet files -> {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0

    with open(output_path, "wb") as fout:
        for i, pf_path in enumerate(parquet_files):
            pf = pq.ParquetFile(pf_path)
            schema_names = pf.schema_arrow.names

            # Auto-detect text field
            tf = text_field
            if tf not in schema_names:
                for candidate in ["text", "content", "sentence"]:
                    if candidate in schema_names:
                        tf = candidate
                        break
                else:
                    print(f"  [{pf_path.name}] No text field found in {schema_names}, skipping")
                    continue

            # Determine available metadata columns
            meta_cols = [c for c in ["url", "source", "timestamp"] if c in schema_names]
            read_cols = [tf] + meta_cols

            for batch in pf.iter_batches(batch_size=10_000, columns=read_cols):
                texts = batch.column(tf)
                urls = batch.column("url") if "url" in meta_cols else None

                for j in range(len(batch)):
                    text = texts[j].as_py()
                    if not text or len(text.strip()) < 100:
                        skipped += 1
                        continue

                    url = (urls[j].as_py() or "") if urls else ""
                    doc = _make_doc(text=text, source=source, url=url)
                    fout.write(orjson.dumps(doc) + b"\n")
                    count += 1

            print(f"  [{i + 1}/{len(parquet_files)}] {pf_path.name} -- running total: {count:,} docs")

    print(f"[convert] Done: {count:,} docs written, {skipped:,} skipped (< 100 chars)")


def convert_cc100(input_path: str, output_path: str):
    """Convert CC-100 raw text file to pipeline JSONL.

    CC-100 format: one sentence per line, blank lines separate documents.
    Supports .xz compressed input.
    """
    import lzma

    input_path = Path(input_path)
    print(f"[convert] CC-100: {input_path} -> {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    opener = lzma.open if input_path.suffix == ".xz" else open
    count = 0
    skipped = 0
    lines_buf: list[str] = []

    with opener(str(input_path), "rt", encoding="utf-8") as fin, \
         open(output_path, "wb") as fout:

        for line in fin:
            line = line.rstrip("\n")

            if not line:
                if lines_buf:
                    text = "\n".join(lines_buf)
                    if len(text.strip()) >= 100:
                        doc = _make_doc(text=text, source="cc100")
                        fout.write(orjson.dumps(doc) + b"\n")
                        count += 1
                        if count % 100_000 == 0:
                            print(f"  {count:,} docs converted")
                    else:
                        skipped += 1
                    lines_buf = []
            else:
                lines_buf.append(line)

        # Flush last document
        if lines_buf:
            text = "\n".join(lines_buf)
            if len(text.strip()) >= 100:
                doc = _make_doc(text=text, source="cc100")
                fout.write(orjson.dumps(doc) + b"\n")
                count += 1
            else:
                skipped += 1

    print(f"[convert] Done: {count:,} docs written, {skipped:,} skipped (< 100 chars)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert datasets to pipeline JSONL format")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_pq = sub.add_parser("parquet", help="Convert Parquet files to JSONL")
    p_pq.add_argument("dir", help="Directory containing .parquet files (searched recursively)")
    p_pq.add_argument("output", help="Output JSONL path")
    p_pq.add_argument("source", help="Source name for docs (e.g. culturax, sangraha)")
    p_pq.add_argument("--text-field", default="text", help="Name of the text column")

    p_cc = sub.add_parser("cc100", help="Convert CC-100 .txt.xz to JSONL")
    p_cc.add_argument("input", help="Input .txt or .txt.xz file")
    p_cc.add_argument("output", help="Output JSONL path")

    args = parser.parse_args()

    if args.mode == "parquet":
        convert_parquet(args.dir, args.output, args.source, args.text_field)
    elif args.mode == "cc100":
        convert_cc100(args.input, args.output)
