"""Export processed data as compressed JSONL chunks with manifest."""

import gzip
import hashlib
import time
from pathlib import Path

import orjson
from tqdm import tqdm

from config import EXPORT_CHUNK_SIZE_MB, EXPORT_COMPRESSION_LEVEL, FINAL_DIR, PROCESSED_DIR, ensure_dirs


def export_chunks(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    chunk_size_mb: int = EXPORT_CHUNK_SIZE_MB,
) -> Path:
    """Split processed JSONL into compressed chunks with a manifest.

    Args:
        input_dir: Directory with processed JSONL files
        output_dir: Where to write compressed chunks
        chunk_size_mb: Target uncompressed size per chunk in MB

    Returns:
        Path to the manifest file
    """
    ensure_dirs()
    src = input_dir or PROCESSED_DIR
    dst = output_dir or FINAL_DIR
    dst.mkdir(parents=True, exist_ok=True)

    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    chunks = []
    current_chunk = 1
    current_size = 0
    current_docs = 0
    total_docs = 0

    gz_path = dst / f"bangla_{current_chunk:03d}.jsonl.gz"
    gz_file = gzip.open(gz_path, "wb", compresslevel=EXPORT_COMPRESSION_LEVEL)

    # Collect all processed JSONL files
    jsonl_files = sorted(src.glob("*.jsonl"))
    if not jsonl_files:
        print("[export] No processed JSONL files found.")
        gz_file.close()
        gz_path.unlink(missing_ok=True)
        return dst / "manifest.json"

    print(f"[export] Exporting {len(jsonl_files)} files into {chunk_size_mb}MB chunks...")

    for jsonl_file in jsonl_files:
        with open(jsonl_file, "rb") as f:
            for line in tqdm(f, desc=f"Exporting {jsonl_file.name}", unit=" docs"):
                line = line.strip()
                if not line:
                    continue

                data = line + b"\n"
                current_size += len(data)
                current_docs += 1
                total_docs += 1
                gz_file.write(data)

                # Check if chunk is full
                if current_size >= chunk_size_bytes:
                    gz_file.close()
                    chunk_hash = _file_sha256(gz_path)
                    compressed_size = gz_path.stat().st_size
                    chunks.append({
                        "filename": gz_path.name,
                        "docs": current_docs,
                        "uncompressed_bytes": current_size,
                        "compressed_bytes": compressed_size,
                        "sha256": chunk_hash,
                    })
                    print(
                        f"[export] Chunk {current_chunk}: {current_docs} docs, "
                        f"{compressed_size / 1024 / 1024:.1f}MB compressed"
                    )

                    # Start new chunk
                    current_chunk += 1
                    current_size = 0
                    current_docs = 0
                    gz_path = dst / f"bangla_{current_chunk:03d}.jsonl.gz"
                    gz_file = gzip.open(
                        gz_path, "wb", compresslevel=EXPORT_COMPRESSION_LEVEL
                    )

    # Close final chunk
    gz_file.close()
    if current_docs > 0:
        chunk_hash = _file_sha256(gz_path)
        compressed_size = gz_path.stat().st_size
        chunks.append({
            "filename": gz_path.name,
            "docs": current_docs,
            "uncompressed_bytes": current_size,
            "compressed_bytes": compressed_size,
            "sha256": chunk_hash,
        })
        print(
            f"[export] Chunk {current_chunk}: {current_docs} docs, "
            f"{compressed_size / 1024 / 1024:.1f}MB compressed"
        )
    else:
        gz_path.unlink(missing_ok=True)

    # Write manifest
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_docs": total_docs,
        "total_chunks": len(chunks),
        "chunk_size_target_mb": chunk_size_mb,
        "chunks": chunks,
    }
    manifest_path = dst / "manifest.json"
    manifest_path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

    print(f"\n[export] Done: {total_docs} docs in {len(chunks)} chunks")
    print(f"[export] Manifest: {manifest_path}")
    return manifest_path


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    export_chunks()
