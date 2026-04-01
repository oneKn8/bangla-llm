"""Base collector with resume support, state management, and JSONL output."""

import abc
import hashlib
import time
from pathlib import Path

import orjson
import requests
from tqdm import tqdm

from config import (
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF,
    STATE_DIR,
    USER_AGENT,
    ensure_dirs,
)


class BaseCollector(abc.ABC):
    """Abstract base for all data collectors.

    Provides:
    - Resumable state (save/load progress as JSON)
    - HTTP downloads with retry and resume (Range headers)
    - JSONL writing with buffered output
    - Document ID generation (SHA-256 of first 128 chars)
    """

    def __init__(self, name: str, output_dir: Path):
        self.name = name
        self.output_dir = output_dir
        self.state_file = STATE_DIR / f"{name}_state.json"
        self.state: dict = {}
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._doc_count = 0
        self._jsonl_file = None
        ensure_dirs()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Abstract ---

    @abc.abstractmethod
    def collect(self) -> None:
        """Run the collection process. Must be implemented by subclasses."""

    # --- State Management ---

    def load_state(self) -> dict:
        """Load saved state for resuming interrupted collection."""
        if self.state_file.exists():
            self.state = orjson.loads(self.state_file.read_bytes())
            print(f"[{self.name}] Resumed state: {len(self.state)} keys")
        else:
            self.state = {}
        return self.state

    def save_state(self) -> None:
        """Persist current state to disk."""
        self.state_file.write_bytes(
            orjson.dumps(self.state, option=orjson.OPT_INDENT_2)
        )

    # --- Document Helpers ---

    @staticmethod
    def make_id(text: str) -> str:
        """Generate document ID from SHA-256 of first 128 characters."""
        prefix = text[:128].encode("utf-8")
        return hashlib.sha256(prefix).hexdigest()

    def make_document(
        self,
        text: str,
        source: str,
        url: str = "",
        title: str = "",
        metadata: dict | None = None,
    ) -> dict:
        """Create a document dict matching the pipeline's JSONL schema."""
        return {
            "id": self.make_id(text),
            "text": text,
            "source": source,
            "url": url,
            "title": title,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "lang": "bn",
            "lang_score": 0.0,  # filled by lang_detect later
            "char_count": len(text),
            "metadata": metadata or {},
        }

    # --- JSONL Output ---

    def open_jsonl(self, filename: str) -> Path:
        """Open a JSONL file for writing. Returns the output path."""
        path = self.output_dir / filename
        self._jsonl_file = open(path, "ab")  # append mode for resume
        return path

    def write_document(self, doc: dict) -> None:
        """Write a single document to the open JSONL file."""
        if self._jsonl_file is None:
            raise RuntimeError("Call open_jsonl() before writing documents")
        self._jsonl_file.write(orjson.dumps(doc) + b"\n")
        self._doc_count += 1
        if self._doc_count % 1000 == 0:
            self._jsonl_file.flush()

    def write_documents(self, docs: list[dict]) -> None:
        """Write multiple documents to the open JSONL file."""
        for doc in docs:
            self.write_document(doc)

    def close_jsonl(self) -> None:
        """Flush and close the JSONL file."""
        if self._jsonl_file is not None:
            self._jsonl_file.flush()
            self._jsonl_file.close()
            self._jsonl_file = None

    @property
    def doc_count(self) -> int:
        return self._doc_count

    # --- HTTP Downloads ---

    def download_file(
        self,
        url: str,
        dest: Path,
        desc: str = "",
        resume: bool = True,
    ) -> Path:
        """Download a file with retry and optional resume via Range headers.

        Args:
            url: URL to download
            dest: Local destination path
            desc: Description for progress bar
            resume: If True, resume partial downloads
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        existing_size = dest.stat().st_size if (resume and dest.exists()) else 0
        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            print(f"[{self.name}] Resuming download from {existing_size} bytes")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()

                total = resp.headers.get("Content-Length")
                total = int(total) if total else None
                if total and existing_size > 0:
                    total += existing_size

                mode = "ab" if existing_size > 0 else "wb"
                with (
                    open(dest, mode) as f,
                    tqdm(
                        total=total,
                        initial=existing_size,
                        unit="B",
                        unit_scale=True,
                        desc=desc or dest.name,
                    ) as pbar,
                ):
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

                return dest

            except (requests.RequestException, OSError) as e:
                wait = RETRY_BACKOFF ** attempt
                print(
                    f"[{self.name}] Download attempt {attempt}/{MAX_RETRIES} "
                    f"failed: {e}. Retrying in {wait:.0f}s..."
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
                else:
                    raise

        return dest  # unreachable, but satisfies type checker

    def fetch_text(self, url: str) -> str | None:
        """Fetch a URL and return text content, or None on failure."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as e:
                wait = RETRY_BACKOFF ** attempt
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
                else:
                    print(f"[{self.name}] Failed to fetch {url}: {e}")
                    return None
        return None

    # --- Lifecycle ---

    def run(self) -> None:
        """Full lifecycle: load state -> collect -> save state -> close."""
        self.load_state()
        try:
            self.collect()
        finally:
            self.save_state()
            self.close_jsonl()
            print(f"[{self.name}] Done. {self._doc_count} documents written.")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close_jsonl()
        self.session.close()
