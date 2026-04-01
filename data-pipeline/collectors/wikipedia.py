"""Collect Bangla text from Wikipedia (bnwiki) and Wikisource (bnwikisource) dumps."""

import bz2
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from collectors.base import BaseCollector
from config import RAW_WIKIPEDIA, RAW_WIKISOURCE, WIKI_STUB_THRESHOLD

# Wikimedia dump URLs (latest)
BNWIKI_DUMP = "https://dumps.wikimedia.org/bnwiki/latest/bnwiki-latest-pages-articles.xml.bz2"
BNWIKISOURCE_DUMP = "https://dumps.wikimedia.org/bnwikisource/latest/bnwikisource-latest-pages-articles.xml.bz2"

# Patterns to filter
DISAMBIGUATION_PATTERNS = [
    "দ্ব্যর্থতা নিরসন",
    "দ্ব্যর্থতা",
    "বহুবিকল্প",
]
REDIRECT_PATTERN = re.compile(r"^#(পুনর্নির্দেশ|REDIRECT)", re.IGNORECASE)


class WikipediaCollector(BaseCollector):
    """Download and extract bnwiki + bnwikisource dumps to JSONL."""

    def __init__(self):
        super().__init__("wikipedia", RAW_WIKIPEDIA)
        self.wikisource_dir = RAW_WIKISOURCE
        self.wikisource_dir.mkdir(parents=True, exist_ok=True)

    def collect(self) -> None:
        self._collect_dump(
            url=BNWIKI_DUMP,
            source="wikipedia",
            output_dir=self.output_dir,
            jsonl_name="bnwiki.jsonl",
            state_key="bnwiki_done",
        )
        self._collect_dump(
            url=BNWIKISOURCE_DUMP,
            source="wikisource",
            output_dir=self.wikisource_dir,
            jsonl_name="bnwikisource.jsonl",
            state_key="bnwikisource_done",
        )

    def _collect_dump(
        self,
        url: str,
        source: str,
        output_dir: Path,
        jsonl_name: str,
        state_key: str,
    ) -> None:
        if self.state.get(state_key):
            print(f"[{self.name}] {source} already collected, skipping.")
            return

        # Download dump
        dump_file = output_dir / url.split("/")[-1]
        print(f"[{self.name}] Downloading {source} dump...")
        self.download_file(url, dump_file, desc=f"{source} dump")

        # Extract with wikiextractor
        print(f"[{self.name}] Extracting {source} articles...")
        extracted_dir = self._run_wikiextractor(dump_file)

        # Convert to JSONL
        self.open_jsonl(jsonl_name) if source == "wikipedia" else self._open_wikisource_jsonl(jsonl_name)
        self._process_extracted(extracted_dir, source)
        self.close_jsonl()

        self.state[state_key] = True
        self.save_state()
        print(f"[{self.name}] {source}: {self.doc_count} documents extracted.")

    def _open_wikisource_jsonl(self, filename: str) -> Path:
        """Open JSONL file in the wikisource output directory."""
        path = self.wikisource_dir / filename
        self._jsonl_file = open(path, "ab")
        return path

    def _run_wikiextractor(self, dump_path: Path) -> Path:
        """Run wikiextractor on a Wikipedia XML dump."""
        output_dir = dump_path.parent / "extracted"
        output_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable, "-m", "wikiextractor.WikiExtractor",
            str(dump_path),
            "--output", str(output_dir),
            "--json",
            "--no-templates",
            "--processes", "4",
            "--min_text_length", str(WIKI_STUB_THRESHOLD),
        ]

        print(f"[{self.name}] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return output_dir

    def _process_extracted(self, extracted_dir: Path, source: str) -> None:
        """Read wikiextractor JSON output and convert to our JSONL schema."""
        import orjson

        for json_file in sorted(extracted_dir.rglob("wiki_*")):
            with open(json_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        article = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        continue

                    text = article.get("text", "").strip()
                    title = article.get("title", "")
                    url = article.get("url", "")

                    if not text or len(text) < WIKI_STUB_THRESHOLD:
                        continue
                    if self._is_disambiguation(title, text):
                        continue
                    if REDIRECT_PATTERN.match(text):
                        continue

                    doc = self.make_document(
                        text=text,
                        source=source,
                        url=url,
                        title=title,
                        metadata={"wiki_id": article.get("id", "")},
                    )
                    self.write_document(doc)

    @staticmethod
    def _is_disambiguation(title: str, text: str) -> bool:
        """Check if article is a disambiguation page."""
        combined = title + " " + text[:500]
        return any(p in combined for p in DISAMBIGUATION_PATTERNS)


if __name__ == "__main__":
    collector = WikipediaCollector()
    collector.run()
