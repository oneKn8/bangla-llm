"""Collect public domain Bangla literature from Internet Archive."""

import time

from collectors.base import BaseCollector
from config import ARCHIVE_ORG_RPS, RAW_LITERATURE

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata"
DOWNLOAD_URL = "https://archive.org/download"

# Search query for Bangla text on Internet Archive
SEARCH_PARAMS = {
    "q": 'language:(Bengali OR Bangla) AND mediatype:texts AND format:"DjVuTXT"',
    "fl[]": ["identifier", "title", "creator", "date", "language"],
    "sort[]": "downloads desc",
    "rows": 500,
    "page": 1,
    "output": "json",
}


class LiteratureCollector(BaseCollector):
    """Collect Bangla literature from Internet Archive."""

    def __init__(self):
        super().__init__("literature", RAW_LITERATURE)
        self.delay = 1.0 / ARCHIVE_ORG_RPS

    def collect(self) -> None:
        collected_ids = set(self.state.get("collected_ids", []))
        self.open_jsonl("literature.jsonl")

        # Search for Bangla texts
        items = self._search_items()
        print(f"[{self.name}] Found {len(items)} items on Internet Archive")

        for item in items:
            identifier = item.get("identifier", "")
            if identifier in collected_ids:
                continue

            text = self._download_text(identifier)
            if not text or len(text) < 200:
                continue

            title = item.get("title", "")
            creator = item.get("creator", "")

            doc = self.make_document(
                text=text,
                source="literature",
                url=f"https://archive.org/details/{identifier}",
                title=title,
                metadata={
                    "identifier": identifier,
                    "creator": creator,
                    "date": item.get("date", ""),
                },
            )
            self.write_document(doc)
            collected_ids.add(identifier)

            if self.doc_count % 10 == 0:
                self.state["collected_ids"] = list(collected_ids)
                self.save_state()
                print(f"[{self.name}] {self.doc_count} texts collected")

            time.sleep(self.delay)

        self.state["collected_ids"] = list(collected_ids)

    def _search_items(self) -> list[dict]:
        """Search Internet Archive for Bangla texts."""
        params = dict(SEARCH_PARAMS)
        # flatten fl[] for requests
        resp = self.session.get(
            SEARCH_URL,
            params={
                "q": params["q"],
                "fl[]": params["fl[]"],
                "sort[]": params["sort[]"],
                "rows": params["rows"],
                "page": params["page"],
                "output": params["output"],
            },
        )
        if resp.status_code != 200:
            print(f"[{self.name}] Search failed: {resp.status_code}")
            return []

        data = resp.json()
        return data.get("response", {}).get("docs", [])

    def _download_text(self, identifier: str) -> str | None:
        """Download the DjVuTXT or plain text file for an item."""
        time.sleep(self.delay)

        # Get metadata to find text files
        resp = self.session.get(f"{METADATA_URL}/{identifier}")
        if resp.status_code != 200:
            return None

        metadata = resp.json()
        files = metadata.get("files", [])

        # Prefer DjVuTXT, then plain text
        text_file = None
        for f in files:
            name = f.get("name", "")
            if name.endswith("_djvu.txt"):
                text_file = name
                break
        if not text_file:
            for f in files:
                name = f.get("name", "")
                if name.endswith(".txt") and f.get("format") == "Text":
                    text_file = name
                    break

        if not text_file:
            return None

        time.sleep(self.delay)
        url = f"{DOWNLOAD_URL}/{identifier}/{text_file}"
        return self.fetch_text(url)


if __name__ == "__main__":
    collector = LiteratureCollector()
    collector.run()
