"""Stream Bangla text from HuggingFace datasets: OSCAR, Sangraha, CC-100."""

from datasets import load_dataset

from collectors.base import BaseCollector
from config import HF_DATASETS, RAW_HF_CORPUS


class HFCorpusCollector(BaseCollector):
    """Stream and collect Bengali subsets from HuggingFace datasets."""

    def __init__(self, sources: list[str] | None = None):
        """
        Args:
            sources: List of dataset keys to collect. Defaults to all configured.
                     Options: "oscar", "sangraha", "cc100"
        """
        super().__init__("hf_corpus", RAW_HF_CORPUS)
        self.targets = {
            k: v for k, v in HF_DATASETS.items()
            if sources is None or k in sources
        }

    def collect(self) -> None:
        for name, cfg in self.targets.items():
            if self.state.get(f"{name}_done"):
                print(f"[{self.name}] {name} already collected, skipping.")
                continue
            print(f"[{self.name}] Streaming {name}...")
            self._collect_dataset(name, cfg)
            self.state[f"{name}_done"] = True
            self.save_state()

    def _collect_dataset(self, name: str, cfg: dict) -> None:
        """Stream a single HuggingFace dataset and write to JSONL."""
        self.open_jsonl(f"{name}.jsonl")
        count = 0
        skipped = 0

        try:
            ds = load_dataset(
                cfg["path"],
                name=cfg.get("name"),
                split=cfg["split"],
                streaming=cfg.get("streaming", True),
            )

            # Figure out which field contains the text
            text_field = self._detect_text_field(ds)
            if not text_field:
                print(f"[{self.name}] {name}: could not detect text field, skipping")
                self.close_jsonl()
                return

            for row in ds:
                text = row.get(text_field, "")
                if not text or not isinstance(text, str):
                    skipped += 1
                    continue

                text = text.strip()
                if len(text) < 100:
                    skipped += 1
                    continue

                # Extract whatever metadata is available
                meta = {}
                for key in ("url", "source", "domain", "warc_id", "timestamp"):
                    if key in row and key != text_field:
                        meta[key] = str(row[key])

                doc = self.make_document(
                    text=text,
                    source=name,
                    url=meta.get("url", ""),
                    title="",
                    metadata=meta,
                )
                self.write_document(doc)
                count += 1

                if count % 10000 == 0:
                    print(f"[{self.name}] {name}: {count} docs collected, {skipped} skipped")

        except Exception as e:
            print(f"[{self.name}] {name}: error during streaming: {e}")

        self.close_jsonl()
        self.state[f"{name}_count"] = count
        print(f"[{self.name}] {name}: finished with {count} docs ({skipped} skipped)")

    @staticmethod
    def _detect_text_field(ds) -> str | None:
        """Auto-detect the text field name from dataset features.

        Common field names across OSCAR/Sangraha/CC-100:
        - "text" (most common)
        - "content"
        - "sentence"
        """
        # Try to get features from the dataset
        features = None
        try:
            features = ds.features
        except AttributeError:
            # Streaming dataset - try to peek at first item
            try:
                first = next(iter(ds))
                return next(
                    (k for k in ["text", "content", "sentence"] if k in first),
                    None,
                )
            except StopIteration:
                return None

        if features:
            for candidate in ["text", "content", "sentence"]:
                if candidate in features:
                    return candidate

        return None


if __name__ == "__main__":
    collector = HFCorpusCollector()
    collector.run()
