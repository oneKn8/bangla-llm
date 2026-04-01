"""Collect Banglish (Bangla written in Latin script) from Reddit."""

import re
import time

import praw

from collectors.base import BaseCollector
from config import RAW_BANGLISH, REDDIT_DELAY, SEEDS_DIR

# Heuristic: common Bangla words written in Latin script
BANGLISH_KEYWORDS = {
    "ami", "tumi", "apni", "se", "tara", "amra", "tomra",
    "kemon", "achen", "acho", "bhalo", "valo", "kothay",
    "ki", "keno", "hobe", "hoye", "hoyeche", "korte",
    "bhai", "apa", "dada", "didi", "mama", "chacha",
    "boro", "choto", "shob", "sob", "kichu", "onek",
    "bangla", "bangladesh", "dhaka", "desh",
    "khub", "eto", "tai", "kintu", "ar", "ba",
    "ache", "nai", "nei", "jani", "boli", "bujhi",
    "dekhi", "shuni", "kori", "jai", "ashi",
    "khabar", "pani", "bari", "rasta", "manush",
    "bhalolage", "kharap", "shundor",
}

# Minimum keyword hits to classify as Banglish
MIN_KEYWORD_HITS = 3


class BanglishCollector(BaseCollector):
    """Collect Banglish text from Reddit using PRAW."""

    def __init__(self):
        super().__init__("banglish", RAW_BANGLISH)
        self.subreddits = self._load_subreddits()
        self.reddit = None

    def _load_subreddits(self) -> list[str]:
        """Load target subreddits from seeds/subreddits.txt."""
        seed_file = SEEDS_DIR / "subreddits.txt"
        if not seed_file.exists():
            return ["bangladesh", "bangla"]
        subs = []
        with open(seed_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    subs.append(line)
        return subs if subs else ["bangladesh", "bangla"]

    def _init_reddit(self) -> None:
        """Initialize PRAW Reddit client.

        Expects environment variables:
            REDDIT_CLIENT_ID
            REDDIT_CLIENT_SECRET
            REDDIT_USER_AGENT (optional, has default)
        """
        import os

        self.reddit = praw.Reddit(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ.get(
                "REDDIT_USER_AGENT",
                "BanglaLLM:v1.0 (by /u/bangla-llm-research)",
            ),
        )

    def collect(self) -> None:
        self._init_reddit()
        collected_ids = set(self.state.get("collected_ids", []))
        self.open_jsonl("banglish.jsonl")

        for sub_name in self.subreddits:
            print(f"[{self.name}] Scanning r/{sub_name}")
            count = 0

            try:
                subreddit = self.reddit.subreddit(sub_name)
                # Get submissions from multiple sort orders
                for sort_method in ["hot", "new", "top"]:
                    submissions = getattr(subreddit, sort_method)(limit=1000)
                    if sort_method == "top":
                        submissions = subreddit.top(time_filter="all", limit=1000)

                    for submission in submissions:
                        if submission.id in collected_ids:
                            continue

                        # Check submission selftext
                        if submission.selftext:
                            self._process_text(
                                text=submission.selftext,
                                post_id=submission.id,
                                url=f"https://reddit.com{submission.permalink}",
                                title=submission.title,
                                collected_ids=collected_ids,
                            )

                        # Check comments
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments.list():
                            if comment.id in collected_ids:
                                continue
                            if comment.body and comment.body != "[deleted]":
                                self._process_text(
                                    text=comment.body,
                                    post_id=comment.id,
                                    url=f"https://reddit.com{comment.permalink}",
                                    title="",
                                    collected_ids=collected_ids,
                                )

                        collected_ids.add(submission.id)
                        count += 1
                        if count % 50 == 0:
                            self.state["collected_ids"] = list(collected_ids)
                            self.save_state()
                            print(f"[{self.name}] r/{sub_name}: processed {count} posts")

                        time.sleep(REDDIT_DELAY)

            except Exception as e:
                print(f"[{self.name}] Error on r/{sub_name}: {e}")

        self.state["collected_ids"] = list(collected_ids)

    def _process_text(
        self,
        text: str,
        post_id: str,
        url: str,
        title: str,
        collected_ids: set,
    ) -> None:
        """Check if text is Banglish and write to JSONL if so."""
        if len(text) < 50:
            return
        if not self._is_banglish(text):
            return

        doc = self.make_document(
            text=text,
            source="banglish",
            url=url,
            title=title,
            metadata={"reddit_id": post_id, "script": "bn-Latn"},
        )
        self.write_document(doc)
        collected_ids.add(post_id)

    @staticmethod
    def _is_banglish(text: str) -> bool:
        """Heuristic check if text is Banglish (Bangla in Latin script).

        Criteria:
        - Mostly Latin characters (not Bengali script)
        - Contains enough known Banglish keywords
        - Not primarily English (checked by keyword ratio)
        """
        text_lower = text.lower()

        # Must be primarily Latin script
        latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha == 0:
            return False
        if latin_chars / total_alpha < 0.8:
            return False

        # Count Banglish keyword hits
        words = set(re.findall(r"[a-z]+", text_lower))
        hits = len(words & BANGLISH_KEYWORDS)

        return hits >= MIN_KEYWORD_HITS


if __name__ == "__main__":
    collector = BanglishCollector()
    collector.run()
