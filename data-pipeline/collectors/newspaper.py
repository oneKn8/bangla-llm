"""Async newspaper collector for Prothom Alo, Kaler Kantho, and Ittefaq.

Sitemap-driven collection with per-domain rate limiting, SQLite state for
resumability, and newspaper-specific HTML parsing.
"""

import asyncio
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup

from collectors.base import BaseCollector
from config import NEWSPAPERS, RAW_NEWSPAPERS, REQUEST_TIMEOUT, STATE_DIR, USER_AGENT


class NewspaperCollector(BaseCollector):
    """Collect articles from Bangla newspaper websites via sitemaps."""

    def __init__(self, names: list[str] | None = None):
        """
        Args:
            names: List of newspaper keys to collect. Defaults to all configured.
        """
        super().__init__("newspaper", RAW_NEWSPAPERS)
        self.targets = {
            k: v for k, v in NEWSPAPERS.items()
            if names is None or k in names
        }
        self.db_path = STATE_DIR / "newspaper.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for tracking collected URLs."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                url TEXT PRIMARY KEY,
                newspaper TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                collected_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_newspaper_status
            ON articles(newspaper, status)
        """)
        conn.commit()
        conn.close()

    def _url_exists(self, conn: sqlite3.Connection, url: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM articles WHERE url = ?", (url,)
        ).fetchone()
        return row is not None

    def _mark_done(self, conn: sqlite3.Connection, url: str, newspaper: str) -> None:
        conn.execute(
            """INSERT INTO articles (url, newspaper, status, collected_at)
               VALUES (?, ?, 'done', datetime('now'))
               ON CONFLICT(url) DO UPDATE SET status='done', collected_at=datetime('now')""",
            (url, newspaper),
        )
        conn.commit()

    def _mark_pending(self, conn: sqlite3.Connection, url: str, newspaper: str) -> None:
        conn.execute(
            """INSERT OR IGNORE INTO articles (url, newspaper, status)
               VALUES (?, ?, 'pending')""",
            (url, newspaper),
        )

    def collect(self) -> None:
        """Run async collection for all target newspapers."""
        asyncio.run(self._collect_all())

    async def _collect_all(self) -> None:
        connector = aiohttp.TCPConnector(limit=10)
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        ) as session:
            for name, cfg in self.targets.items():
                print(f"[{self.name}] Collecting: {name}")
                await self._collect_newspaper(session, name, cfg)

    async def _collect_newspaper(
        self,
        session: aiohttp.ClientSession,
        name: str,
        cfg: dict,
    ) -> None:
        limiter = AsyncLimiter(cfg["rps"], 1.0)
        jsonl_path = self.open_jsonl(f"{name}.jsonl")

        # Discover article URLs from sitemap
        urls = await self._discover_urls(session, cfg["sitemap_url"], limiter)
        print(f"[{self.name}] {name}: found {len(urls)} URLs in sitemap")

        # Filter already-collected URLs
        conn = sqlite3.connect(self.db_path)
        new_urls = []
        for url in urls:
            if not self._url_exists(conn, url):
                self._mark_pending(conn, url, name)
                new_urls.append(url)
        conn.commit()
        print(f"[{self.name}] {name}: {len(new_urls)} new URLs to collect")

        # Collect articles
        count = 0
        for url in new_urls:
            async with limiter:
                article = await self._fetch_article(session, url, name)
                if article:
                    self.write_document(article)
                    self._mark_done(conn, url, name)
                    count += 1
                    if count % 100 == 0:
                        print(f"[{self.name}] {name}: {count} articles collected")

        conn.close()
        self.close_jsonl()
        print(f"[{self.name}] {name}: finished with {count} articles")

    async def _discover_urls(
        self,
        session: aiohttp.ClientSession,
        sitemap_url: str,
        limiter: AsyncLimiter,
    ) -> list[str]:
        """Recursively parse sitemap XML to find article URLs."""
        urls = []
        sitemaps_to_process = [sitemap_url]

        while sitemaps_to_process:
            current = sitemaps_to_process.pop()
            async with limiter:
                try:
                    async with session.get(current) as resp:
                        if resp.status != 200:
                            continue
                        text = await resp.text()
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    continue

            try:
                root = ET.fromstring(text)
            except ET.ParseError:
                continue

            # Handle namespace
            ns = ""
            if root.tag.startswith("{"):
                ns = root.tag.split("}")[0] + "}"

            # Check if this is a sitemap index or a urlset
            for sitemap in root.findall(f"{ns}sitemap"):
                loc = sitemap.find(f"{ns}loc")
                if loc is not None and loc.text:
                    sitemaps_to_process.append(loc.text.strip())

            for url_elem in root.findall(f"{ns}url"):
                loc = url_elem.find(f"{ns}loc")
                if loc is not None and loc.text:
                    urls.append(loc.text.strip())

        return urls

    async def _fetch_article(
        self,
        session: aiohttp.ClientSession,
        url: str,
        newspaper: str,
    ) -> dict | None:
        """Fetch and parse a single newspaper article."""
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None

        text, title = self._extract_article(html, newspaper)
        if not text or len(text) < 100:
            return None

        return self.make_document(
            text=text,
            source=newspaper,
            url=url,
            title=title,
            metadata={"newspaper": newspaper},
        )

    def _extract_article(self, html: str, newspaper: str) -> tuple[str, str]:
        """Extract article body and title from HTML.

        Each newspaper has its own HTML structure, so we use
        newspaper-specific selectors with fallbacks.
        """
        soup = BeautifulSoup(html, "lxml")
        title = ""
        text = ""

        # Get title
        title_tag = soup.find("h1")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Newspaper-specific article body selectors
        selectors = {
            "prothomalo": [
                {"class_": re.compile(r"story-content|article-content")},
                {"itemprop": "articleBody"},
            ],
            "kalerkantho": [
                {"class_": re.compile(r"some-class-news-content|col-content")},
                {"id": "news-content"},
            ],
            "ittefaq": [
                {"class_": re.compile(r"content-details|news-content")},
                {"itemprop": "articleBody"},
            ],
        }

        article_div = None
        for selector in selectors.get(newspaper, []):
            article_div = soup.find("div", **selector)
            if article_div:
                break

        # Fallback: find largest <article> or <div> with most <p> tags
        if not article_div:
            article_div = soup.find("article")
        if not article_div:
            candidates = soup.find_all("div")
            best = None
            best_p_count = 0
            for div in candidates:
                p_count = len(div.find_all("p"))
                if p_count > best_p_count:
                    best_p_count = p_count
                    best = div
            if best and best_p_count >= 3:
                article_div = best

        if article_div:
            # Extract text from paragraphs
            paragraphs = article_div.find_all("p")
            text = "\n\n".join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )

        return text, title


if __name__ == "__main__":
    collector = NewspaperCollector()
    collector.run()
