"""Seed-based web crawler for Bangla websites with language filtering."""

import asyncio
import re
from collections import deque
from urllib.parse import urljoin, urlparse

import aiohttp
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup
from readability import Document

from collectors.base import BaseCollector
from config import (
    RAW_WEB_CRAWL,
    SEEDS_DIR,
    USER_AGENT,
    WEB_CRAWL_MAX_DEPTH,
    WEB_CRAWL_MAX_PAGES,
    WEB_CRAWL_RPS,
    REQUEST_TIMEOUT,
)


class WebCrawlCollector(BaseCollector):
    """BFS web crawler starting from seed Bangla websites."""

    def __init__(self):
        super().__init__("web_crawl", RAW_WEB_CRAWL)
        self.seeds = self._load_seeds()
        self.visited: set[str] = set()

    def _load_seeds(self) -> list[str]:
        """Load seed URLs from seeds/bangla_websites.txt."""
        seed_file = SEEDS_DIR / "bangla_websites.txt"
        if not seed_file.exists():
            print(f"[{self.name}] No seed file found at {seed_file}")
            return []
        urls = []
        with open(seed_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
        return urls

    def collect(self) -> None:
        if not self.seeds:
            print(f"[{self.name}] No seeds to crawl. Add URLs to seeds/bangla_websites.txt")
            return
        self.visited = set(self.state.get("visited", []))
        asyncio.run(self._crawl_all())
        self.state["visited"] = list(self.visited)

    async def _crawl_all(self) -> None:
        limiter = AsyncLimiter(WEB_CRAWL_RPS, 1.0)
        connector = aiohttp.TCPConnector(limit=10)
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        ) as session:
            for seed in self.seeds:
                print(f"[{self.name}] Crawling seed: {seed}")
                await self._crawl_domain(session, seed, limiter)

    async def _crawl_domain(
        self,
        session: aiohttp.ClientSession,
        seed_url: str,
        limiter: AsyncLimiter,
    ) -> None:
        """BFS crawl a single domain starting from seed URL."""
        domain = urlparse(seed_url).netloc
        self.open_jsonl(f"{domain}.jsonl")

        queue: deque[tuple[str, int]] = deque()  # (url, depth)
        queue.append((seed_url, 0))
        domain_visited = 0

        while queue and domain_visited < WEB_CRAWL_MAX_PAGES:
            url, depth = queue.popleft()

            if url in self.visited:
                continue
            if depth > WEB_CRAWL_MAX_DEPTH:
                continue

            self.visited.add(url)
            async with limiter:
                html = await self._fetch_page(session, url)
                if not html:
                    continue

            # Extract article text
            text, title = self._extract_content(html)
            if text and len(text) >= 100:
                doc = self.make_document(
                    text=text,
                    source="web_crawl",
                    url=url,
                    title=title,
                    metadata={"domain": domain, "depth": depth},
                )
                self.write_document(doc)
                domain_visited += 1

            # Find links for BFS
            if depth < WEB_CRAWL_MAX_DEPTH:
                links = self._extract_links(html, url, domain)
                for link in links:
                    if link not in self.visited:
                        queue.append((link, depth + 1))

            if domain_visited % 100 == 0 and domain_visited > 0:
                print(f"[{self.name}] {domain}: {domain_visited} pages crawled")

        self.close_jsonl()
        print(f"[{self.name}] {domain}: finished with {domain_visited} pages")

    async def _fetch_page(
        self, session: aiohttp.ClientSession, url: str
    ) -> str | None:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    return None
                return await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError, UnicodeDecodeError):
            return None

    @staticmethod
    def _extract_content(html: str) -> tuple[str, str]:
        """Extract main content using readability-lxml."""
        try:
            doc = Document(html)
            title = doc.title()
            summary_html = doc.summary()
            soup = BeautifulSoup(summary_html, "lxml")
            text = soup.get_text(separator="\n\n", strip=True)
            # Clean up excessive whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text, title
        except Exception:
            return "", ""

    @staticmethod
    def _extract_links(html: str, base_url: str, domain: str) -> list[str]:
        """Extract same-domain links from HTML."""
        soup = BeautifulSoup(html, "lxml")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            # Same domain only, no fragments/query params
            if parsed.netloc == domain and parsed.scheme in ("http", "https"):
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if clean_url not in links:
                    links.append(clean_url)
        return links


if __name__ == "__main__":
    collector = WebCrawlCollector()
    collector.run()
