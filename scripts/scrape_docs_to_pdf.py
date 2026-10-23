#!/usr/bin/env python3
"""
Crawl a documentation site and save every page as a PDF — the automated
equivalent of visiting each page and doing Ctrl+P -> Save as PDF, for sites
too large to do by hand.

Uses a real headless browser (Playwright) for BOTH discovering links and
exporting PDFs — not a plain HTTP request. Most modern documentation sites
(Next.js, Docusaurus, Mintlify, VitePress, etc.) render their navigation
sidebar with client-side JavaScript, so a plain `requests`/`aiohttp` GET
only sees the initial HTML shell with no real links in it. Rendering each
page in an actual browser and reading the DOM *after* JS has run is what
makes the crawl actually find more than the start page.

Only use this on sites you own or have explicit permission to scrape. It
stays within the same domain + a path prefix (e.g. only /docs) and crawls
with bounded concurrency rather than hammering the server.

Usage:
    python scripts/scrape_docs_to_pdf.py \\
        --url https://eduversepak.cloud/docs \\
        --prefix /docs \\
        --out data/companies/eduversepak/raw_pdfs \\
        --max-pages 200

Requires (kept separate from the main project — see requirements-scraping.txt):
    pip install -r requirements-scraping.txt
    playwright install chromium
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


def _proxy_config() -> dict | None:
    """Build a Playwright proxy config from standard env vars (if set)."""
    import os

    raw = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY")
    if not raw:
        return None
    parsed = urlparse(raw)
    cfg: dict = {"server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"}
    if parsed.username:
        cfg["username"] = parsed.username
    if parsed.password:
        cfg["password"] = parsed.password
    return cfg


def sanitize_filename(url: str) -> str:
    path = urlparse(url).path.strip("/") or "index"
    safe = re.sub(r"[^a-zA-Z0-9\-_/]", "_", path).replace("/", "__")
    return f"{safe}.pdf"


def extract_links(html: str, base_url: str, domain: str, prefix: str) -> list[str]:
    """Pulls same-domain, prefix-matching links out of already-rendered HTML."""
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        link = urljoin(base_url, a["href"]).split("#")[0].rstrip("/")
        parsed = urlparse(link)
        if parsed.netloc != domain:
            continue
        if prefix and not parsed.path.startswith(prefix):
            continue
        links.append(link)
    return links


async def crawl_and_save(start_url: str, prefix: str, out_dir: Path, max_pages: int, concurrency: int) -> list[str]:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print(
            "ERROR: playwright is not installed. Run:\n"
            "  pip install -r requirements-scraping.txt\n"
            "  playwright install chromium",
            file=sys.stderr,
        )
        raise SystemExit(1)

    domain = urlparse(start_url).netloc
    out_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = {start_url}
    queue: list[str] = [start_url]
    saved: list[str] = []

    async with async_playwright() as p:
        launch_kwargs: dict = {}
        proxy = _proxy_config()
        if proxy:
            launch_kwargs["proxy"] = proxy
            print("  (using egress proxy from environment)")
        browser = await p.chromium.launch(**launch_kwargs)
        semaphore = asyncio.Semaphore(concurrency)

        async def visit(url: str) -> list[str]:
            async with semaphore:
                page = await browser.new_page()
                new_links: list[str] = []
                try:
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                    # Give client-side-rendered nav menus a beat to finish mounting
                    # on sites that fetch their sidebar data after initial load.
                    await page.wait_for_timeout(500)

                    html = await page.content()
                    dest = out_dir / sanitize_filename(url)
                    await page.pdf(path=str(dest), format="A4", print_background=True)
                    saved.append(url)
                    print(f"  saved [{len(saved)}]: {dest.name}  <-  {url}")

                    new_links = extract_links(html, url, domain, prefix)
                except Exception as exc:  # noqa: BLE001 — one bad page must not kill the whole crawl
                    print(f"  FAILED: {url} ({exc})", file=sys.stderr)
                finally:
                    await page.close()
                return new_links

        while queue and len(saved) < max_pages:
            batch = queue[: max(concurrency, 1)]
            queue = queue[len(batch):]

            results = await asyncio.gather(*(visit(u) for u in batch))

            for links in results:
                for link in links:
                    if link not in seen:
                        seen.add(link)
                        queue.append(link)

        await browser.close()

    return saved


async def main(start_url: str, prefix: str, out_dir: Path, max_pages: int, concurrency: int) -> None:
    print(f"Crawling {start_url} (prefix={prefix!r}, max={max_pages} pages, {concurrency} concurrent)...\n")
    saved = await crawl_and_save(start_url, prefix, out_dir, max_pages, concurrency)
    print(f"\nDone. {len(saved)} PDFs saved to {out_dir}")
    if len(saved) <= 1:
        print(
            "\nOnly found 1 page — if this site's nav is still not showing up, it may load links\n"
            "asynchronously after a longer delay, or require scrolling/clicking to reveal them.\n"
            "Try increasing the wait, or share the site's sitemap.xml URL if it has one."
        )
    print("Next: run scripts/ingest_company_pdfs.py --company <id> to index them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="Starting URL, e.g. https://example.com/docs")
    parser.add_argument("--prefix", default="", help="Only crawl URLs whose path starts with this, e.g. /docs")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for PDFs")
    parser.add_argument("--max-pages", type=int, default=100, help="Safety cap on number of pages to crawl")
    parser.add_argument("--concurrency", type=int, default=3, help="How many pages to render in parallel")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.prefix, args.out, args.max_pages, args.concurrency))