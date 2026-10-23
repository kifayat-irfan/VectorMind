#!/usr/bin/env python3
"""
Fetch documentation pages and save each as a PDF — without a browser.

Why this exists: the Playwright-based scrape_docs_to_pdf.py needs a real
browser with direct internet access. In sandboxed/proxied environments where
only plain HTTP (requests/curl) works, this script does the same job:
  1. Discover page URLs from the site's sitemap.xml (via requests)
  2. Fetch each page's HTML (via requests)
  3. Extract the main article content (BeautifulSoup, common selectors)
  4. Render a clean PDF with weasyprint

Usage:
    python scripts/fetch_docs_to_pdf.py \\
        --sitemap https://fastapi.tiangolo.com/sitemap.xml \\
        --prefix /tutorial/ \\
        --out data/companies/fastapi/raw_pdfs \\
        --max-pages 25 \\
        --company "FastAPI"

Requires: pip install weasyprint beautifulsoup4 lxml  (in .venv-rag already)
"""

from __future__ import annotations

import argparse
import html as html_lib
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# Selectors tried in order to find the main article content.
CONTENT_SELECTORS = [
    "main",
    "article",
    "[role='main']",
    ".content",
    "#content",
    ".documentation",
    ".docs-content",
    ".markdown-body",
    ".prose",
]


def fetch(url: str, timeout: int = 30) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp


def sitemap_urls(sitemap_url: str, prefix: str = "") -> list[str]:
    """Pull URLs from a sitemap.xml (handles sitemap indexes too)."""
    print(f"  fetching sitemap: {sitemap_url}")
    resp = fetch(sitemap_url)
    root = ElementTree.fromstring(resp.content)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls: list[str] = []

    # Sitemap index -> recurse into child sitemaps
    for sm in root.findall("s:sitemap", ns):
        loc = sm.find("s:loc", ns)
        if loc is not None and loc.text:
            urls.extend(sitemap_urls(loc.text.strip(), prefix))

    for u in root.findall("s:url", ns):
        loc = u.find("s:loc", ns)
        if loc is not None and loc.text:
            url = loc.text.strip()
            if not prefix or urlparse(url).path.startswith(prefix):
                urls.append(url)

    # No namespace (some sites omit it)
    if not urls:
        for loc in root.iter("loc"):
            if loc.text and (not prefix or urlparse(loc.text.strip()).path.startswith(prefix)):
                # skip nested sitemap locs already handled
                if loc.text.strip().endswith(".xml"):
                    continue
                urls.append(loc.text.strip())
    return urls


def sanitize_filename(url: str) -> str:
    path = urlparse(url).path.strip("/") or "index"
    safe = re.sub(r"[^a-zA-Z0-9\-_]", "_", path.replace("/", "__"))
    return f"{safe[:120]}.pdf"


def extract_article(html_text: str, url: str) -> tuple[str, str]:
    """Return (title, clean_html) for the page's main content."""
    soup = BeautifulSoup(html_text, "lxml")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    body = None
    for sel in CONTENT_SELECTORS:
        body = soup.select_one(sel)
        if body and len(body.get_text(strip=True)) > 200:
            break

    if body is None:
        body = soup.find("body") or soup

    # Keep only meaningful tags
    clean = BeautifulSoup("", "lxml")
    wrapper = clean.new_tag("div")
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code", "blockquote", "table"]):
        text = el.get_text(" ", strip=True)
        if not text or len(text) < 2:
            continue
        new = clean.new_tag(el.name if el.name in ("h1", "h2", "h3", "h4", "p", "li", "blockquote") else "p")
        new.string = text
        wrapper.append(new)

    return title, str(wrapper)


def to_pdf(title: str, article_html: str, company: str, url: str, dest: Path) -> None:
    from weasyprint import HTML as WeasyHTML

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <style>
      body {{ font-family: sans-serif; font-size: 11pt; line-height: 1.5; margin: 40px; }}
      h1 {{ font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 8px; }}
      h2 {{ font-size: 15pt; color: #1a1a2e; margin-top: 24px; }}
      h3 {{ font-size: 12pt; color: #333; }}
      pre, code {{ background: #f4f4f4; font-size: 9pt; }}
      pre {{ padding: 10px; white-space: pre-wrap; }}
      .meta {{ color: #666; font-size: 9pt; border-bottom: 1px solid #ccc; padding-bottom: 10px; }}
    </style></head><body>
    <div class="meta">{html_lib.escape(company)} — Source: {html_lib.escape(url)}</div>
    <h1>{html_lib.escape(title)}</h1>
    {article_html}
    </body></html>"""
    WeasyHTML(string=doc).write_pdf(str(dest))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sitemap", required=True, help="sitemap.xml URL, e.g. https://example.com/sitemap.xml")
    ap.add_argument("--prefix", default="", help="only URLs whose path starts with this, e.g. /docs/")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-pages", type=int, default=25)
    ap.add_argument("--company", default="Company", help="company name stamped on each PDF")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (be polite)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    try:
        urls = sitemap_urls(args.sitemap, args.prefix)
    except Exception as exc:
        print(f"ERROR fetching sitemap: {exc}", file=sys.stderr)
        raise SystemExit(1)

    # de-dup, keep order
    seen, unique = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    urls = unique[: args.max_pages]
    print(f"  found {len(urls)} pages (capped at {args.max_pages})")

    saved = 0
    for i, url in enumerate(urls, 1):
        dest = args.out / sanitize_filename(url)
        if dest.exists():
            saved += 1
            continue
        try:
            resp = fetch(url)
            title, article = extract_article(resp.text, url)
            if len(BeautifulSoup(article, "lxml").get_text(strip=True)) < 300:
                print(f"  [{i}/{len(urls)}] SKIP (thin content): {url}")
                continue
            to_pdf(title, article, args.company, url, dest)
            saved += 1
            print(f"  [{i}/{len(urls)}] saved: {dest.name}")
        except Exception as exc:
            print(f"  [{i}/{len(urls)}] FAILED: {url} ({str(exc)[:120]})", file=sys.stderr)
        time.sleep(args.delay)

    print(f"\nDone. {saved} PDFs in {args.out}")


if __name__ == "__main__":
    main()
