#!/bin/bash
# Crawl documentation sites for 10 companies -> PDFs for VectorMind testing.
# Run: bash scripts/crawl_10_companies.sh (takes a while; run in background)
VENV=~/workspace/github-fix/.venv-rag/bin/python
BASE=~/workspace/github-fix/VectorMind/data/companies

crawl() {
  local id="$1"; local url="$2"; local prefix="$3"
  echo "=== [$id] $url ==="
  $VENV scripts/scrape_docs_to_pdf.py --url "$url" --prefix "$prefix" \
    --out "$BASE/$id/raw_pdfs" --max-pages 25 --concurrency 2 2>&1 | tail -3
}

cd ~/workspace/github-fix/VectorMind
crawl "supabase"   "https://docs.supabase.com/guides" "guides"
crawl "vercel"     "https://vercel.com/docs" "/docs"
crawl "cloudflare" "https://developers.cloudflare.com/fundamentals/" "/fundamentals"
crawl "stripe"     "https://docs.stripe.com/payments" "/payments"
crawl "twilio"     "https://www.twilio.com/docs/messaging" "/docs/messaging"
crawl "docker"     "https://docs.docker.com/get-started/" "/get-started"
crawl "fastapi"    "https://fastapi.tiangolo.com/tutorial/" "/tutorial"
crawl "tailwind"   "https://tailwindcss.com/docs/installation" "/docs"
crawl "notion"     "https://developers.notion.com/docs" "/docs"
crawl "figma"      "https://developers.figma.com/docs" "/docs"
echo "ALL DONE"
