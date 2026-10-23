#!/bin/bash
# Fetch docs PDFs for 10 companies using fetch_docs_to_pdf.py (no browser needed).
VENV=~/workspace/github-fix/.venv-rag/bin/python
BASE=~/workspace/github-fix/VectorMind/data/companies
cd ~/workspace/github-fix/VectorMind

fetch_co() {
  local id="$1"; local name="$2"; local sitemap="$3"; local prefix="$4"
  echo "=== [$id] ==="
  $VENV scripts/fetch_docs_to_pdf.py --sitemap "$sitemap" --prefix "$prefix" \
    --out "$BASE/$id/raw_pdfs" --max-pages 25 --company "$name" --delay 0.7 2>&1 | tail -2
}

fetch_co "fastapi"    "FastAPI"    "https://fastapi.tiangolo.com/sitemap.xml" "/tutorial/"
fetch_co "supabase"   "Supabase"   "https://docs.supabase.com/sitemap.xml" "/guides/"
fetch_co "vercel"     "Vercel"     "https://vercel.com/sitemap.xml" "/docs/"
fetch_co "cloudflare" "Cloudflare" "https://developers.cloudflare.com/sitemap.xml" "/fundamentals/"
fetch_co "stripe"     "Stripe"     "https://docs.stripe.com/sitemap.xml" "/payments/"
fetch_co "twilio"     "Twilio"     "https://www.twilio.com/sitemap.xml" "/docs/messaging/"
fetch_co "docker"     "Docker"     "https://docs.docker.com/sitemap.xml" "/get-started/"
fetch_co "tailwind"   "Tailwind CSS" "https://tailwindcss.com/sitemap.xml" "/docs/"
fetch_co "notion"     "Notion"     "https://developers.notion.com/sitemap.xml" "/docs/"
fetch_co "figma"      "Figma"      "https://developers.figma.com/sitemap.xml" "/docs/"
echo "ALL DONE"
