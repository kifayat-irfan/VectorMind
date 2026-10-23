#!/bin/bash
# FULL documentation fetch for 10 companies (no 25-page cap).
# Safety cap: 2000 pages per company (covers full docs for 9/10 sites).
# Skips already-downloaded PDFs, so re-runs continue where they left off.
# This takes HOURS — run in background.
VENV=~/workspace/github-fix/.venv-rag/bin/python
BASE=~/workspace/github-fix/VectorMind/data/companies
cd ~/workspace/github-fix/VectorMind

fetch_co() {
  local id="$1"; local name="$2"; local sitemap="$3"; local prefix="$4"; local maxp="${5:-2000}"
  echo "=== [$id] ==="
  $VENV scripts/fetch_docs_to_pdf.py --sitemap "$sitemap" --prefix "$prefix" \
    --out "$BASE/$id/raw_pdfs" --max-pages "$maxp" --company "$name" --delay 0.4 2>&1 | tail -2
}

fetch_co "fastapi"    "FastAPI"    "https://fastapi.tiangolo.com/sitemap.xml" "/tutorial/"
fetch_co "vercel"     "Vercel"     "https://vercel.com/sitemap.xml" "/docs/"
fetch_co "stripe"     "Stripe"     "https://docs.stripe.com/sitemap.xml" "/payments/"
fetch_co "supabase"   "Supabase"   "https://supabase.com/sitemap.xml" "/docs/"
fetch_co "docker"     "Docker"     "https://docs.docker.com/sitemap.xml" "/"
fetch_co "cloudflare" "Cloudflare" "https://developers.cloudflare.com/sitemap.xml" "/fundamentals/"
fetch_co "figma"      "Figma"      "https://developers.figma.com/sitemap.xml" "/docs/"
fetch_co "django"     "Django"     "https://docs.djangoproject.com/sitemap.xml" "/en/5.2/topics/" 500
fetch_co "notion"     "Notion"     "https://developers.notion.com/sitemap.xml" "/docs/"
# 10th: DigitalOcean
fetch_co "digitalocean" "DigitalOcean" "https://docs.digitalocean.com/sitemap.xml" "/products/" 500
echo "ALL DONE"
