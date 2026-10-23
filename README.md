# VectorMind — Multi-Tenant RAG Platform for Company Documentation

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-49%20passing-brightgreen)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)
[![Live Demo](https://img.shields.io/badge/demo-live-success)](https://bkoxhezgya3iddj6xdkdyc.streamlit.app/)

**🚀 [Try the live demo](https://bkoxhezgya3iddj6xdkdyc.streamlit.app/)** — ask questions about FastAPI documentation with page-accurate citations.

A local-first, offline-capable RAG (Retrieval-Augmented Generation) platform
for company engineering documentation. Ingests PDFs, isolates data per
company, and answers questions with page-accurate citations. Runs fully
offline via Ollama by default, with SenseNova cloud fallback for
public deployments.

**[Screenshots below](#screenshots) · Built with pypdf, pdfplumber, BGE embeddings, ChromaDB, BM25, cross-encoder reranking, Ollama, SenseNova, and Streamlit**

## Screenshots

<!--
  Add 2-3 screenshots here before publishing, e.g.:

  ![Chat with citations](screenshots/chat.png)
  ![Compare Companies](screenshots/compare.png)
  ![Company sidebar](screenshots/sidebar.png)

  Put the image files in a `screenshots/` folder at the repo root — GitHub
  renders them automatically once referenced like this in the README.
-->

*(Screenshots coming — see the Streamlit UI section below for what to capture.)*

## Architecture

```
                     ┌─────────────────────────────────────────┐
                     │            configs/{id}.yaml              │
                     │      (inherits from template.yaml)         │
                     └───────────────────┬───────────────────────┘
                                          │
   raw_pdfs/*.pdf  ──▶  pdf_extractor.py  ──▶  chunker.py  ──▶  embedder.py
   (per company)      (pypdf + pdfplumber      (code-block aware,   (BGE, local)
                        fallback + heading/       page-number carrying)
                        code detection)                 │
                                                          ▼
                                              vector_store.py (ChromaDB,
                                              ISOLATED per company_id)
                                                          │
                              manifest.jsonl  ◀──────────┤  (SHA-256 dedup,
                              (append-only,               local-first skip)
                               per company)
                                                          ▼
   query  ──▶  retriever.py (dense + BM25, RRF)  ──▶  reranker.py (cross-encoder)
                                                          │
                                                          ▼
                                          qa_pipeline.py  ──▶  llm.py (Ollama / SenseNova)
                                          (retrieve → rerank → generate → cite)
```

**Isolation guarantee:** every company gets its own `data/companies/{id}/`
directory tree — its own raw PDFs, manifest, checkpoint, and **separate
ChromaDB instance** (not a shared DB with a metadata filter). A bug in a
query's filter can't leak another company's data, because there's no shared
database to leak from.

## Project Layout

```
Enterprise_RAG_Platform/
├── src/rag_platform/
│   ├── config.py            # pydantic settings, company_id-driven paths
│   ├── config_loader.py      # YAML config loading + template inheritance
│   ├── logging_config.py     # structlog JSON, company_id auto-bound
│   ├── pdf_extractor.py      # pypdf + pdfplumber, headings/code/pages
│   ├── chunker.py            # recursive splitter, code-block aware
│   ├── manifest.py           # append-only JSONL, SHA-256 dedup
│   ├── checkpoint.py         # resumable ingestion runs
│   ├── embedder.py           # BGE embedding wrapper
│   ├── vector_store.py       # ChromaDB, one instance per company
│   ├── retriever.py          # BM25 + dense hybrid search (RRF)
│   ├── reranker.py           # cross-encoder reranking
│   ├── llm.py                # Ollama client (Local mode) + SenseNova client (Public mode)
│   ├── qa_pipeline.py        # retrieve → rerank → generate → cite
│   ├── company_registry.py   # company CRUD over registry.json
│   ├── cache.py              # in-memory TTL+LRU query cache (cachetools)
│   ├── chat_history.py       # SQLite-backed persistent chat history
│   └── voice.py              # Groq Whisper client for voice input
├── configs/
│   ├── template.yaml         # defaults every company config inherits
│   └── company_a.yaml        # example company config
├── scripts/
│   ├── ingest_company_pdfs.py  # main ingestion CLI
│   ├── query_company.py        # per-company query CLI
│   ├── manage_companies.py     # list/add/info/remove/update
│   ├── check_manifest.py       # audit manifest vs disk
│   ├── reset_data.py           # wipe a company's derived state
│   └── scrape_docs_to_pdf.py   # crawl a docs website -> PDFs (Playwright, JS-rendered aware)
├── tests/
│   ├── conftest.py
│   ├── test_pdf_extractor.py
│   ├── test_company_registry.py
│   ├── test_config_loader.py
│   ├── test_integration.py
│   ├── test_chat_history_and_cache.py
│   └── test_qa_pipeline.py
├── data/
│   ├── companies/{id}/       # isolated per-tenant data (gitignored)
│   ├── registry.json         # global company registry (gitignored)
│   └── chat_history.sqlite3  # global chat history, per-row company_id (gitignored)
├── streamlit_app.py          # chat UI: multi-company, Local/Public mode toggle, compare, history, voice
├── Dockerfile, docker-compose.yml, .dockerignore
├── pyproject.toml, requirements.txt, requirements-dev.txt, requirements-scraping.txt
├── .env.example, .gitignore, Makefile
└── README.md
```

## Setup

```bash
cd Enterprise_RAG_Platform
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Or with the Makefile: `make install-dev`.

For answer generation you'll also need [Ollama](https://ollama.com) running
locally with a model pulled:
```bash
ollama pull llama3.1:8b
```
Retrieval-only usage (`--no-llm` on `query_company.py`) doesn't need Ollama.

## Onboarding a New Company (3 steps)

> Prefer clicking over typing? The [Streamlit UI](#streamlit-ui) below does
> all three of these steps from the sidebar — register, upload PDFs, and
> ingest — with no terminal commands needed. The CLI steps here are for
> scripting, CI, or bulk onboarding.

**1. Register the company** — creates its isolated data directories:
```bash
python scripts/manage_companies.py add company_a --name "Acme Corp"
```

**2. Create its config**, copying the template and filling in `company.id`:
```bash
cp configs/template.yaml configs/company_a.yaml
# edit configs/company_a.yaml: set company.id: company_a, company.name, etc.
```

**3. Drop PDFs into its raw_pdfs directory and ingest:**
```bash
cp your_docs/*.pdf data/companies/company_a/raw_pdfs/
python scripts/ingest_company_pdfs.py --company company_a
```

## Querying a Company

```bash
# Full QA with LLM-generated answer + citations
python scripts/query_company.py --company company_a --query "What is the deployment process?"

# Retrieval only — see the ranked chunks without generating an answer
python scripts/query_company.py --company company_a --query "rollback steps" --no-llm --top-k 5
```

## CLI Reference

| Command | Purpose |
|---|---|
| `scripts/ingest_company_pdfs.py --company ID` | Extract, chunk, embed, index a company's PDFs. Resumable. |
| `scripts/query_company.py --company ID --query "..."` | Query a company's indexed corpus. `--no-llm` for retrieval only. |
| `scripts/manage_companies.py list` | Show all registered companies with stats. |
| `scripts/manage_companies.py add ID --name "Name"` | Register a new company, create its data directories. |
| `scripts/manage_companies.py info ID` | Show detailed stats for one company. |
| `scripts/manage_companies.py remove ID [--delete-data]` | Unregister; optionally delete its data too. |
| `scripts/manage_companies.py update ID` | Refresh registry stats from the manifest on disk. |
| `scripts/check_manifest.py --company ID` | Audit manifest vs disk; exits non-zero on drift (CI-friendly). |
| `scripts/reset_data.py --company ID [--yes]` | Wipe derived state (clean/, manifest, checkpoint, vector store). Raw PDFs untouched. |
| `scripts/scrape_docs_to_pdf.py --url URL --out DIR` | Crawl a docs website into PDFs (see [Scraping a Documentation Website](#scraping-a-documentation-website-into-pdfs)). |

Makefile shortcuts: `make ingest COMPANY=company_a`, `make query COMPANY=company_a QUERY="..."`, `make companies`, `make check-manifest COMPANY=company_a`, `make reset COMPANY=company_a`, `make scrape-docs URL=... PREFIX=/docs OUT=data/companies/example/raw_pdfs`.

## Scraping a Documentation Website into PDFs

If a company's docs live on a website rather than as downloadable PDFs,
`scripts/scrape_docs_to_pdf.py` crawls it and saves every page as a PDF —
the automated equivalent of visiting each page and doing Ctrl+P.

```bash
pip install -r requirements-scraping.txt
playwright install chromium

python scripts/scrape_docs_to_pdf.py \
    --url https://example.com/docs \
    --prefix /docs \
    --out data/companies/example/raw_pdfs \
    --max-pages 200
```

Uses a real headless browser (Playwright), not a plain HTTP request, for
both discovering links and exporting PDFs — most modern documentation
sites (Next.js, Docusaurus, Mintlify, VitePress) render their navigation
with client-side JavaScript, which a plain GET request never sees. Stays
within the given domain + path prefix, and crawls with bounded concurrency.
Only use this on sites you own or have explicit permission to scrape.

Kept as a separate optional dependency (`requirements-scraping.txt`) since
Playwright + a bundled Chromium binary (~150–300MB) is only needed for this
one-off step, not for running the RAG platform itself.

## Streamlit UI

A full chat-style UI is included (`streamlit_app.py`) — company switcher, PDF
upload with in-browser ingestion, retrieval + reranked citations, optional
LLM-generated answers, persistent chat history, cross-company comparison,
in-memory caching, and voice input.

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Three tabs once a company is selected:

- **💬 Chat** — ask questions, optionally record audio instead of typing
  (needs a free [Groq API key](https://console.groq.com/keys) — see below).
  Retrieval + citations work with no setup; check "Generate LLM answer" to
  also generate a synthesized response. Two LLM modes: **🔒 Local** (Ollama,
  fully offline) or **🌐 Public** (SenseNova's hosted LLM — works even
  without a local Ollama, e.g. on a deployed demo). Switch per-question.
  Follow-up questions ("how do I fix it?") are automatically rewritten
  using recent conversation history before retrieval, so they don't need
  to repeat context the previous message already established.
- **🔍 Compare Companies** — ask the same question across 2+ companies at
  once, answers shown side by side. Useful for "how does GitLab's policy on
  X differ from Basecamp's?" style questions.
- **📜 History** — every question you've asked (per company) is saved to a
  local SQLite database automatically. Search, star, or delete past
  conversations.

**Caching:** identical (company, question, top_k, LLM-on/off) combinations
are served from an in-memory cache (`cachetools.TTLCache`, 1 hour TTL by
default) instead of re-running embedding + retrieval + reranking. This is a
process-local cache, not Redis — see the design decisions section below for
why that's the right call at this scale.

**Voice input** is the one feature that isn't fully local: it calls Groq's
hosted Whisper API (free tier) to transcribe recorded audio to text. Text
queries, retrieval, and LLM generation (via Ollama) all remain local; only
the optional microphone button leaves the machine, and only when you use it.
Set `GROQ_API_KEY` (or `RAG_GROQ_API_KEY`) in your environment or `.env` to
enable it — without a key, everything else works exactly the same, the mic
button just shows a clear error if used.

### Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (see `.gitignore` — company data and `.env`
   are excluded by default, so you're not committing indexed PDFs).
2. If you want the hosted demo to show real results out of the box, force-add
   one small pre-ingested company so visitors see something immediately:
   ```bash
   git add -f data/companies/demo_co/ configs/demo_co.yaml
   ```
3. On [share.streamlit.io](https://share.streamlit.io), point a new app at
   this repo, main branch, `streamlit_app.py` as the entry point.
4. **Important:** the hosted app has no access to your local Ollama —
   "localhost:11434" only means something on your own machine. Two options:
   - Leave "Generate LLM answer" unchecked by default on the hosted demo.
     Retrieval + reranking + citations alone are a complete, honest showcase
     of the pipeline and need nothing extra to run.
   - Point the sidebar's Ollama URL field at a publicly reachable Ollama
     instance (e.g. tunneled via `ngrok http 11434` while your machine is
     on) if you want live LLM answers during a demo session.
5. Streamlit Cloud's filesystem is **ephemeral** — anything ingested through
   the hosted UI (new companies, uploaded PDFs) is lost on redeploy/restart.
   That's expected for a live demo; treat step 2's pre-ingested company as
   the persistent "always works" showcase, and uploads as a "try it live"
   feature that resets periodically.
6. For voice input on the hosted demo, add `GROQ_API_KEY` under the app's
   **Secrets** in the Streamlit Cloud dashboard (Settings → Secrets) —
   never commit a real key to the repo.

### Deploying with Docker

```bash
docker compose up --build
```

This starts both the app (`localhost:8501`) and an Ollama instance in the
same Compose stack — `docker compose exec ollama ollama pull llama3.1:8b`
to pull a model into the containerized Ollama the first time. Company data
persists in a named Docker volume (`rag_data`) across container restarts.
If you'd rather use an Ollama already running on your host instead of the
one in Compose, delete the `ollama` service from `docker-compose.yml` and
point `RAG_OLLAMA_BASE_URL` at `http://host.docker.internal:11434` (Mac/
Windows) or your host's LAN IP (Linux).

## Running Tests

```bash
make test
# or directly:
PYTHONPATH=src pytest tests/ -v
```

49 tests across 7 files, all passing:
- `test_pdf_extractor.py` — heading/code-block detection, page markers, scanned-PDF flagging
- `test_company_registry.py` — add/list/info/remove/duplicate/not-found
- `test_config_loader.py` — YAML template inheritance, validation errors, tenant isolation
- `test_integration.py` — full pipeline (extract → chunk → index → retrieve) against a synthetic reportlab-generated PDF, plus scanned-PDF skip behavior
- `test_chat_history_and_cache.py` — chat history isolation/search/star/delete, cache hit/miss behavior
- `test_qa_pipeline.py` — relevance-gap filtering against cross-chunk hallucination, system prompt guardrails, multi-turn query condensing
- `test_regressions.py` — page-attribution accuracy (no citation drift), stale-version cleanup on re-ingest, multi-tenant retrieval isolation, corrupt-registry handling

## Validated on Real Documentation

Tested end-to-end on **828 PDFs / 9,107 chunks** across 5 companies:

| Company | PDFs | Chunks |
|---------|------|--------|
| FastAPI | 51 | 900 |
| Vue.js | 108 | 1,244 |
| Nuxt | 284 | 1,512 |
| Vite | 24 | 378 |
| VS Code | 361 | 5,073 |

**13 hard questions** stress-tested (multi-hop reasoning, nuanced comparisons like `ref` vs `reactive` vs `shallowRef`, edge cases like `response_model` extra-field filtering): all answered accurately with page-numbered citations. **Zero hallucination** — on unanswerable questions (e.g. "Does FastAPI support GraphQL subscriptions?"), the system honestly reports the docs don't cover it instead of inventing an answer.

Integration and unit tests use a deterministic fake embedder instead of
downloading the real BGE model, so the suite runs fully offline and in
under 2 seconds. To smoke-test with the real model, run
`scripts/ingest_company_pdfs.py` against a real company directly (first run
downloads ~130MB from Hugging Face).

## Design Decisions

1. **PDF library:** pypdf primary (fast, handles most digitally-authored
   PDFs), pdfplumber fallback — used both when pypdf's output looks sparse
   AND as the only source of per-character font metadata, which is what
   heading/code-block detection actually needs.
2. **Per-company ChromaDB, not shared + filter:** stronger isolation
   guarantee — see Architecture above.
3. **Process pool for PDF extraction:** text extraction is CPU-bound;
   `ProcessPoolExecutor` sidesteps the GIL, unlike a thread pool.
4. **Config inheritance:** company YAMLs deep-merge onto `template.yaml` —
   only override what differs; platform-wide defaults change in one place.
5. **Registry storage:** single `registry.json`, not SQLite — simpler to
   inspect/diff/back up at the scale (tens of companies) this targets.
6. **Page numbers:** inline `<!-- page:N -->` markers in the extracted
   markdown, not a side-channel metadata list — survives the chunker's
   paragraph merging/splitting, which a separate index-aligned list wouldn't.
7. **Chunking for technical docs:** fenced code blocks are treated as
   atomic units, never split mid-line, and never blended into the
   overlap carried between chunks (see the docstring on
   `chunker.py::_merge_with_overlap` for the specific bug this avoids).
8. **Cache: in-memory (cachetools), not Redis:** at single-user/portfolio
   scale, an external cache server is operational overhead with no payoff —
   a process-local `TTLCache` gives the same hit/miss/expiry behavior with
   zero extra infrastructure. Redis becomes the right call once multiple
   app instances need to share one cache, not before.
9. **Chat history: one shared SQLite DB, not per-company:** every row is
   tagged with `company_id`, but there's a single `data/chat_history.sqlite3`
   rather than one DB per tenant. Chat history is the user's own question
   log, not another tenant's source documents — it doesn't need the hard
   isolation vector stores need, and a single DB makes "show everything
   I've asked" and cross-company comparison a plain filtered query instead
   of a fan-out across N databases.
10. **Voice input is the one non-local step, by design:** Whisper models are
    heavy enough that local CPU transcription is slow, so voice input calls
    Groq's hosted Whisper API (free tier) instead. Every other step —
    embeddings, retrieval, reranking, LLM generation — stays fully local;
    voice is opt-in per question and only runs when you click the mic.
11. **A relative score-gap filter, not an absolute threshold or just a
    bigger `top_k`:** user testing surfaced a real hallucination — the LLM
    blended two unrelated retrieved chunks (an "attendance ownership"
    passage and a "grade editing" question) into a false causal explanation
    neither source actually stated. An absolute score cutoff was tried
    first and broke on the very next real query: the reranker scored a
    genuinely correct match at -7.6 for one question while scoring an
    irrelevant chunk at -8.2 for a *different* question — cross-encoder
    scores aren't on a fixed scale between queries. `QAPipeline` now drops
    any chunk scoring more than `max_score_gap` points below *that same
    query's own* top-ranked result (always keeping at least the top
    match), and the system prompt explicitly forbids inferring a
    relationship between context blocks unless one block states it
    directly — see `tests/test_qa_pipeline.py`, built from both the
    original hallucination and the real production data that broke the
    first fix attempt.
12. **Multi-turn memory via query condensing, not raw context stuffing:** a
    bare follow-up like "how do I fix it?" embeds and retrieves poorly on
    its own — nothing in the vector alone indicates what "it" refers to.
    Before retrieval, conversation history is used to rewrite the follow-up
    into a standalone question (one extra, cheap LLM call, made only when
    history exists). Only the single *immediately preceding* turn is
    included in the final generation prompt, not several — an earlier
    2-turn version caused a real bug where the model blended a much-earlier,
    unrelated topic (a "room conflict" question) into the answer for a
    completely different follow-up (a CSV-import question), because that
    unrelated topic happened to be the second-most-recent turn. One turn is
    enough for legitimate pronoun/context resolution without dragging in
    whatever else the conversation covered several messages back. The
    system prompt also explicitly forbids treating prior *answers* as a
    source of facts — only as context for understanding the current
    question — otherwise multi-turn memory reopens the same hallucination
    path decision 11 closed. If the condensing call itself fails, the
    pipeline falls back to the raw follow-up rather than erroring out.

## Troubleshooting

**"HTTP 403" / "couldn't connect to huggingface.co" during ingestion**
The embedding model downloads from Hugging Face on first use. Check your
network connection and that huggingface.co isn't blocked by a firewall/proxy.
Subsequent runs use the local cache (`~/.cache/huggingface`) and don't need
network access.

**A PDF is flagged as scanned but it has real text**
The scanned-detection heuristic (>50% of pages under 20 extracted
characters) is deliberately conservative. If a real PDF trips it, check
`scripts/check_manifest.py --company ID` — it will show `[SCANNED]` for
that file. Multi-column layouts or unusual fonts can occasionally cause
false positives; there's no OCR fallback by design, so such a PDF should be
re-exported or manually converted before ingesting.

**Ollama connection refused**
`query_company.py` (without `--no-llm`) needs a local Ollama server.
Start it with `ollama serve` and confirm the model is pulled:
`ollama pull llama3.1:8b`.

**Re-running ingestion doesn't pick up a PDF I edited**
Confirm the file's actual bytes changed (SHA-256-based dedup, not
mtime-based) — re-saving without changing content won't trigger
reprocessing. `scripts/check_manifest.py --company ID` shows
`HASH_MISMATCH` for files that will be reprocessed on the next run.

**A company's data directory got corrupted / I want to start over**
`scripts/reset_data.py --company ID` wipes derived state (manifest, clean
files, checkpoint, vector store) without touching `raw_pdfs/` — re-run
`ingest_company_pdfs.py` afterward to rebuild everything from the source PDFs.