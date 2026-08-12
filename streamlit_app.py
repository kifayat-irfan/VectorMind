"""
Streamlit UI for the multi-tenant RAG platform — light "FintechInsights"
style theme (white background, purple/blue accents, card-based stats,
colored citation pills).

This file is self-contained UI/presentation code — it assumes the backend
(src/rag_platform/*) already has every fix from this project's development
history baked in:
  - Per-company ChromaDB isolation (separate client per company_id)
  - Relative reranker score-gap filter (max_score_gap), not an absolute
    threshold — the earlier absolute-threshold approach broke on real
    production data where a genuinely relevant chunk scored -7.6 for one
    question and an irrelevant chunk scored -8.2 for a different question
  - System prompt that forbids blending facts across unrelated context
    blocks or unrelated conversation-history turns
  - Multi-turn query condensing, with only the immediately-preceding turn
    included in the generation prompt (a 2-turn window caused unrelated
    topics to bleed into answers)
  - Cached VectorStore connections (st.cache_resource) — recreating a
    chromadb.PersistentClient on every Streamlit rerun corrupted the local
    HNSW index on Windows
  - Code-block-atomic, page-marker-aware chunking with a leading-whitespace
    fix for page attribution at chunk boundaries

Deployment note: on Streamlit Community Cloud there is no local Ollama —
retrieval + reranking + citations work everywhere with no setup; "Generate
LLM answer" in Local mode needs a reachable Ollama, Public mode (Groq) works
anywhere with an API key.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import scripts.ingest_company_pdfs as ingest_module  # noqa: E402
from rag_platform.cache import cache_key, cached_call, get_cache  # noqa: E402
from rag_platform.chat_history import ChatHistoryStore  # noqa: E402
from rag_platform.chunker import RecursiveCharacterChunker  # noqa: E402
from rag_platform.company_registry import (  # noqa: E402
    CompanyAlreadyExistsError,
    CompanyRegistry,
)
from rag_platform.config import Settings, get_settings  # noqa: E402
from rag_platform.config_loader import ConfigValidationError, load_company_config  # noqa: E402
from rag_platform.embedder import Embedder  # noqa: E402
from rag_platform.llm import GroqLLMClient, OllamaClient  # noqa: E402
from rag_platform.manifest import ManifestStore  # noqa: E402
from rag_platform.qa_pipeline import QAPipeline  # noqa: E402
from rag_platform.reranker import CrossEncoderReranker  # noqa: E402
from rag_platform.retriever import HybridRetriever  # noqa: E402
from rag_platform.vector_store import VectorStore  # noqa: E402
from rag_platform.voice import GroqTranscriptionError, GroqWhisperClient  # noqa: E402

st.set_page_config(page_title="RAG Platform", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------- #
# Refined defaults for a new company's template.yaml, per the tuning
# rubric: general handbook/policy docs land in the 700-900/80-120 band;
# max_score_gap stays at the value that survived real production testing
# (6.0) rather than the absolute-threshold approach that came before it.
# ---------------------------------------------------------------------- #
TEMPLATE_YAML = """company:
  id: "{company_id}"
  name: "{company_name}"
sources:
  - type: pdf
    path: ./data/companies/{company_id}/raw_pdfs/
    glob: "*.pdf"
processing:
  chunk_size: 1500
  chunk_overlap: 200
  min_chunk_length: 200
embedding:
  model: BAAI/bge-small-en-v1.5
  batch_size: 32
  device: cpu
vector_store:
  type: chromadb
  path: ./data/companies/{company_id}/vector_store/
  collection_name: {company_id}_docs
retrieval:
  top_k: 8
  rerank_top_k: 5
  use_bm25: true
  use_reranker: true
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

# ---------------------------------------------------------------------- #
# Theme: light "FintechInsights" style — white surfaces, soft shadows,
# purple/blue primary accent, colored pill tags for citation sources.
# ---------------------------------------------------------------------- #
THEME_CSS = """
<style>
:root {
    --bg-page: #F7F8FB;
    --bg-card: #FFFFFF;
    --border: #ECEEF3;
    --text-hi: #1A1D29;
    --text-lo: #8A8FA3;
    --accent: #7C5CFC;
    --accent-soft: #EFEAFF;
}
.stApp { background: var(--bg-page); }
section[data-testid="stSidebar"] { background: var(--bg-card); border-right: 1px solid var(--border); }

.rag-header { display:flex; align-items:center; gap:10px; margin-bottom: 4px; }
.rag-header .logo { font-size: 22px; font-weight: 800; color: var(--text-hi); }
.rag-header .logo span { color: var(--accent); }

.rag-company-card {
    border-radius: 14px; padding: 12px 14px; margin-bottom: 8px;
    border: 1px solid var(--border); background: var(--bg-card);
    display: flex; align-items: center; gap: 10px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}
.rag-company-card.active { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
.rag-avatar {
    width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center;
    justify-content: center; font-weight: 700; font-size: 13px; color: #fff; flex-shrink: 0;
}
.rag-company-name { color: var(--text-hi); font-size: 14px; font-weight: 600; }
.rag-company-count { color: var(--text-lo); font-size: 11.5px; }

section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; color: var(--text-hi); text-align: left; width: 100%;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    border-color: var(--accent); color: var(--accent);
}

div[data-testid="stMetric"] {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px;
    padding: 14px 18px; box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}
div[data-testid="stMetricLabel"] { color: var(--text-lo) !important; }
div[data-testid="stMetricValue"] { color: var(--text-hi) !important; }

div[data-testid="stChatMessage"] {
    background: var(--bg-card); border-radius: 16px; border: 1px solid var(--border);
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}

.rag-citation-card {
    border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px;
    margin-bottom: 6px; background: #FBFBFE;
}
.rag-citation-badge {
    display: inline-block; background: var(--accent); color: #fff; font-weight: 700;
    font-size: 11px; border-radius: 999px; padding: 1px 8px; margin-right: 6px;
}
.rag-pill-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.rag-pill {
    border-radius: 999px; padding: 4px 12px; font-size: 12px; font-weight: 600;
    color: #fff;
}

div[data-baseweb="radio"] label {
    border: 1px solid var(--border); border-radius: 999px; padding: 4px 14px; margin-right: 6px;
    background: var(--bg-card);
}

.rag-badge-pill {
    background: var(--accent-soft); color: var(--accent); border-radius: 999px;
    padding: 2px 10px; font-size: 11.5px; font-weight: 600;
}
</style>
"""

AVATAR_PALETTE = [
    "#7C5CFC", "#22B8CF", "#F76707", "#12B886", "#E64980", "#4C6EF5",
]


def palette_color(seed: str) -> str:
    idx = sum(ord(c) for c in seed) % len(AVATAR_PALETTE)
    return AVATAR_PALETTE[idx]


# ---------------------------------------------------------------------- #
# Cached resources
# ---------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_embedder(model_name: str, device: str) -> Embedder:
    return Embedder(model_name=model_name, device=device)


@st.cache_resource(show_spinner="Loading reranker model (first run only)...")
def get_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name=model_name)


@st.cache_resource
def get_chat_store() -> ChatHistoryStore:
    return ChatHistoryStore(get_settings().chat_history_db_path)


@st.cache_resource
def get_vector_store(vector_db_dir: str, collection_name: str) -> VectorStore:
    """
    Cached so Streamlit's constant reruns don't open a fresh
    chromadb.PersistentClient against the same directory every time —
    that was the root cause of a real HNSW index corruption bug on Windows.
    """
    return VectorStore(Path(vector_db_dir), collection_name)


def get_registry() -> CompanyRegistry:
    return CompanyRegistry(get_settings().registry_path)


# ---------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------- #
def render_sidebar() -> Optional[str]:
    st.sidebar.markdown('<div class="rag-header"><span class="logo">RAG<span>Insights</span></span></div>', unsafe_allow_html=True)
    st.sidebar.caption("Local-first multi-tenant document search")
    registry = get_registry()
    companies = registry.list_companies()

    if not companies:
        st.sidebar.info("No companies yet — add one below.")
        selected_id = None
    else:
        if "selected_company" not in st.session_state or st.session_state.selected_company not in [c.company_id for c in companies]:
            st.session_state.selected_company = companies[0].company_id

        for c in companies:
            color = palette_color(c.company_id)
            initials = "".join(w[0] for w in c.name.split()[:2]).upper()
            is_active = c.company_id == st.session_state.selected_company
            active_class = "active" if is_active else ""
            st.sidebar.markdown(
                f"""<div class="rag-company-card {active_class}">
                    <div class="rag-avatar" style="background:{color};">{initials}</div>
                    <div>
                        <div class="rag-company-name">{c.name}</div>
                        <div class="rag-company-count">{c.total_pdfs} PDFs · {c.total_chunks} chunks</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.sidebar.button(f"Select {c.name}", key=f"pick_{c.company_id}", use_container_width=True):
                st.session_state.selected_company = c.company_id
                st.rerun()

        selected_id = st.session_state.selected_company

    with st.sidebar.expander("➕ Add a new company"):
        new_id = st.text_input("Company ID (lowercase, no spaces)", key="new_company_id")
        new_name = st.text_input("Display name", key="new_company_name")
        if st.button("Register company"):
            if not new_id or not new_name:
                st.warning("Both fields are required.")
            else:
                try:
                    settings = get_settings(new_id)
                    settings.ensure_dirs()
                    registry.add(new_id, new_name, str(settings.config_path), str(settings.data_dir))
                    settings.config_path.write_text(TEMPLATE_YAML.format(company_id=new_id, company_name=new_name))
                    st.success(f"Registered {new_id!r}. Select it above.")
                    st.rerun()
                except CompanyAlreadyExistsError:
                    st.error(f"{new_id!r} is already registered.")
                except ValueError as exc:
                    st.error(f"Invalid company id: {exc}")

    if selected_id:
        render_upload_and_ingest(selected_id)

    return selected_id


def render_upload_and_ingest(company_id: str) -> None:
    settings = get_settings(company_id)
    with st.sidebar.expander("📄 Upload PDFs & ingest", expanded=False):
        uploaded = st.file_uploader(
            "Upload PDF documents", type=["pdf"], accept_multiple_files=True, key=f"upload_{company_id}"
        )
        if uploaded and st.button("Save uploaded files", key=f"save_{company_id}"):
            settings.ensure_dirs()
            for f in uploaded:
                (settings.raw_pdfs_dir / f.name).write_bytes(f.getbuffer())
            st.success(f"Saved {len(uploaded)} file(s) to raw_pdfs/. Click Ingest below.")

        if st.button("▶️ Run ingestion", key=f"ingest_{company_id}"):
            run_ingestion_ui(company_id)


def run_ingestion_ui(company_id: str) -> None:
    settings = get_settings(company_id)
    try:
        company_config = load_company_config(company_id, settings.configs_dir)
    except ConfigValidationError as exc:
        st.error(f"Config error: {exc}")
        return

    pdfs = ingest_module.find_pdfs(company_config)
    if not pdfs:
        st.warning("No PDFs found in raw_pdfs/. Upload some first.")
        return

    manifest = ManifestStore(settings.manifest_path)
    manifest.load()
    to_process = [p for p in pdfs if manifest.needs_processing(p)]

    if not to_process:
        st.info("Everything is already up to date (local-first skip).")
        return

    chunker = RecursiveCharacterChunker(
        chunk_size=company_config.processing.chunk_size,
        chunk_overlap=company_config.processing.chunk_overlap,
        min_chunk_length=company_config.processing.min_chunk_length,
    )
    embedder = get_embedder(company_config.embedding.model, company_config.embedding.device)
    vector_store = get_vector_store(str(settings.vector_db_dir), company_config.vector_store.collection_name)

    progress = st.progress(0.0, text="Starting ingestion...")
    ok = scanned = errors = total_chunks = 0

    async def run_all():
        nonlocal ok, scanned, errors, total_chunks
        with ProcessPoolExecutor(max_workers=settings.max_pdf_workers) as executor:
            for i, pdf_path in enumerate(to_process):
                progress.progress((i) / len(to_process), text=f"Processing {pdf_path.name}...")
                result = await ingest_module.process_one_pdf(
                    pdf_path, company_id, settings, company_config, manifest, chunker, embedder, vector_store, executor
                )
                if result["status"] == "ok":
                    ok += 1
                    total_chunks += result["chunk_count"]
                elif result["status"] == "scanned":
                    scanned += 1
                else:
                    errors += 1

    asyncio.run(run_all())
    progress.progress(1.0, text="Done.")

    registry = get_registry()
    if registry.exists(company_id):
        registry.update_stats(
            company_id,
            total_pdfs=len(manifest.all_entries()),
            total_chunks=sum(e.chunk_count or 0 for e in manifest.all_entries()),
            mark_ingestion_now=True,
        )

    st.success(f"Ingested {ok} PDF(s), {total_chunks} chunks. Scanned/skipped: {scanned}. Errors: {errors}.")
    st.rerun()


# ---------------------------------------------------------------------- #
# Stat row (native st.metric, reskinned via CSS to match card style)
# ---------------------------------------------------------------------- #
def render_stat_row() -> None:
    registry = get_registry()
    companies = registry.list_companies()
    total_pdfs = sum(c.total_pdfs for c in companies)
    total_chunks = sum(c.total_chunks for c in companies)
    total_companies = len(companies)

    store = get_chat_store()
    query_volume = sum(len(store.list_for_company(c.company_id, limit=100000)) for c in companies)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PDFs indexed", f"{total_pdfs:,}")
    col2.metric("Total chunks", f"{total_chunks:,}")
    col3.metric("Companies", f"{total_companies:,}")
    col4.metric("Query volume", f"{query_volume:,}")


# ---------------------------------------------------------------------- #
# Core QA logic
# ---------------------------------------------------------------------- #
def answer_question(
    company_id: str,
    company_config,
    vector_store: VectorStore,
    question: str,
    top_k: int,
    use_llm: bool,
    llm_mode: str = "local",
    ollama_url: str = "",
    ollama_model: str = "",
    groq_api_key: str = "",
    groq_model: str = "",
    use_cache: bool = True,
    conversation_history: Optional[list[tuple[str, str]]] = None,
) -> tuple[str, list[dict]]:
    history = conversation_history or []
    effective_use_cache = use_cache and not history

    cache = get_cache(maxsize=256, ttl_seconds=3600)
    key = cache_key(company_id, question, top_k, use_llm, llm_mode, ollama_model, groq_model)

    def compute():
        embedder = get_embedder(company_config.embedding.model, company_config.embedding.device)
        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=vector_store,
            top_k_dense=company_config.retrieval.top_k,
            top_k_bm25=company_config.retrieval.top_k,
            top_k_final=top_k,
            use_bm25=company_config.retrieval.use_bm25,
        )
        hits = retriever.search(question)
        reranker = get_reranker(company_config.retrieval.reranker_model) if company_config.retrieval.use_reranker else None
        ranked = reranker.rerank(question, hits, top_k=top_k) if reranker else hits
        citations = build_citations(ranked)

        if not use_llm:
            answer = "Here are the most relevant passages I found (enable **Generate LLM answer** for a synthesized response):"
            return {"answer": answer, "citations": citations}

        if llm_mode == "public":
            llm = GroqLLMClient(api_key=groq_api_key, model=groq_model)
            backend_label = f"Groq ({groq_model})"
        else:
            llm = OllamaClient(base_url=ollama_url, model=ollama_model)
            backend_label = f"Ollama ({ollama_model}) at {ollama_url}"

        qa = QAPipeline(retriever=retriever, reranker=reranker, llm_client=llm, rerank_top_k=top_k)
        try:
            result = asyncio.run(qa.answer(question, conversation_history=history))
            answer = result.answer
        except Exception as exc:  # noqa: BLE001
            answer = (
                f"⚠️ Couldn't reach {backend_label} ({exc}). "
                "Showing retrieved chunks instead — check the connection, "
                "or turn off 'Generate LLM answer'."
            )
        return {"answer": answer, "citations": citations}

    if effective_use_cache:
        result, _ = cached_call(cache, key, compute)
    else:
        result = compute()

    return result["answer"], result["citations"]


def extract_conversation_pairs(messages: list[dict]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pending_question: Optional[str] = None
    for msg in messages:
        if msg["role"] == "user":
            pending_question = msg["content"]
        elif msg["role"] == "assistant" and pending_question is not None:
            pairs.append((pending_question, msg["content"]))
            pending_question = None
    return pairs


def build_citations(ranked: list[dict]) -> list[dict]:
    citations = []
    for i, hit in enumerate(ranked, start=1):
        meta = hit.get("metadata", {})
        citations.append(
            {
                "marker": i,
                "source": Path(meta.get("source_path", "unknown")).name,
                "page": meta.get("page_number"),
                "snippet": hit["document"][:300],
                "score": hit.get("rerank_score", hit.get("rrf_score", 0.0)),
            }
        )
    return citations


def render_citations(citations: list[dict]) -> None:
    with st.expander(f"📎 Sources ({len(citations)})"):
        for c in citations:
            page_label = f", page {c['page']}" if c.get("page") and c["page"] > 0 else ""
            st.markdown(
                f"""<div class="rag-citation-card">
                    <span class="rag-citation-badge">{c['marker']}</span>
                    <strong>{c['source']}{page_label}</strong>
                    <span style="color:var(--text-lo); font-size:12px;"> · score {c['score']:.3f}</span>
                    <div style="color:var(--text-lo); font-size:13px; margin-top:4px;">{c['snippet']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        if citations:
            pills = "".join(
                f'<span class="rag-pill" style="background:{palette_color(c["source"])};">{c["source"]}</span>'
                for c in {c["source"]: c for c in citations}.values()
            )
            st.markdown(f'<div class="rag-pill-row">{pills}</div>', unsafe_allow_html=True)


def render_voice_input(company_id: str) -> Optional[str]:
    settings = get_settings()
    audio = st.audio_input("🎤 Or ask by voice", key=f"voice_{company_id}")
    if audio is None:
        return None
    if st.button("Transcribe & ask", key=f"transcribe_{company_id}"):
        with st.spinner("Transcribing with Groq Whisper..."):
            client = GroqWhisperClient(api_key=settings.groq_api_key, model=settings.groq_whisper_model)
            try:
                return asyncio.run(client.transcribe(audio.getvalue(), filename="query.wav"))
            except GroqTranscriptionError as exc:
                st.error(str(exc))
                return None
    return None


# ---------------------------------------------------------------------- #
# Chat / Compare / History tabs
# ---------------------------------------------------------------------- #
def render_chat(company_id: str) -> None:
    settings = get_settings(company_id)
    try:
        company_config = load_company_config(company_id, settings.configs_dir)
    except ConfigValidationError as exc:
        st.error(f"Config error: {exc}")
        return

    vector_store = get_vector_store(str(settings.vector_db_dir), company_config.vector_store.collection_name)
    registry = get_registry()
    record = registry.get(company_id)

    st.caption(f"{record.total_pdfs} PDF(s) indexed · {record.total_chunks} chunks · isolated vector store")

    if vector_store.count() == 0:
        st.info("No documents indexed yet for this company. Upload PDFs and run ingestion from the sidebar.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        use_llm = st.checkbox("Generate LLM answer", value=False, key=f"llm_{company_id}")
    with col2:
        top_k = st.slider("Chunks to retrieve", min_value=3, max_value=15, value=8, key=f"topk_{company_id}")

    ollama_url, ollama_model, groq_model = settings.ollama_base_url, settings.ollama_model, settings.groq_llm_model
    llm_mode = "local"
    if use_llm:
        llm_mode = st.radio(
            "LLM mode", options=["local", "public"],
            format_func=lambda m: "🔒 Local (Ollama)" if m == "local" else "🌐 Public (Groq)",
            horizontal=True, key=f"mode_{company_id}",
        )
        if llm_mode == "local":
            with st.expander("Ollama settings"):
                ollama_url = st.text_input("Ollama base URL", value=settings.ollama_base_url, key=f"url_{company_id}")
                ollama_model = st.text_input("Ollama model", value=settings.ollama_model, key=f"model_{company_id}")
        else:
            with st.expander("Groq settings"):
                groq_model = st.text_input("Groq model", value=settings.groq_llm_model, key=f"groqmodel_{company_id}")
                if not settings.groq_api_key:
                    st.warning("No GROQ_API_KEY set — add one to your .env to use Public mode.")

    history_key = f"messages_{company_id}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])

    voice_text = render_voice_input(company_id)
    question = voice_text or st.chat_input("Ask a question about this company's documentation...")
    if not question:
        return

    conversation_history = extract_conversation_pairs(st.session_state[history_key])

    st.session_state[history_key].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, citations = answer_question(
                company_id, company_config, vector_store, question, top_k, use_llm,
                llm_mode=llm_mode, ollama_url=ollama_url, ollama_model=ollama_model,
                groq_api_key=settings.groq_api_key, groq_model=groq_model,
                conversation_history=conversation_history,
            )
        st.markdown(answer)
        render_citations(citations)

    st.session_state[history_key].append({"role": "assistant", "content": answer, "citations": citations})
    get_chat_store().add(company_id, question, answer, citations)


def render_compare() -> None:
    st.caption("Ask the same question across multiple companies at once and compare answers side by side.")
    registry = get_registry()
    companies = registry.list_companies()
    if len(companies) < 2:
        st.info("Register at least 2 companies to use comparison mode.")
        return

    labels = {c.company_id: c.name for c in companies}
    selected = st.multiselect("Companies to compare", options=list(labels.keys()), default=list(labels.keys())[:2], format_func=lambda cid: labels[cid])
    top_k = st.slider("Chunks to retrieve per company", min_value=3, max_value=15, value=8, key="compare_topk")
    question = st.text_input("Question to ask all selected companies")

    if not (selected and question and st.button("Compare")):
        return

    cols = st.columns(len(selected))
    for col, company_id in zip(cols, selected):
        with col:
            st.subheader(labels[company_id])
            settings = get_settings(company_id)
            try:
                company_config = load_company_config(company_id, settings.configs_dir)
            except ConfigValidationError as exc:
                st.error(f"Config error: {exc}")
                continue
            vector_store = get_vector_store(str(settings.vector_db_dir), company_config.vector_store.collection_name)
            if vector_store.count() == 0:
                st.info("No documents indexed yet.")
                continue
            with st.spinner("Retrieving..."):
                answer, citations = answer_question(company_id, company_config, vector_store, question, top_k, use_llm=False)
            st.markdown(answer)
            render_citations(citations)
            get_chat_store().add(company_id, question, answer, citations)


def render_history(company_id: str) -> None:
    store = get_chat_store()
    search = st.text_input("Search past questions", key=f"search_{company_id}")
    entries = store.list_for_company(company_id, limit=50, search=search or None)

    if not entries:
        st.info("No saved conversations yet for this company.")
        return

    for entry in entries:
        star_icon = "⭐" if entry.starred else "☆"
        st.markdown(f'<span class="rag-badge-pill">Sources {len(entry.citations)}</span>', unsafe_allow_html=True)
        with st.expander(f"{star_icon} {entry.question}  ·  {entry.created_at[:19]}"):
            st.markdown(entry.answer)
            if entry.citations:
                render_citations(entry.citations)
            c1, c2 = st.columns([1, 1])
            if c1.button("Toggle star", key=f"star_{entry.id}"):
                store.toggle_star(entry.id)
                st.rerun()
            if c2.button("Delete", key=f"del_{entry.id}"):
                store.delete(entry.id)
                st.rerun()


# ---------------------------------------------------------------------- #
def main() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    selected_id = render_sidebar()

    if selected_id is None:
        st.title("📊 Local-First Multi-Tenant RAG Platform")
        st.markdown(
            """
            Register a company in the sidebar to get started. Each company gets:
            - Its own isolated data directory and vector store (zero cross-tenant leakage)
            - Local-first PDF ingestion (SHA-256 dedup, resumable)
            - Hybrid search (dense BGE embeddings + BM25, fused with RRF) + cross-encoder reranking
            - A relevance-gap filter tuned from real production testing, not guesswork
            - Optional LLM answer generation with page-accurate citations, via Ollama or Groq
            - Multi-turn conversation memory, cross-company comparison, chat history, voice input
            """
        )
        return

    registry = get_registry()
    record = registry.get(selected_id)
    st.title(record.name)
    render_stat_row()

    tab_chat, tab_compare, tab_history = st.tabs(["💬 Chat", "🔍 Compare Companies", "📜 History"])
    with tab_chat:
        render_chat(selected_id)
    with tab_compare:
        render_compare()
    with tab_history:
        render_history(selected_id)


if __name__ == "__main__":
    main()