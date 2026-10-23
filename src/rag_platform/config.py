"""
Central configuration for the RAG platform.

Every path in this file is a function of `company_id`. There is no shared
mutable state and no module-level singleton — a singleton built at import
time would be wrong the moment a second company shows up, since CLI scripts
only learn their company_id *after* argparse runs. Call get_settings()
explicitly wherever you need a Settings instance.

Path layout:
    data/companies/{company_id}/raw_pdfs/       source PDFs
    data/companies/{company_id}/clean/          extracted markdown, one file per PDF (sha256.md)
    data/companies/{company_id}/manifest.jsonl  append-only ingestion manifest
    data/companies/{company_id}/checkpoint.json resumable run state
    data/companies/{company_id}/vector_store/   ChromaDB (embedded, persistent)
    data/registry.json                          global — tracks all companies
    configs/{company_id}.yaml                   global — per-company YAML config
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()
_VALID_COMPANY_ID_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_-")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # --- Tenancy ------------------------------------------------------------
    company_id: str = "default"

    # --- Global (not company-scoped) paths -----------------------------------
    project_root: Path = Field(default=Path(__file__).resolve().parents[2])
    configs_dir: Optional[Path] = Field(default=None)
    registry_path: Optional[Path] = Field(default=None)

    # --- Per-company paths, resolved in model_post_init ----------------------
    data_dir: Optional[Path] = Field(default=None)
    raw_pdfs_dir: Optional[Path] = Field(default=None)
    clean_dir: Optional[Path] = Field(default=None)
    manifest_path: Optional[Path] = Field(default=None)
    checkpoint_path: Optional[Path] = Field(default=None)
    vector_db_dir: Optional[Path] = Field(default=None)

    # --- Chunking --------------------------------------------------------------
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_length: int = 100
    preserve_code_blocks: bool = True

    # --- Embedding ---------------------------------------------------------------
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    vector_collection_name: Optional[str] = None

    # --- Retrieval -----------------------------------------------------------------
    hybrid_top_k_dense: int = 20
    hybrid_top_k_bm25: int = 20
    hybrid_top_k_final: int = 5
    rrf_k: int = 60
    use_bm25: bool = True
    use_reranker: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LLM ----------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # --- Groq (optional, free-tier voice transcription only) ------------------
    groq_api_key: str = ""  # set via GROQ_API_KEY env or RAG_GROQ_API_KEY
    groq_whisper_model: str = "whisper-large-v3-turbo"
    groq_llm_model: str = "llama-3.3-70b-versatile"  # legacy; SenseNova is the Public LLM now

    # --- SenseNova (Public LLM mode — replaced Groq 2026-10-23) ---------------
    # No key needed on machines with the connected custom.sensenova credential
    # (the client uses it automatically). For deployed demos, set the key via
    # SENSENOVA_API_KEY env or Streamlit secrets.
    sensenova_api_key: str = ""  # set via SENSENOVA_API_KEY env or RAG_SENSENOVA_API_KEY
    sensenova_llm_model: str = "sensenova-6.8-flash-lite"

    # --- Caching ---------------------------------------------------------------
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 256

    # --- Chat history ------------------------------------------------------------
    chat_history_db_path: Optional[Path] = Field(default=None)  # resolved in model_post_init

    # --- PDF ingestion -----------------------------------------------------------------
    max_pdf_workers: int = 4  # process pool size for CPU-bound PDF text extraction

    @field_validator("company_id")
    @classmethod
    def _validate_company_id(cls, v: str) -> str:
        """
        company_id becomes a directory name and a ChromaDB collection name suffix,
        so it must be filesystem-safe and collection-name-safe. Reject anything
        that isn't lowercase alnum/underscore/hyphen — silently "sanitizing" it
        instead would mean the id you pass on the CLI doesn't match the id on
        disk, which is a much nastier bug to track down later.
        """
        if not v:
            raise ValueError("company_id must not be empty")
        if not set(v).issubset(_VALID_COMPANY_ID_CHARS):
            raise ValueError(
                f"company_id {v!r} contains invalid characters — use only "
                "lowercase letters, digits, underscore, hyphen"
            )
        if v in {".", ".."}:
            raise ValueError(f"company_id {v!r} is not allowed")
        return v

    def model_post_init(self, __context) -> None:
        import os

        if not self.groq_api_key:
            object.__setattr__(self, "groq_api_key", os.environ.get("GROQ_API_KEY", ""))
        if not self.sensenova_api_key:
            object.__setattr__(
                self,
                "sensenova_api_key",
                os.environ.get("SENSENOVA_API_KEY", os.environ.get("RAG_SENSENOVA_API_KEY", "")),
            )
        if self.chat_history_db_path is None:
            object.__setattr__(self, "chat_history_db_path", self.project_root / "data" / "chat_history.sqlite3")

        if self.configs_dir is None:
            object.__setattr__(self, "configs_dir", self.project_root / "configs")

        # VECTORMIND_DATA_DIR overrides the data root (used for the Streamlit
        # Cloud demo, which ships a pre-built vector store under demo/).
        data_root = Path(os.environ.get("VECTORMIND_DATA_DIR", self.project_root / "data"))
        if self.registry_path is None:
            object.__setattr__(self, "registry_path", data_root / "registry.json")

        base = data_root / "companies" / self.company_id

        if self.data_dir is None:
            object.__setattr__(self, "data_dir", base)
        if self.raw_pdfs_dir is None:
            object.__setattr__(self, "raw_pdfs_dir", base / "raw_pdfs")
        if self.clean_dir is None:
            object.__setattr__(self, "clean_dir", base / "clean")
        if self.manifest_path is None:
            object.__setattr__(self, "manifest_path", base / "manifest.jsonl")
        if self.checkpoint_path is None:
            object.__setattr__(self, "checkpoint_path", base / "checkpoint.json")
        if self.vector_db_dir is None:
            object.__setattr__(self, "vector_db_dir", base / "vector_store")
        if self.vector_collection_name is None:
            object.__setattr__(self, "vector_collection_name", f"{self.company_id}_docs")

    def ensure_dirs(self) -> None:
        """Create every directory this tenant needs. Safe to call repeatedly."""
        for path in (self.data_dir, self.raw_pdfs_dir, self.clean_dir, self.vector_db_dir, self.configs_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def config_path(self) -> Path:
        """YAML config file this tenant should load, e.g. configs/acme_corp.yaml."""
        return self.configs_dir / f"{self.company_id}.yaml"


def get_settings(company_id: Optional[str] = None) -> Settings:
    """
    Build a Settings instance, optionally scoped to a company.

    Args:
        company_id: Tenant identifier. Defaults to "default" if omitted —
            useful for tests and for any tooling that hasn't been made
            company-aware yet.

    Returns:
        A fresh Settings instance with every path resolved for that tenant.

    Example:
        >>> s = get_settings("acme_corp")
        >>> s.vector_db_dir
        PosixPath('.../data/companies/acme_corp/vector_store')
    """
    if company_id is not None:
        return Settings(company_id=company_id)
    return Settings()
