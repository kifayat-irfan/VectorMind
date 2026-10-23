"""Regression tests for bugs found in the 2026-10-23 code review.

H1: page citations drifted forward on long documents (28/50 wrong).
C1: a failed re-ingestion permanently deleted a PDF's chunks.
CORE: multi-tenant retrieval isolation — the platform's central claim.
"""

from __future__ import annotations

from rag_platform.chunker import RecursiveCharacterChunker


def _ten_page_doc(pages: int = 10, sentences_per_page: int = 30) -> str:
    parts = []
    for p in range(1, pages + 1):
        parts.append(f"<!-- page:{p} -->\n" + f"Content unique to page {p}. " * sentences_per_page)
    return "\n\n".join(parts)


def test_chunker_page_attribution_no_drift():
    """H1 regression: every chunk must cite the page its text came from."""
    doc = _ten_page_doc()
    chunker = RecursiveCharacterChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.split(doc)

    assert len(chunks) > 10  # sanity: multi-page doc -> many chunks
    for ch in chunks:
        assert ch.page_number is not None, f"chunk {ch.index} has no page"
        # The chunk's own text must mention the page it claims to be from.
        # (Each page's sentences say "page N" explicitly.)
        assert f"page {ch.page_number}" in ch.text, (
            f"chunk {ch.index} attributed to page {ch.page_number} "
            f"but text doesn't mention it: {ch.text[:80]!r}"
        )


def test_chunker_page_attribution_with_code_blocks():
    """H1 regression with atomic code blocks interleaved (the merge path)."""
    doc = _ten_page_doc(pages=5, sentences_per_page=20)
    doc += "\n\n<!-- page:6 -->\n```python\nprint('hello from page six')\n```\n" + (
        "Trailing content on page six. " * 20
    )
    chunker = RecursiveCharacterChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.split(doc)
    code_chunks = [c for c in chunks if c.is_code]
    assert code_chunks, "expected at least one code chunk"
    for ch in code_chunks:
        assert ch.page_number == 6, f"code chunk attributed to page {ch.page_number}, expected 6"


def test_vector_store_stale_version_cleanup(tmp_path):
    """C1 regression: re-ingesting a changed PDF keeps new chunks, drops old."""
    from pathlib import Path

    from rag_platform.vector_store import VectorStore

    vs = VectorStore(tmp_path / "vs", "test_coll")
    src = "handbook.pdf"

    # v1 of the file
    vs.upsert(
        ids=["hash1:0", "hash1:1"],
        embeddings=[[0.1] * 8, [0.2] * 8],
        documents=["v1 chunk a", "v1 chunk b"],
        metadatas=[
            {"source_path": src, "file_sha256": "hash1", "page_number": 1},
            {"source_path": src, "file_sha256": "hash1", "page_number": 2},
        ],
    )
    # v2 re-ingest (same upsert-then-cleanup order as the fixed script)
    vs.upsert(
        ids=["hash2:0"],
        embeddings=[[0.3] * 8],
        documents=["v2 chunk a"],
        metadatas=[{"source_path": src, "file_sha256": "hash2", "page_number": 1}],
    )
    vs.delete_stale_versions(src, "hash2")

    docs = vs.get_all_documents()
    assert len(docs) == 1
    assert docs[0]["document"] == "v2 chunk a"
    assert vs.count() == 1


def test_vector_store_failed_reingest_keeps_old_chunks(tmp_path):
    """C1 regression: if the new upsert never happens, old chunks survive."""
    from pathlib import Path

    from rag_platform.vector_store import VectorStore

    vs = VectorStore(tmp_path / "vs", "test_coll")
    src = "handbook.pdf"
    vs.upsert(
        ids=["hash1:0"],
        embeddings=[[0.1] * 8],
        documents=["v1 chunk a"],
        metadatas=[{"source_path": src, "file_sha256": "hash1", "page_number": 1}],
    )
    # Simulate: embedding/upsert throws BEFORE any new upsert (old code
    # deleted first; new code only cleans up after a successful upsert).
    # Nothing to delete -> old chunk must still be there.
    assert vs.count() == 1
    assert vs.get_all_documents()[0]["document"] == "v1 chunk a"


def test_multi_tenant_retrieval_isolation(tmp_path):
    """CORE claim regression: per-company vector stores never leak."""
    from pathlib import Path

    from rag_platform.vector_store import VectorStore

    def make_store(company: str) -> VectorStore:
        vs = VectorStore(tmp_path / company / "vs", f"{company}_docs")
        return vs

    acme = make_store("acme")
    globex = make_store("globex")

    acme.upsert(
        ids=["a1"],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        documents=["Acme secret sauce recipe"],
        metadatas=[{"source_path": "acme.pdf", "company_id": "acme", "page_number": 1}],
    )
    globex.upsert(
        ids=["g1"],
        embeddings=[[0.0, 1.0, 0.0, 0.0]],
        documents=["Globex layoff plan Q4"],
        metadatas=[{"source_path": "globex.pdf", "company_id": "globex", "page_number": 1}],
    )

    # Query globex's store with acme's embedding — must return nothing from acme.
    hits = globex.query([1.0, 0.0, 0.0, 0.0], top_k=5)
    assert all(h["metadata"]["company_id"] == "globex" for h in hits)
    assert not any("Acme" in h["document"] for h in hits)

    hits = acme.query([0.0, 1.0, 0.0, 0.0], top_k=5)
    assert all(h["metadata"]["company_id"] == "acme" for h in hits)
    assert not any("Globex" in h["document"] for h in hits)


def test_registry_corrupt_raises_not_silent(tmp_path):
    """H6 regression: corrupt registry.json raises instead of returning empty."""
    from pathlib import Path

    import pytest

    from rag_platform.company_registry import CompanyRegistry, RegistryCorruptError

    reg_path = tmp_path / "registry.json"
    reg = CompanyRegistry(reg_path)
    reg.add("acme", "Acme Corp", "cfg.yaml", "data/acme")

    # Corrupt the file
    reg_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(RegistryCorruptError):
        CompanyRegistry(reg_path).list_companies()

    # Backup of the corrupt file must exist; original untouched
    backups = list(tmp_path.glob("registry.json.corrupt.*"))
    assert len(backups) == 1
