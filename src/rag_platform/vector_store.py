"""
Vector store wrapper around ChromaDB (embedded, persistent mode).

Design decision — per-company DB, not shared DB + metadata filter: each
company gets its own ChromaDB PersistentClient pointed at
data/companies/{company_id}/vector_store/. This is a stronger isolation
guarantee than "one shared DB, always filter by company_id in queries" —
a single missed `where` clause in a shared setup leaks data across
tenants; a wrong company_id here just means you're pointed at an empty or
nonexistent database, not someone else's data. The cost is one ChromaDB
instance per company, which is cheap at the scale (dozens of companies,
100s of PDFs each) this platform targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


class VectorStore:
    """
    Args:
        persist_dir: This tenant's isolated vector store directory
            (Settings.vector_db_dir) — never shared across company_id values.
        collection_name: Collection name within that tenant's own DB
            (Settings.vector_collection_name, e.g. "acme_corp_docs").
    """

    def __init__(self, persist_dir: Path, collection_name: str):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            import chromadb

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def upsert(
        self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict[str, Any]]
    ) -> None:
        if not ids:
            return
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(
        self, query_embedding: list[float], top_k: int = 10, where: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        result = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
        hits: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append({"id": i, "document": doc, "metadata": meta, "distance": dist, "score": 1.0 - dist})
        return hits

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    def delete_by_source(self, source_path: str) -> None:
        """Remove every chunk belonging to one source file — used when a PDF is re-processed after a hash change."""
        self.collection.delete(where={"source_path": source_path})

    def delete_stale_versions(self, source_path: str, current_sha256: str) -> None:
        """Remove chunks of previous versions of the same file, keeping the
        just-upserted current version. Called AFTER a successful upsert so a
        mid-ingestion failure can never leave the file with zero chunks."""
        self.collection.delete(
            where={
                "$and": [
                    {"source_path": source_path},
                    {"file_sha256": {"$ne": current_sha256}},
                ]
            }
        )

    def count(self) -> int:
        return self.collection.count()

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Every id/document/metadata triple — used to (re)build the BM25 index."""
        result = self.collection.get()
        return [
            {"id": i, "document": doc, "metadata": meta}
            for i, doc, meta in zip(result.get("ids", []), result.get("documents", []), result.get("metadatas", []))
        ]
