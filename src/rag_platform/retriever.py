"""
Hybrid retrieval: dense vector search (semantic) + BM25 (exact keyword
match), fused with Reciprocal Rank Fusion (RRF).

RRF, not a raw score blend, because cosine similarity and BM25 scores live
on incompatible scales — RRF fuses on rank position, sidestepping that.

Multi-query expansion: bare questions often miss relevant chunks because
the user's vocabulary doesn't match the docs. Generate 2-3 synonym variants
of the question, retrieve for each, then merge all results via RRF. This
recalls more chunks than single-query retrieval, especially when the
question uses different terminology than the documentation.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from rag_platform.embedder import Embedder
from rag_platform.vector_store import VectorStore
from rag_platform.query_expander import expand_query


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """In-memory BM25 index, rebuilt from whatever's currently in the vector store."""

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._meta: list[dict[str, Any]] = []

    def build(self, documents: list[dict[str, Any]]) -> None:
        self._ids = [d["id"] for d in documents]
        self._docs = [d["document"] for d in documents]
        self._meta = [d.get("metadata", {}) for d in documents]
        tokenized = [_tokenize(doc) for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def is_built(self) -> bool:
        return self._bm25 is not None

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        if self._bm25 is None or not self._docs:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"id": self._ids[i], "document": self._docs[i], "metadata": self._meta[i], "score": float(scores[i])}
            for i in ranked
            if scores[i] > 0
        ]


class HybridRetriever:
    """
    Args:
        embedder: Query/document embedder for the dense side.
        vector_store: This tenant's isolated ChromaDB wrapper.
        top_k_dense / top_k_bm25: Candidates pulled from each side before fusion.
        top_k_final: Candidates returned after RRF fusion.
        rrf_k: RRF constant (higher = flatter weighting of rank position; 60 is the common default).
        use_bm25: If False, skip BM25 entirely and return dense-only results (still RRF-ranked for consistency).
        use_multi_query: If True, expand query to variants and merge results via RRF (better recall).
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k_dense: int = 20,
        top_k_bm25: int = 20,
        top_k_final: int = 5,
        rrf_k: int = 60,
        use_bm25: bool = True,
        use_multi_query: bool = True,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = BM25Index()
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.top_k_final = top_k_final
        self.rrf_k = rrf_k
        self.use_bm25 = use_bm25
        self.use_multi_query = use_multi_query

    def refresh_bm25(self) -> None:
        self.bm25_index.build(self.vector_store.get_all_documents())

    def search(self, query: str, where: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        # Multi-query expansion: get variants
        if self.use_multi_query:
            variants = expand_query(query)
        else:
            variants = [query]

        # Retrieve for each variant and merge via RRF
        all_dense_lists = []
        all_bm25_lists = []

        for variant in variants:
            # Dense search
            query_vector = self.embedder.embed_query(variant)
            dense_hits = self.vector_store.query(query_vector, top_k=self.top_k_dense, where=where)
            all_dense_lists.append(dense_hits)

            # BM25 search
            if self.use_bm25:
                if not self.bm25_index.is_built():
                    self.refresh_bm25()
                bm25_hits = self.bm25_index.search(variant, top_k=self.top_k_bm25)
                all_bm25_lists.append(bm25_hits)

        # Merge all variants via RRF
        fused = self._multi_variant_rrf(all_dense_lists, all_bm25_lists)
        return fused[: self.top_k_final]

    def _multi_variant_rrf(
        self,
        dense_lists: list[list[dict[str, Any]]],
        bm25_lists: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """
        Reciprocal Rank Fusion across multiple query variants and search methods.
        Each (variant, method) combination contributes to the final score.
        """
        scores: dict[str, float] = {}
        payload: dict[str, dict[str, Any]] = {}

        # Score from all dense variants
        for dense_hits in dense_lists:
            for rank, hit in enumerate(dense_hits):
                hit_id = hit["id"]
                scores[hit_id] = scores.get(hit_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                payload[hit_id] = hit

        # Score from all BM25 variants
        for bm25_hits in bm25_lists:
            for rank, hit in enumerate(bm25_hits):
                hit_id = hit["id"]
                scores[hit_id] = scores.get(hit_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                payload.setdefault(hit_id, hit)

        # Sort by RRF score
        ranked_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

        results = []
        for hit_id in ranked_ids:
            item = dict(payload[hit_id])
            item["rrf_score"] = scores[hit_id]
            results.append(item)

        return results

    def _reciprocal_rank_fusion(
        self, dense_hits: list[dict[str, Any]], bm25_hits: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Legacy RRF (kept for backward compatibility, not used by default)."""
        return self._multi_variant_rrf([dense_hits], [bm25_hits])
