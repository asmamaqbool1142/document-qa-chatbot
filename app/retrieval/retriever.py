"""Query-time retrieval: embed the question, search, apply the relevance threshold.

The relevance threshold is a *heuristic*. A chunk scoring above it is not
guaranteed to contain the answer, and a chunk below it may still be relevant.
It mainly stops clearly unrelated questions from reaching the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models.domain import RetrievedChunk
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class RetrievalResult:
    kept: list[RetrievedChunk]  # above the threshold, best first
    candidates: list[RetrievedChunk]  # everything the store returned (for evaluation/debugging)
    min_score: float


class Retriever:
    def __init__(self, embedder: Embedder, store: ChromaVectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(
        self,
        question: str,
        top_k: int,
        min_score: float,
        document_ids: list[str] | None = None,
    ) -> RetrievalResult:
        query_vector = self._embedder.embed_query(question)
        candidates = self._store.query(query_vector, top_k=top_k, document_ids=document_ids)
        kept = [c for c in candidates if c.score >= min_score]
        return RetrievalResult(kept=kept, candidates=candidates, min_score=min_score)
