"""Embeddings, vector store and retriever behaviour."""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar

import pytest

from app.core.errors import EmbeddingModelMismatchError
from app.models.domain import Chunk, RetrievedChunk
from app.retrieval.context import build_context
from app.retrieval.embeddings import HashingEmbedder, tokenize
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import ChromaVectorStore


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingEmbedder(256)
    v1 = emb.embed_query("Solar panels on the warehouse roof")
    v2 = HashingEmbedder(256).embed_documents(["Solar panels on the warehouse roof"])[0]
    assert v1 == v2
    assert len(v1) == 256
    assert math.isclose(sum(x * x for x in v1), 1.0, rel_tol=1e-9)


def test_hashing_embedder_similarity_ordering() -> None:
    emb = HashingEmbedder()
    q = emb.embed_query("annual leave days")
    related = emb.embed_query("Employees receive 22 days of paid annual leave.")
    unrelated = emb.embed_query("The RO membrane filters water under pressure.")
    assert _cos(q, related) > _cos(q, unrelated)


def test_hashing_embedder_empty_text_gives_zero_vector() -> None:
    assert not any(HashingEmbedder(64).embed_query("the of and"))


def test_tokenize_keeps_numbers_and_drops_stopwords() -> None:
    assert tokenize("The W300 costs 2.5 euros, and is NEW") == [
        "w300",
        "costs",
        "2.5",
        "euros",
        "new",
    ]


def _chunk(doc: str, idx: int, text: str, page: int | None = 1) -> Chunk:
    return Chunk(f"{doc}-{idx:05d}", doc, idx, text, page, 0, len(text))


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(tmp_path / "chroma", HashingEmbedder().model_id)


def _index(store: ChromaVectorStore, chunks: list[Chunk], filename: str = "f.txt") -> None:
    emb = HashingEmbedder()
    store.add_chunks(chunks, emb.embed_documents([c.text for c in chunks]), {"filename": filename})


def test_store_query_returns_ranked_hits_with_metadata(store: ChromaVectorStore) -> None:
    _index(
        store,
        [
            _chunk("doc_a", 0, "Parental leave is 16 weeks at full pay.", page=4),
            _chunk("doc_a", 1, "Passwords must be changed every 180 days.", page=None),
        ],
    )
    hits = store.query(HashingEmbedder().embed_query("How long is parental leave?"), top_k=2)
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].chunk_id == "doc_a-00000"
    assert hits[0].page_number == 4 and hits[0].filename == "f.txt"
    assert hits[1].page_number is None
    assert hits[0].score > hits[1].score


def test_store_filter_and_delete(store: ChromaVectorStore) -> None:
    _index(store, [_chunk("doc_a", 0, "alpha text"), _chunk("doc_b", 0, "alpha beta")])
    q = HashingEmbedder().embed_query("alpha")
    assert {h.document_id for h in store.query(q, 5, document_ids=["doc_b"])} == {"doc_b"}
    assert store.delete_document("doc_a") == 1
    assert store.count() == 1 and store.count_for_document("doc_a") == 0


def test_store_empty_query_returns_nothing(store: ChromaVectorStore) -> None:
    assert store.query(HashingEmbedder().embed_query("anything"), 3) == []


def test_store_persists_and_rejects_different_embedding_model(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    first = ChromaVectorStore(path, "model-a")
    first.add_chunks([_chunk("doc_a", 0, "x")], [[1.0, 0.0]], {"filename": "f"})
    assert ChromaVectorStore(path, "model-a").count() == 1  # persisted
    with pytest.raises(EmbeddingModelMismatchError, match="model-a"):
        ChromaVectorStore(path, "model-b")


def test_retriever_applies_threshold(store: ChromaVectorStore) -> None:
    _index(store, [_chunk("doc_a", 0, "Remote work is allowed 3 days per week.")])
    retriever = Retriever(HashingEmbedder(), store)
    relevant = retriever.retrieve("remote work days per week", top_k=3, min_score=0.2)
    assert len(relevant.kept) == 1
    irrelevant = retriever.retrieve("volcano eruption lava", top_k=3, min_score=0.2)
    assert irrelevant.kept == [] and len(irrelevant.candidates) == 1


def _retrieved(text: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(f"c{rank}", "d", "f.txt", None, rank, text, 0.5, rank)


def test_build_context_respects_budget_and_labels() -> None:
    chunks = [_retrieved("a" * 400, 1), _retrieved("b" * 400, 2), _retrieved("c" * 400, 3)]
    sources = build_context(chunks, max_chars=1000)
    assert [s.label for s in sources] == ["S1", "S2", "S3"]
    assert sum(len(s.text) for s in sources) <= 1000
    assert sources[2].text.endswith("[...]")
    assert len(build_context(chunks, max_chars=850)) == 2  # remainder too small to be useful


class _FakeSTModel:
    """Stands in for sentence_transformers.SentenceTransformer (no download needed)."""

    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, name: str, device: str = "cpu") -> None:
        if name == "missing/model":
            raise OSError("not found")
        self.name, self.device = name, device

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        _FakeSTModel.calls.append(kwargs)
        return [[1.0, 0.0] for _ in texts]


def test_sentence_transformer_embedder_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    from app.core.errors import ProviderUnavailableError
    from app.retrieval.embeddings import SentenceTransformerEmbedder

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _FakeSTModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    emb = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    assert emb.model_id == "sentence_transformers:sentence-transformers/all-MiniLM-L6-v2"
    assert emb.embed_documents([]) == []
    assert emb.embed_query("hello") == [1.0, 0.0]
    assert _FakeSTModel.calls[-1]["normalize_embeddings"] is True  # needed for cosine scores
    with pytest.raises(ProviderUnavailableError, match="Could not load embedding model"):
        SentenceTransformerEmbedder("missing/model")


def test_sentence_transformer_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from app.core.errors import ConfigurationError
    from app.retrieval.embeddings import SentenceTransformerEmbedder

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ConfigurationError, match="pip install"):
        SentenceTransformerEmbedder("any")
