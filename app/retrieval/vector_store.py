"""Persistent vector storage backed by ChromaDB.

Only this module knows about Chroma. The rest of the app talks to
``ChromaVectorStore`` through plain Python types, so swapping the database
later means rewriting one file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.errors import EmbeddingModelMismatchError
from app.models.domain import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "document_chunks"
_UPSERT_BATCH = 256


class ChromaVectorStore:
    def __init__(self, path: Path, embedding_model_id: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": embedding_model_id},
            embedding_function=None,  # we always pass our own vectors
        )
        stored = (self._collection.metadata or {}).get("embedding_model")
        if stored and stored != embedding_model_id:
            raise EmbeddingModelMismatchError(
                f"The vector database at '{path}' was built with embedding model '{stored}', "
                f"but the app is configured to use '{embedding_model_id}'. Vectors from "
                "different models are not comparable. Either restore the original "
                "EMBEDDING_* settings, point DATA_DIR at a new directory, or delete the "
                "existing data and re-upload your documents."
            )
        self.embedding_model_id = embedding_model_id

    # --- writes ----------------------------------------------------------
    def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]], base_metadata: dict[str, Any]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for i in range(0, len(chunks), _UPSERT_BATCH):
            batch = chunks[i : i + _UPSERT_BATCH]
            metadatas: list[dict[str, Any]] = []
            for c in batch:
                meta = {
                    **base_metadata,
                    "document_id": c.document_id,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                }
                if c.page_number is not None:  # Chroma metadata can't hold None reliably
                    meta["page_number"] = c.page_number
                metadatas.append(meta)
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=cast(Any, embeddings[i : i + _UPSERT_BATCH]),
                documents=[c.text for c in batch],
                metadatas=cast(Any, metadatas),
            )

    def delete_document(self, document_id: str) -> int:
        result = self._collection.delete(where={"document_id": document_id})
        deleted = result.get("deleted") if isinstance(result, dict) else None
        return int(deleted) if deleted is not None else 0

    # --- reads -----------------------------------------------------------
    def count(self) -> int:
        return int(self._collection.count())

    def count_for_document(self, document_id: str) -> int:
        result = self._collection.get(where={"document_id": document_id}, include=[])
        return len(result["ids"])

    def get_document_chunks(
        self, document_id: str, limit: int, offset: int
    ) -> list[tuple[str, int, int | None, str]]:
        """Return (chunk_id, chunk_index, page_number, text) sorted by chunk index."""
        result = self._collection.get(
            where={"document_id": document_id}, include=["documents", "metadatas"]
        )
        rows = []
        docs = result["documents"] or []
        metas = result["metadatas"] or []
        for chunk_id, text, meta in zip(result["ids"], docs, metas, strict=True):
            page = meta.get("page_number")
            rows.append(
                (
                    chunk_id,
                    int(cast(int, meta["chunk_index"])),
                    int(cast(int, page)) if page is not None else None,
                    text or "",
                )
            )
        rows.sort(key=lambda r: r[1])
        return rows[offset : offset + limit]

    def query(
        self, embedding: list[float], top_k: int, document_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        where: dict[str, Any] | None = None
        if document_ids:
            where = {"document_id": {"$in": list(document_ids)}}
        result = self._collection.query(
            query_embeddings=cast(Any, [embedding]),
            n_results=top_k,
            where=cast(Any, where),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        docs = (result["documents"] or [[]])[0]
        metas = (result["metadatas"] or [[]])[0]
        dists = (result["distances"] or [[]])[0]
        hits: list[RetrievedChunk] = []
        for rank, (chunk_id, text, meta, dist) in enumerate(
            zip(ids, docs, metas, dists, strict=True), start=1
        ):
            page = meta.get("page_number")
            hits.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=str(meta["document_id"]),
                    filename=str(meta.get("filename", "unknown")),
                    page_number=int(cast(int, page)) if page is not None else None,
                    chunk_index=int(cast(int, meta["chunk_index"])),
                    text=text or "",
                    score=round(1.0 - float(dist), 4),  # cosine distance -> similarity
                    rank=rank,
                )
            )
        return hits

    def healthcheck(self) -> None:
        self._client.heartbeat()
