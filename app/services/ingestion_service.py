"""Document lifecycle: validate -> extract -> chunk -> embed -> index -> register.

Uploaded bytes are processed in memory. By default (STORE_UPLOADS=false) the
original file is *not* kept: only the extracted text chunks, their vectors and
a registry row are persisted. With STORE_UPLOADS=true a single copy is written
to ``data/uploads/<document_id>.<ext>`` (an internal name, never the user's
filename) and removed when the document is deleted.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.core.errors import DocumentNotFoundError, DuplicateDocumentError
from app.ingestion.chunking import chunk_pages
from app.ingestion.extractors import extract_text
from app.ingestion.validation import detect_file_type, sanitize_filename, validate_content
from app.models.domain import Chunk, DocumentRecord
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import ChromaVectorStore
from app.services.registry import DocumentRegistry

logger = logging.getLogger(__name__)


def compute_document_id(content_hash: str) -> str:
    """Stable ID derived from content: the same bytes always get the same ID."""
    return f"doc_{content_hash[:16]}"


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: ChromaVectorStore,
        registry: DocumentRegistry,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._registry = registry
        # Serialises the duplicate check + write so two identical uploads racing
        # each other can't both be indexed.
        self._write_lock = threading.Lock()

    def ingest(self, raw_filename: str | None, data: bytes) -> tuple[DocumentRecord, list[str]]:
        filename = sanitize_filename(raw_filename)
        file_type = detect_file_type(filename)
        validate_content(data, file_type, self._settings.max_upload_bytes)

        content_hash = hashlib.sha256(data).hexdigest()
        existing = self._registry.find_by_hash(content_hash)
        if existing:
            raise DuplicateDocumentError(
                f"This file was already uploaded as '{existing.filename}'.", existing.document_id
            )

        extracted = extract_text(data, file_type, max_pages=self._settings.max_pdf_pages)
        document_id = compute_document_id(content_hash)
        chunks = chunk_pages(
            extracted.pages, document_id, self._settings.chunk_size, self._settings.chunk_overlap
        )
        embeddings = self._embedder.embed_documents([c.text for c in chunks])
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            content_hash=content_hash,
            size_bytes=len(data),
            page_count=extracted.page_count,
            chunk_count=len(chunks),
            uploaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
            embedding_model=self._embedder.model_id,
        )

        with self._write_lock:
            if self._registry.find_by_hash(content_hash):
                raise DuplicateDocumentError("This file was already uploaded.", document_id)
            self._index(record, chunks, embeddings)
            self._store_original(record, data)

        logger.info(
            "Indexed document %s (%s, %d pages, %d chunks)",
            document_id,
            file_type.value,
            record.page_count,
            record.chunk_count,
        )
        return record, extracted.warnings

    def _index(
        self, record: DocumentRecord, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        base_metadata = {
            "filename": record.filename,
            "file_type": record.file_type.value,
            "uploaded_at": record.uploaded_at,
        }
        try:
            self._store.add_chunks(chunks, embeddings, base_metadata)
            self._registry.add(record)
        except Exception:
            # Keep the two stores consistent: no vectors without a registry row.
            logger.exception("Indexing failed for %s; rolling back", record.document_id)
            self._store.delete_document(record.document_id)
            raise

    def _upload_path(self, record: DocumentRecord) -> Path:
        # Internal name only; the user's filename never touches the filesystem.
        return self._settings.uploads_dir / f"{record.document_id}.{record.file_type.value}"

    def _store_original(self, record: DocumentRecord, data: bytes) -> None:
        if not self._settings.store_uploads:
            return
        path = self._upload_path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def delete(self, document_id: str) -> int:
        record = self._registry.get(document_id)
        if record is None:
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
        with self._write_lock:
            deleted = self._store.delete_document(document_id)
            self._registry.delete(document_id)
            path = self._upload_path(record)
            path.unlink(missing_ok=True)
        logger.info("Deleted document %s (%d chunks)", document_id, deleted)
        return deleted

    def list_documents(self) -> list[DocumentRecord]:
        return self._registry.list()

    def get_document(self, document_id: str) -> DocumentRecord:
        record = self._registry.get(document_id)
        if record is None:
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
        return record
