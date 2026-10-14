"""SQLite registry of uploaded documents.

Chroma stores chunks and vectors; this small table stores one row per document
(filename, hash, counts, timestamps). Keeping it separate makes listing
documents and detecting duplicates cheap, without scanning every vector.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.models.domain import DocumentRecord, FileType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id     TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    file_type       TEXT NOT NULL,
    content_hash    TEXT NOT NULL UNIQUE,
    size_bytes      INTEGER NOT NULL,
    page_count      INTEGER NOT NULL,
    chunk_count     INTEGER NOT NULL,
    uploaded_at     TEXT NOT NULL,
    embedding_model TEXT NOT NULL
);
"""

_COLUMNS = (
    "document_id, filename, file_type, content_hash, size_bytes, page_count, "
    "chunk_count, uploaded_at, embedding_model"
)


class DocumentRegistry:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10)
        try:
            with conn:  # commits on success, rolls back on error
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _row_to_record(row: tuple[object, ...]) -> DocumentRecord:
        return DocumentRecord(
            document_id=str(row[0]),
            filename=str(row[1]),
            file_type=FileType(str(row[2])),
            content_hash=str(row[3]),
            size_bytes=int(str(row[4])),
            page_count=int(str(row[5])),
            chunk_count=int(str(row[6])),
            uploaded_at=str(row[7]),
            embedding_model=str(row[8]),
        )

    def add(self, record: DocumentRecord) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT INTO documents ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.document_id,
                    record.filename,
                    record.file_type.value,
                    record.content_hash,
                    record.size_bytes,
                    record.page_count,
                    record.chunk_count,
                    record.uploaded_at,
                    record.embedding_model,
                ),
            )

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def find_by_hash(self, content_hash: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list(self) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM documents ORDER BY uploaded_at DESC, filename"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def delete(self, document_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return cur.rowcount > 0

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
