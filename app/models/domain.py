"""Internal data structures passed between pipeline stages.

These are plain dataclasses (cheap, no validation overhead) because they are
built by our own code. Data crossing the HTTP boundary uses the Pydantic
schemas in ``app.models.schemas`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class FileType(StrEnum):
    PDF = "pdf"
    TXT = "txt"


@dataclass(frozen=True)
class PageText:
    """Text of one page. ``page_number`` is 1-based, or None for plain text files."""

    page_number: int | None
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    file_type: FileType
    pages: list[PageText]
    page_count: int
    warnings: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    page_number: int | None
    char_start: int
    char_end: int


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    filename: str
    file_type: FileType
    content_hash: str
    size_bytes: int
    page_count: int
    chunk_count: int
    uploaded_at: str  # ISO-8601 UTC
    embedding_model: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by the vector store, with its similarity score."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    text: str
    score: float  # cosine similarity in [-1, 1]; higher is more relevant
    rank: int  # 1-based position in the result list


@dataclass(frozen=True)
class ContextSource:
    """A retrieved chunk as it was placed into the prompt, labelled S1, S2, ..."""

    label: str
    chunk: RetrievedChunk
    text: str  # possibly truncated to respect the context budget


@dataclass(frozen=True)
class Citation:
    label: str
    document_id: str
    filename: str
    page_number: int | None
    chunk_id: str
    score: float
    excerpt: str
    explicitly_cited: bool


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class GeneratedAnswer:
    """Raw output of an answer generator before citation post-processing."""

    text: str
    insufficient_evidence: bool
    provider: str
    model: str
