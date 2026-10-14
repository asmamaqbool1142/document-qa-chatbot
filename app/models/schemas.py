"""Pydantic request/response schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.models.domain import AnswerStatus, FileType


class ErrorResponse(BaseModel):
    error: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable explanation")
    request_id: str | None = None
    document_id: str | None = Field(
        default=None, description="Set for duplicate uploads: the existing document"
    )


class ComponentStatus(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' or 'degraded'")
    version: str
    embedding: ComponentStatus
    vector_store: ComponentStatus
    llm: ComponentStatus
    llm_provider: str
    llm_is_external: bool = Field(
        description="True if questions and document excerpts are sent to a remote service"
    )
    document_count: int


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    file_type: FileType
    size_bytes: int
    page_count: int
    chunk_count: int
    uploaded_at: str
    embedding_model: str


class UploadResponse(DocumentResponse):
    warnings: list[str] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class DeleteResponse(BaseModel):
    document_id: str
    deleted_chunks: int


class ChunkResponse(BaseModel):
    chunk_id: str
    chunk_index: int
    page_number: int | None
    text: str


class ChunkListResponse(BaseModel):
    document_id: str
    total: int
    offset: int
    limit: int
    chunks: list[ChunkResponse]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    document_ids: list[str] | None = Field(
        default=None, max_length=100, description="Restrict search to these documents"
    )

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if len(cleaned) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return cleaned


class CitationResponse(BaseModel):
    label: str = Field(description="Source label used in the answer, e.g. 'S1'")
    document_id: str
    filename: str
    page_number: int | None
    chunk_id: str
    score: float
    excerpt: str
    explicitly_cited: bool = Field(
        description="True if the answer text referenced this source label"
    )


class ChatResponse(BaseModel):
    answer: str
    status: AnswerStatus
    citations: list[CitationResponse]
    retrieved_count: int
    provider: str
    model: str
    latency_ms: int
