"""Application configuration.

All settings come from environment variables (optionally loaded from a ``.env``
file). Nothing secret is hardcoded. ``get_settings()`` is cached so every module
sees the same values; tests build their own ``Settings`` objects instead.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(StrEnum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HASHING = "hashing"


class LLMProvider(StrEnum):
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"
    EXTRACTIVE = "extractive"


# Similarity scores live on different scales for different embedders, so the
# default relevance threshold depends on which one is active. These are heuristics,
# not guarantees: 0.12 for hashing keeps every answerable question in the sample
# evaluation set; 0.30 for MiniLM is a common starting point that has NOT been tuned
# on this project's data. Re-tune with the evaluation's threshold sweep.
DEFAULT_MIN_SCORE = {
    EmbeddingProvider.SENTENCE_TRANSFORMERS: 0.30,
    EmbeddingProvider.HASHING: 0.12,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_parse_none_str="",  # "MIN_RELEVANCE_SCORE=" in .env means "not set"
    )

    # --- General -----------------------------------------------------------
    app_name: str = "RAG Knowledge Assistant"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])
    enable_debug_endpoints: bool = False

    # --- Uploads -----------------------------------------------------------
    max_upload_mb: float = Field(default=20.0, gt=0, le=200)
    max_pdf_pages: int = Field(default=500, ge=1)
    store_uploads: bool = False  # keep original files on disk? (text is always indexed)

    # --- Chunking ----------------------------------------------------------
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0)

    # --- Embeddings --------------------------------------------------------
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    hashing_dimensions: int = Field(default=1024, ge=64, le=16384)

    # --- Retrieval ---------------------------------------------------------
    top_k: int = Field(default=5, ge=1, le=20)
    min_relevance_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    max_context_chars: int = Field(default=6000, ge=500, le=50000)

    # --- Answer generation -------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_tokens: int = Field(default=600, ge=16, le=8192)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    # --- Derived values ----------------------------------------------------
    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.sqlite3"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def effective_min_score(self) -> float:
        if self.min_relevance_score is not None:
            return self.min_relevance_score
        return DEFAULT_MIN_SCORE[self.embedding_provider]

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.chroma_dir, self.uploads_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
