"""Wire all components together once at startup (simple dependency injection).

If the embedding model or vector store can't be initialised, the API still
starts so /health can explain what's wrong; document and chat endpoints then
return 503 with the reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.core.config import Settings
from app.core.errors import AppError, ProviderUnavailableError
from app.generation.base import AnswerGenerator
from app.generation.factory import build_generator
from app.retrieval.embeddings import Embedder, build_embedder
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import ChromaVectorStore
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService
from app.services.registry import DocumentRegistry

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    generator: AnswerGenerator
    embedder: Embedder | None = None
    store: ChromaVectorStore | None = None
    registry: DocumentRegistry | None = None
    ingestion: IngestionService | None = None
    chat: ChatService | None = None
    startup_errors: dict[str, str] = field(default_factory=dict)

    def require_ingestion(self) -> IngestionService:
        if self.ingestion is None:
            raise ProviderUnavailableError(self._unavailable_reason())
        return self.ingestion

    def require_chat(self) -> ChatService:
        if self.chat is None:
            raise ProviderUnavailableError(self._unavailable_reason())
        return self.chat

    def _unavailable_reason(self) -> str:
        reasons = "; ".join(f"{k}: {v}" for k, v in self.startup_errors.items())
        return f"The document index is unavailable ({reasons or 'not initialised'})."


def build_container(
    settings: Settings,
    embedder: Embedder | None = None,
    generator: AnswerGenerator | None = None,
) -> Container:
    """Build everything. Tests pass their own ``embedder``/``generator``."""
    settings.ensure_dirs()
    container = Container(settings=settings, generator=generator or build_generator(settings))

    try:
        container.embedder = embedder or build_embedder(settings)
    except AppError as exc:
        logger.error("Embedding model unavailable: %s", exc.message)
        container.startup_errors["embedding"] = exc.message
        return container

    try:
        container.store = ChromaVectorStore(settings.chroma_dir, container.embedder.model_id)
        container.registry = DocumentRegistry(settings.registry_path)
    except AppError as exc:
        logger.error("Vector store unavailable: %s", exc.message)
        container.startup_errors["vector_store"] = exc.message
        return container

    container.ingestion = IngestionService(
        settings, container.embedder, container.store, container.registry
    )
    container.chat = ChatService(
        settings, Retriever(container.embedder, container.store), container.generator
    )
    status = container.generator.check()
    log = logger.info if status.ok else logger.warning
    log("LLM provider '%s': %s", container.generator.name, status.detail)
    if container.generator.is_external:
        logger.warning(
            "LLM provider is an external service: questions and retrieved document "
            "excerpts will be sent to it."
        )
    return container
