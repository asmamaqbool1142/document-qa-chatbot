"""The question-answering pipeline, independent of HTTP.

retrieve -> threshold -> bounded context -> generate -> validate citations
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.core.config import Settings
from app.generation.base import AnswerGenerator
from app.generation.citations import build_citations, strip_unknown_markers
from app.models.domain import AnswerStatus, Citation
from app.retrieval.context import build_context
from app.retrieval.retriever import RetrievalResult, Retriever

logger = logging.getLogger(__name__)

NO_EVIDENCE_MESSAGE = (
    "I couldn't find enough information in the uploaded documents to answer this question."
)


@dataclass(frozen=True)
class ChatResult:
    answer: str
    status: AnswerStatus
    citations: list[Citation]
    retrieval: RetrievalResult
    provider: str
    model: str
    latency_ms: int


class ChatService:
    def __init__(
        self, settings: Settings, retriever: Retriever, generator: AnswerGenerator
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._generator = generator

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
        document_ids: list[str] | None = None,
    ) -> ChatResult:
        started = time.perf_counter()
        k = top_k or self._settings.top_k
        threshold = self._settings.effective_min_score if min_score is None else min_score

        retrieval = self._retriever.retrieve(question, k, threshold, document_ids)

        def finish(answer: str, status: AnswerStatus, citations: list[Citation]) -> ChatResult:
            elapsed = int((time.perf_counter() - started) * 1000)
            logger.info(
                "chat: candidates=%d kept=%d status=%s provider=%s latency_ms=%d",
                len(retrieval.candidates),
                len(retrieval.kept),
                status.value,
                self._generator.name,
                elapsed,
            )
            return ChatResult(
                answer,
                status,
                citations,
                retrieval,
                self._generator.name,
                self._generator.model,
                elapsed,
            )

        if not retrieval.kept:
            # Nothing cleared the relevance threshold: don't call the LLM at all.
            return finish(NO_EVIDENCE_MESSAGE, AnswerStatus.INSUFFICIENT_EVIDENCE, [])

        sources = build_context(retrieval.kept, self._settings.max_context_chars)
        generated = self._generator.generate(question, sources)
        if generated.insufficient_evidence:
            return finish(NO_EVIDENCE_MESSAGE, AnswerStatus.INSUFFICIENT_EVIDENCE, [])

        answer = strip_unknown_markers(generated.text, {s.label for s in sources}).strip()
        citations = build_citations(answer, question, sources)
        return finish(answer, AnswerStatus.ANSWERED, citations)
