from __future__ import annotations

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import ContainerDep
from app.models.schemas import ChatRequest, ChatResponse, CitationResponse, ErrorResponse

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={code: {"model": ErrorResponse} for code in (422, 503)},
    summary="Ask a question about the uploaded documents",
)
async def chat(body: ChatRequest, container: ContainerDep) -> ChatResponse:
    service = container.require_chat()
    result = await run_in_threadpool(
        service.ask, body.question, body.top_k, body.min_score, body.document_ids
    )
    return ChatResponse(
        answer=result.answer,
        status=result.status,
        citations=[
            CitationResponse(
                label=c.label,
                document_id=c.document_id,
                filename=c.filename,
                page_number=c.page_number,
                chunk_id=c.chunk_id,
                score=c.score,
                excerpt=c.excerpt,
                explicitly_cited=c.explicitly_cited,
            )
            for c in result.citations
        ],
        retrieved_count=len(result.retrieval.kept),
        provider=result.provider,
        model=result.model,
        latency_ms=result.latency_ms,
    )
