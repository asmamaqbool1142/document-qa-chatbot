from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.dependencies import ContainerDep
from app.models.schemas import ComponentStatus, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service and dependency status")
def health(container: ContainerDep) -> HealthResponse:
    errors = container.startup_errors
    embedding = ComponentStatus(
        ok=container.embedder is not None,
        detail=container.embedder.model_id if container.embedder else errors.get("embedding", ""),
    )
    if container.store is not None:
        try:
            container.store.healthcheck()
            store = ComponentStatus(ok=True, detail=f"{container.store.count()} chunks indexed")
        except Exception as exc:  # report, don't crash the health probe
            store = ComponentStatus(ok=False, detail=f"Vector store error ({type(exc).__name__})")
    else:
        store = ComponentStatus(ok=False, detail=errors.get("vector_store", "not initialised"))

    llm_status = container.generator.check()
    llm = ComponentStatus(ok=llm_status.ok, detail=llm_status.detail)
    doc_count = container.registry.count() if container.registry else 0
    return HealthResponse(
        # The API is usable for indexing without an LLM, so a missing LLM is
        # "degraded" rather than down. Docker's healthcheck only needs HTTP 200.
        status="ok" if (embedding.ok and store.ok and llm.ok) else "degraded",
        version=__version__,
        embedding=embedding,
        vector_store=store,
        llm=llm,
        llm_provider=container.generator.name,
        llm_is_external=container.generator.is_external,
        document_count=doc_count,
    )
