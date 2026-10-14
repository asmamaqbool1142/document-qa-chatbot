"""FastAPI application entry point.

Run locally:  uvicorn app.main:app --reload
API docs:     http://localhost:8000/docs
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.errors import register_error_handlers
from app.api.middleware import request_logging_middleware
from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.services.container import Container, build_container


def create_app(
    settings: Settings | None = None,
    container_factory: Callable[[Settings], Container] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    factory = container_factory or build_container

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = factory(settings)
        yield

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Upload PDF/TXT documents and ask questions answered from their content, "
            "with source citations. Demo-grade: no authentication is built in."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    app.middleware("http")(request_logging_middleware)
    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(chat_router)
    return app


app = create_app()
