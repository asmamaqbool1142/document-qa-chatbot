"""Map exceptions to JSON error responses. Clients never see stack traces."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import AppError, DuplicateDocumentError
from app.models.schemas import ErrorResponse

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _respond(request: Request, status: int, code: str, message: str, **extra: str) -> JSONResponse:
    body = ErrorResponse(error=code, message=message, request_id=_request_id(request), **extra)
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True))


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        extra = {}
        if isinstance(exc, DuplicateDocumentError):
            extra["document_id"] = exc.document_id
        logger.info("%s -> %s (%s)", request.url.path, exc.status_code, exc.code)
        return _respond(request, exc.status_code, exc.code, exc.message, **extra)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Report which fields failed and why, without echoing submitted values back.
        problems = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
            problems.append(f"{loc}: {err.get('msg', 'invalid')}")
        return _respond(request, 422, "validation_error", "; ".join(problems) or "Invalid request")

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _respond(request, exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return _respond(
            request,
            500,
            "internal_error",
            "An unexpected error occurred. Check the server logs using the request ID.",
        )
