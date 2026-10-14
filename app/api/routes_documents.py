from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import ContainerDep, DocumentIdPath
from app.core.errors import FileTooLargeError
from app.models.domain import DocumentRecord
from app.models.schemas import (
    ChunkListResponse,
    ChunkResponse,
    DeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    ErrorResponse,
    UploadResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])

_READ_BLOCK = 1024 * 1024
_ERRORS: dict[int | str, dict[str, object]] = {
    code: {"model": ErrorResponse} for code in (404, 409, 413, 415, 422, 503)
}


def _to_response(record: DocumentRecord) -> DocumentResponse:
    return DocumentResponse(
        document_id=record.document_id,
        filename=record.filename,
        file_type=record.file_type,
        size_bytes=record.size_bytes,
        page_count=record.page_count,
        chunk_count=record.chunk_count,
        uploaded_at=record.uploaded_at,
        embedding_model=record.embedding_model,
    )


async def _read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read the upload in blocks and stop as soon as it exceeds the limit."""
    buffer = bytearray()
    while block := await upload.read(_READ_BLOCK):
        buffer.extend(block)
        if len(buffer) > max_bytes:
            raise FileTooLargeError(
                f"File exceeds the maximum upload size of {max_bytes / (1024 * 1024):.1f} MB."
            )
    return bytes(buffer)


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_ERRORS,
    summary="Upload and index a PDF or TXT document",
)
async def upload_document(
    request: Request,
    container: ContainerDep,
    file: UploadFile = File(description="A .pdf or .txt file"),
) -> UploadResponse:
    service = container.require_ingestion()
    max_bytes = container.settings.max_upload_bytes
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes + 64 * 1024:
        raise FileTooLargeError(
            f"File exceeds the maximum upload size of {container.settings.max_upload_mb:.1f} MB."
        )
    data = await _read_limited(file, max_bytes)
    # Extraction and embedding are CPU-bound; run them in a worker thread so the
    # event loop keeps serving other requests.
    record, warnings = await run_in_threadpool(service.ingest, file.filename, data)
    return UploadResponse(**_to_response(record).model_dump(), warnings=warnings)


@router.get("", response_model=DocumentListResponse, summary="List indexed documents")
def list_documents(container: ContainerDep) -> DocumentListResponse:
    docs = [_to_response(r) for r in container.require_ingestion().list_documents()]
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    responses=_ERRORS,
    summary="Get one document's metadata",
)
def get_document(document_id: DocumentIdPath, container: ContainerDep) -> DocumentResponse:
    return _to_response(container.require_ingestion().get_document(document_id))


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    responses=_ERRORS,
    summary="Delete a document and all of its vectors",
)
def delete_document(document_id: DocumentIdPath, container: ContainerDep) -> DeleteResponse:
    deleted = container.require_ingestion().delete(document_id)
    return DeleteResponse(document_id=document_id, deleted_chunks=deleted)


@router.get(
    "/{document_id}/chunks",
    response_model=ChunkListResponse,
    responses=_ERRORS,
    summary="(Debug) Inspect a document's chunks. Disabled unless ENABLE_DEBUG_ENDPOINTS=true",
)
def list_chunks(
    document_id: DocumentIdPath,
    container: ContainerDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> ChunkListResponse:
    if not container.settings.enable_debug_endpoints:
        # 404 rather than 403 so the endpoint's existence isn't advertised.
        raise HTTPException(status_code=404, detail="Not Found")
    service = container.require_ingestion()
    record = service.get_document(document_id)
    assert container.store is not None  # guaranteed when ingestion is available
    rows = container.store.get_document_chunks(document_id, limit=limit, offset=offset)
    return ChunkListResponse(
        document_id=document_id,
        total=record.chunk_count,
        offset=offset,
        limit=limit,
        chunks=[
            ChunkResponse(chunk_id=c, chunk_index=i, page_number=p, text=t) for c, i, p, t in rows
        ],
    )
