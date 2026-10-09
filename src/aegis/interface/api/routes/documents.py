"""
Document management endpoints.

POST   /api/v1/documents          — Upload and index one or more files.
GET    /api/v1/documents          — List indexed document chunks (paginated).
DELETE /api/v1/documents          — Bulk delete by list of chunk IDs.
DELETE /api/v1/documents/{doc_id} — Delete a single chunk.

File uploads use multipart/form-data. The route rejects an oversized part before
reading it into memory, and the use case enforces the same ceiling again before
parsing. Neither bounds what the multipart parser spools to disk while receiving
the request: cap the request body at the reverse proxy as well (the bundled
nginx config sets ``client_max_body_size`` to match).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from aegis.application.dtos.ingestion_dtos import (
    DeleteResponse,
    DocumentListItem,
    IngestRequest,
    IngestResponse,
    UploadedFile,
)
from aegis.application.use_cases.ingest_documents import (
    FileTooLargeError,
    IngestDocumentsUseCase,
    UnsupportedFileTypeError,
)
from aegis.domain.ports.document_parser import ParseError
from aegis.domain.ports.vector_store import VectorStorePort, VectorStoreUnavailableError
from aegis.interface.api.dependencies import get_ingest_use_case, get_vector_store

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# Keep in sync with client_max_body_size in frontend/nginx.conf.
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
    responses={
        400: {"description": "Unsupported file type or corrupt content."},
        413: {"description": "File exceeds the 50 MB upload limit."},
    },
)
async def ingest_document(
    file: Annotated[UploadFile, File(description="Document to index (TXT, MD, PDF, DOCX).")],
    use_case: Annotated[IngestDocumentsUseCase, Depends(get_ingest_use_case)],
    collection: Annotated[
        str | None,
        Form(description="Target collection. Defaults to the system default."),
    ] = None,
    chunk_size: Annotated[int, Form(ge=128, le=4096)] = 512,
    overlap: Annotated[int, Form(ge=0, le=512)] = 64,
) -> IngestResponse:
    too_large = HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=f"File exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
    )
    # The multipart parser has already spooled the part to a temporary file and
    # recorded its size, so an oversized upload is refused here without ever being
    # copied into process memory.
    if file.size is not None and file.size > _MAX_UPLOAD_BYTES:
        raise too_large

    # Bounded read as a second guard for the case where the size is unknown: at
    # most one byte past the limit is buffered, which is enough to detect it.
    content = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(content) > _MAX_UPLOAD_BYTES:
        raise too_large

    uploaded = UploadedFile(
        filename=file.filename or "unknown",
        content=content,
        declared_mime_type=file.content_type,
    )
    request = IngestRequest(collection=collection, chunk_size=chunk_size, overlap=overlap)

    try:
        return await use_case.execute(uploaded, request)
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc
    except ParseError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "",
    response_model=list[DocumentListItem],
    summary="List indexed document chunks",
)
async def list_documents(
    store: Annotated[VectorStorePort, Depends(get_vector_store)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentListItem]:
    """
    Returns a paginated list of all indexed chunks.
    Full content is never echoed; only a 120-character preview is returned
    to avoid inadvertently leaking sensitive document content via the API.
    """
    try:
        chunks = await store.list_documents(limit=limit, offset=offset)
    except VectorStoreUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized.",
        ) from exc

    return [
        DocumentListItem(
            id=chunk.id,
            source=chunk.metadata.get("source", "unknown"),
            chunk_index=chunk.metadata.get("chunk_index", "0"),
            page=chunk.metadata.get("page"),
            section=chunk.metadata.get("section"),
            content_preview=chunk.content[:120],
        )
        for chunk in chunks
    ]


@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    summary="Delete a single document chunk by ID",
)
async def delete_document(
    doc_id: str,
    store: Annotated[VectorStorePort, Depends(get_vector_store)],
) -> DeleteResponse:
    await store.delete_documents([doc_id])
    return DeleteResponse(deleted_ids=[doc_id], count=1)


@router.delete(
    "",
    response_model=DeleteResponse,
    summary="Bulk delete document chunks by IDs",
)
async def bulk_delete_documents(
    ids: list[str],
    store: Annotated[VectorStorePort, Depends(get_vector_store)],
) -> DeleteResponse:
    if not ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ids list is empty.")
    if len(ids) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bulk delete is limited to 500 IDs per request.",
        )
    await store.delete_documents(ids)
    return DeleteResponse(deleted_ids=ids, count=len(ids))
