"""
Document management endpoints.

POST   /api/v1/documents          — Upload and index one or more files.
GET    /api/v1/documents          — List indexed document chunks (paginated).
DELETE /api/v1/documents          — Bulk delete by list of chunk IDs.
DELETE /api/v1/documents/{doc_id} — Delete a single chunk.

File uploads use multipart/form-data. The API validates file size at the
framework level (before the use case) to reject oversized payloads early,
reducing the attack surface for decompression bombs.
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
from aegis.interface.api.dependencies import get_chromadb_adapter, get_ingest_use_case

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# FastAPI enforces this limit at the ASGI level, before the route handler runs.
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
    content = await file.read()

    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

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
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentListItem]:
    """
    Returns a paginated list of all indexed chunks.
    Full content is never echoed; only a 120-character preview is returned
    to avoid inadvertently leaking sensitive document content via the API.
    """
    adapter = get_chromadb_adapter()
    if adapter._collection is None:  # noqa: SLF001
        raise HTTPException(status_code=503, detail="Vector store not initialized.")

    from typing import Any, cast

    collection: Any = cast(Any, adapter._collection)  # noqa: SLF001
    result = collection.get(
        limit=limit,
        offset=offset,
        include=["documents", "metadatas"],
    )

    items: list[DocumentListItem] = []
    for doc_id, content, meta in zip(
        result.get("ids", []),
        result.get("documents", []),
        result.get("metadatas", []),
        strict=False,
    ):
        meta = meta or {}
        items.append(
            DocumentListItem(
                id=doc_id,
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", "0"),
                page=meta.get("page"),
                section=meta.get("section"),
                content_preview=(content or "")[:120],
            )
        )
    return items


@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    summary="Delete a single document chunk by ID",
)
async def delete_document(doc_id: str) -> DeleteResponse:
    adapter = get_chromadb_adapter()
    await adapter.delete_documents([doc_id])
    return DeleteResponse(deleted_ids=[doc_id], count=1)


@router.delete(
    "",
    response_model=DeleteResponse,
    summary="Bulk delete document chunks by IDs",
)
async def bulk_delete_documents(ids: list[str]) -> DeleteResponse:
    if not ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ids list is empty.")
    if len(ids) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bulk delete is limited to 500 IDs per request.",
        )
    adapter = get_chromadb_adapter()
    await adapter.delete_documents(ids)
    return DeleteResponse(deleted_ids=ids, count=len(ids))
