"""
DTOs for the document ingestion API.

UploadedFile is populated from the multipart form data in the route handler.
It carries the raw bytes and metadata needed by the ingestion use case.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class UploadedFile(BaseModel):
    """
    Represents a single file upload received at the API boundary.
    Bytes are held in memory only for the duration of the request.
    """

    filename: str
    content: bytes
    # Content-Type from the HTTP header; treated as a hint only.
    # Actual MIME type is determined from magic bytes by the ParserRegistry.
    declared_mime_type: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class IngestRequest(BaseModel):
    """Parameters accompanying a document upload."""

    collection: str | None = Field(
        default=None,
        description="Target collection. Defaults to the configured default collection.",
    )
    chunk_size: int = Field(
        default=512,
        ge=128,
        le=4096,
        description="Target character count per chunk.",
    )
    overlap: int = Field(
        default=64,
        ge=0,
        le=512,
        description="Overlap characters between consecutive chunks.",
    )


class IngestResponse(BaseModel):
    """Returned after successful ingestion."""

    source: str
    chunks_created: int
    duplicates_skipped: int
    warnings: list[str] = Field(default_factory=list)
    collection: str


class DocumentListItem(BaseModel):
    """Summary of an indexed document returned by GET /documents."""

    id: str
    source: str
    chunk_index: str
    page: str | None = None
    section: str | None = None
    content_preview: str  # First 120 characters


class DeleteResponse(BaseModel):
    deleted_ids: list[str]
    count: int
