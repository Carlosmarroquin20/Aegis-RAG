"""
Domain models for the document ingestion pipeline.

RawDocument is the output of a parser — pre-chunking, pre-embedding.
It carries provenance metadata that is preserved through chunking and indexing
so that the RAG pipeline can cite exact sources in generated answers.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SupportedMimeType(StrEnum):
    TEXT_PLAIN = "text/plain"
    TEXT_MARKDOWN = "text/markdown"
    APPLICATION_PDF = "application/pdf"
    APPLICATION_DOCX = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    @classmethod
    def values(cls) -> frozenset[str]:
        return frozenset(m.value for m in cls)


class RawDocument(BaseModel):
    """
    Single logical unit of text extracted from a source file.
    A source file may produce multiple RawDocuments (e.g., one per PDF page).
    Not yet chunked or embedded.
    """

    content: str = Field(..., min_length=1)
    source: str  # Filename or URI; used for citation metadata.
    page_number: int | None = None
    section: str | None = None
    # Free-form metadata propagated to the vector store and surfaced in API responses.
    metadata: dict[str, str] = Field(default_factory=dict)

    model_config = {"frozen": True}


class IngestionResult(BaseModel):
    """Summary returned to the caller after an ingestion job completes."""

    source: str
    chunks_created: int
    duplicates_skipped: int = 0
    # Populated when the ingestion partially fails (e.g., one page parse error).
    warnings: list[str] = Field(default_factory=list)
