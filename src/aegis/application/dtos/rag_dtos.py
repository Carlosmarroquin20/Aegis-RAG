"""
Data Transfer Objects for the RAG query use case.

These live in the application layer and are shared between the use case and
the interface layer. They intentionally do not import domain internals so
the interface layer never needs to depend on the domain directly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Inbound DTO from the API consumer."""

    query: str = Field(..., min_length=1, max_length=8192, description="Natural language question.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve.")
    collection: str | None = Field(
        default=None,
        description="Target collection name. Defaults to the configured collection.",
    )


class SourceDocument(BaseModel):
    """Provenance metadata attached to each retrieved evidence chunk."""

    content_preview: str  # First 200 chars; full content is never echoed back.
    metadata: dict[str, str]
    relevance_score: float


class QueryResponse(BaseModel):
    """Outbound DTO returned to the API consumer."""

    answer: str
    sources: list[SourceDocument]
    # The query hash enables audit trail linkage without echoing the raw query.
    query_hash: str
    threat_level: str  # "CLEAN" | "SUSPICIOUS" — always CLEAN in non-strict mode reach here


class SecurityRejectionResponse(BaseModel):
    """Returned when the SecurityGateway blocks a query."""

    detail: str
    query_hash: str
    threat_level: str = "BLOCKED"
