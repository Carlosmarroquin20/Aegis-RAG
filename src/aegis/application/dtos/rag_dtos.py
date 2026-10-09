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


class SecurityRejectionDetail(BaseModel):
    """Why a query was refused, without disclosing which rule matched."""

    message: str
    # First 16 hex chars of the query's SHA-256: enough to find the audit log
    # entry, which carries the full hash and the triggered rules.
    query_hash: str
    threat_level: str  # "BLOCKED" | "SUSPICIOUS" (the latter only in strict mode)


class SecurityRejectionResponse(BaseModel):
    """
    Body of the HTTP 400 returned when the SecurityGateway blocks a query.

    The fields are nested under ``detail`` because the route raises an
    HTTPException, and FastAPI serializes its payload under that key.
    """

    detail: SecurityRejectionDetail
