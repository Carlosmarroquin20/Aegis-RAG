"""
Data Transfer Objects for the RAG query use case.

These live in the application layer and are shared between the use case and
the interface layer. They intentionally do not import domain internals so
the interface layer never needs to depend on the domain directly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Passages retrieved when the caller does not say. Ten, not five: on the example
# corpus five miss a needed document for one multi-document question in seven,
# and answers judged correct went from 7 to 16 of 29 (ADR 0029).
DEFAULT_TOP_K = 10


class QueryRequest(BaseModel):
    """Inbound DTO from the API consumer."""

    query: str = Field(..., min_length=1, max_length=8192, description="Natural language question.")
    top_k: int = Field(
        default=DEFAULT_TOP_K, ge=1, le=20, description="Number of documents to retrieve."
    )


class SourceDocument(BaseModel):
    """Provenance metadata attached to each retrieved evidence chunk."""

    content_preview: str  # First 200 chars; full content is never echoed back.
    metadata: dict[str, str]
    relevance_score: float


class FaithfulnessReport(BaseModel):
    """How well the answer is supported by the retrieved context (FAITHFULNESS_CHECK)."""

    score: float = Field(description="Share of the answer's claims the context supports, 0 to 1.")
    abstained: bool = Field(description="The answer declined for lack of context; not scored.")
    unsupported: list[str] = Field(
        description="Sentences of the answer the context does not support."
    )


class QueryResponse(BaseModel):
    """Outbound DTO returned to the API consumer."""

    answer: str
    sources: list[SourceDocument]
    # The query hash enables audit trail linkage without echoing the raw query.
    query_hash: str
    threat_level: str  # "CLEAN" | "SUSPICIOUS" — always CLEAN in non-strict mode reach here
    # Present only when the deployment runs the faithfulness check.
    faithfulness: FaithfulnessReport | None = None


class StreamStart(BaseModel):
    """First event of a streamed answer: what was retrieved, before any text."""

    query_hash: str
    threat_level: str
    sources: list[SourceDocument]


class StreamDelta(BaseModel):
    """Sanitized text to append to the answer shown so far."""

    text: str


class StreamEnd(BaseModel):
    """
    Last event of a streamed answer. ``answer`` is the complete, sanitized text
    and is authoritative: a client replaces what it assembled from the deltas
    with it.
    """

    answer: str
    truncated: bool
    faithfulness: FaithfulnessReport | None = None


class StreamError(BaseModel):
    """Ends a stream that failed after text was sent. The client discards what it showed."""

    code: str
    message: str


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
