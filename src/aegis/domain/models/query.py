"""
Domain models for query ingestion and sanitization.

RawQuery represents untrusted input at the API boundary.
SanitizedQuery is the output of the SecurityGateway — it is the only form
that may flow into the RAG pipeline.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RawQuery(BaseModel):
    """Untrusted user input. Must not reach the LLM without SecurityGateway evaluation."""

    text: str = Field(..., min_length=1, max_length=8192)

    @field_validator("text")
    @classmethod
    def strip_outer_whitespace(cls, v: str) -> str:
        return v.strip()

    @field_validator("text")
    @classmethod
    def reject_null_bytes(cls, v: str) -> str:
        # Null bytes are a common smuggling vector in multi-modal pipelines.
        if "\x00" in v:
            raise ValueError("Query must not contain null bytes.")
        return v


class SanitizedQuery(BaseModel):
    """
    Immutable, gateway-approved query.
    Carries the SHA-256 hash of the original input for audit trail linkage.
    """

    text: str
    query_hash: str  # hex digest of the original RawQuery.text

    model_config = {"frozen": True}
