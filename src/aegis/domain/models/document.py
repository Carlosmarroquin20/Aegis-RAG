"""
Domain model for retrieved documents.
Metadata is typed as a flat string dict to remain vector-store-agnostic.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A chunk of text retrieved from the vector store, with provenance metadata."""

    id: str
    content: str = Field(..., min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)
    # Cosine similarity score; populated by the vector store adapter.
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = {"frozen": True}
