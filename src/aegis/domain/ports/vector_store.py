"""
Vector store port (abstract interface).

Concrete adapters (ChromaDB, pgvector, Weaviate, etc.) implement this protocol
so the application layer remains decoupled from any specific vector database.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aegis.domain.models.document import Document


class VectorStorePort(ABC):
    """Hexagonal port for semantic document retrieval and indexing."""

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[Document]:
        """
        Returns the top-k documents most semantically similar to query.
        Documents below score_threshold are excluded.
        """

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> None:
        """Upserts documents into the collection. Idempotent on duplicate ids."""

    @abstractmethod
    async def delete_documents(self, ids: list[str]) -> None:
        """Removes documents by id. No-op for unknown ids."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Returns True if the store is reachable and the collection exists."""
