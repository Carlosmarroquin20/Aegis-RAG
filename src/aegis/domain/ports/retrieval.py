"""
Retrieval ports.

The query pipeline asks a ``RetrieverPort`` for the chunks to put in front of the
model. How they are found is behind it: dense similarity from the vector store
alone, or that combined with a lexical index and re-scored by a reranker. The two
optional stages have ports of their own so that BM25 can be replaced by a store's
native full-text search, and a local cross-encoder by a hosted reranking API,
without the use case or the retriever noticing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aegis.domain.models.document import Document


class RetrieverPort(ABC):
    """Finds the chunks most relevant to a query."""

    @abstractmethod
    async def retrieve(self, query: str, k: int) -> list[Document]:
        """
        Returns at most ``k`` chunks, most relevant first. ``relevance_score`` is
        the score of the last stage that ranked them, in [0, 1].
        """


class LexicalIndexPort(ABC):
    """Keyword search over the same chunks the vector store holds."""

    @abstractmethod
    async def search(self, query: str, k: int) -> list[Document]:
        """
        Returns at most ``k`` chunks that share terms with the query, best match
        first. Chunks with no term in common are not returned, so the list can be
        shorter than ``k`` or empty.
        """


class RerankerPort(ABC):
    """Re-scores candidate chunks by reading each one together with the query."""

    @abstractmethod
    async def rerank(self, query: str, documents: list[Document], k: int) -> list[Document]:
        """
        Returns the ``k`` best of ``documents``, best first, each with its
        reranker score in [0, 1] as ``relevance_score``.
        """
