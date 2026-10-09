"""
ChromaDB vector store adapter.

Uses ChromaDB's HTTP client to support both local (embedded) and
remote (containerized) deployments without code changes.
Embedding is handled externally via SentenceTransformers so the
adapter stays decoupled from any specific embedding model.

Concurrency: the ChromaDB client is synchronous, and a query or upsert also runs
the embedding model in-process (CPU-bound, tens to hundreds of milliseconds).
Every call is therefore dispatched to a worker thread. Calling the client
directly from these coroutines would stall the event loop — and with it health
probes, metrics scrapes and every other in-flight request — for the duration.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import structlog

from aegis.domain.models.document import Document
from aegis.domain.ports.vector_store import VectorStorePort, VectorStoreUnavailableError

logger = structlog.get_logger(__name__)


class ChromaDBAdapter(VectorStorePort):
    """
    Wraps the ChromaDB HTTP client.
    Instantiation is deferred to avoid import-time I/O; call initialize() before use.
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        embedding_function: object,  # chromadb.EmbeddingFunction protocol
    ) -> None:
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        # Vendor types stay untyped here: chromadb is imported lazily and must not
        # leak above this module.
        self._client: Any | None = None
        self._collection: Any | None = None

    async def initialize(self) -> None:
        """Creates the ChromaDB HTTP client and ensures the collection exists."""
        self._client, self._collection = await asyncio.to_thread(self._connect)
        logger.info(
            "vector_store.initialized",
            collection=self._collection_name,
            host=self._host,
            port=self._port,
        )

    def _connect(self) -> tuple[Any, Any]:
        import chromadb  # Deferred import: chromadb is optional at module load.

        client = chromadb.HttpClient(host=self._host, port=self._port)
        collection = client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )
        return client, collection

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[Document]:
        collection = self._require_collection()
        results = await asyncio.to_thread(
            collection.query,
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        documents: list[Document] = []
        for doc_id, content, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=False,
        ):
            # ChromaDB returns cosine distances (0 = identical, 2 = opposite).
            # Convert to similarity score in [0, 1].
            similarity = max(0.0, 1.0 - distance / 2.0)
            if similarity < score_threshold:
                continue
            documents.append(
                Document(
                    id=doc_id,
                    content=content,
                    metadata=_as_str_metadata(metadata),
                    relevance_score=round(similarity, 4),
                )
            )

        logger.debug("vector_store.search_complete", k=k, returned=len(documents))
        return documents

    async def add_documents(self, documents: list[Document]) -> None:
        collection = self._require_collection()
        await asyncio.to_thread(
            collection.upsert,
            ids=[doc.id for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )
        logger.info("vector_store.documents_added", count=len(documents))

    async def delete_documents(self, ids: list[str]) -> None:
        collection = self._require_collection()
        await asyncio.to_thread(collection.delete, ids=ids)
        logger.info("vector_store.documents_deleted", count=len(ids))

    async def existing_ids(self, ids: list[str]) -> set[str]:
        collection = self._require_collection()
        # get() silently omits unknown ids; include=[] skips payload transfer.
        result = await asyncio.to_thread(collection.get, ids=ids, include=[])
        return set(result.get("ids") or [])

    async def list_documents(self, limit: int = 20, offset: int = 0) -> list[Document]:
        collection = self._require_collection()
        result = await asyncio.to_thread(
            collection.get,
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )
        return [
            Document(id=doc_id, content=content, metadata=_as_str_metadata(metadata))
            for doc_id, content, metadata in zip(
                result.get("ids") or [],
                result.get("documents") or [],
                result.get("metadatas") or [],
                strict=False,
            )
            # Records written by another client may carry no document text.
            if content
        ]

    async def health_check(self) -> bool:
        # Reuses the client built at startup. /ready is unauthenticated and polled
        # by orchestrators; opening a new connection pool on every probe would let
        # anyone who can reach it churn sockets on demand.
        if self._client is None:
            return False
        try:
            await asyncio.to_thread(self._client.heartbeat)
            return True
        except Exception:
            return False

    def _require_collection(self) -> Any:
        if self._collection is None:
            raise VectorStoreUnavailableError(
                "ChromaDBAdapter.initialize() must be called before use. "
                "Wire this call into the application lifespan event."
            )
        return self._collection


def _as_str_metadata(metadata: Mapping[str, object] | None) -> dict[str, str]:
    """Chroma allows scalar metadata values and null metadata; the domain wants str."""
    return {key: str(value) for key, value in (metadata or {}).items()}
