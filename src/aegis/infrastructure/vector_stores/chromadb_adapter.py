"""
ChromaDB vector store adapter.

Uses ChromaDB's HTTP client to support both local (embedded) and
remote (containerized) deployments without code changes.
Embedding is handled externally via SentenceTransformers so the
adapter stays decoupled from any specific embedding model.
"""
from __future__ import annotations

import structlog

from aegis.domain.models.document import Document
from aegis.domain.ports.vector_store import VectorStorePort

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
        self._collection: object | None = None

    async def initialize(self) -> None:
        """Creates the ChromaDB HTTP client and ensures the collection exists."""
        import chromadb  # Deferred import: chromadb is optional at module load.

        client = chromadb.HttpClient(host=self._host, port=self._port)
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "vector_store.initialized",
            collection=self._collection_name,
            host=self._host,
            port=self._port,
        )

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[Document]:
        self._assert_initialized()
        results = self._collection.query(  # type: ignore[union-attr]
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
                    metadata={k: str(v) for k, v in metadata.items()},
                    relevance_score=round(similarity, 4),
                )
            )

        logger.debug("vector_store.search_complete", k=k, returned=len(documents))
        return documents

    async def add_documents(self, documents: list[Document]) -> None:
        self._assert_initialized()
        self._collection.upsert(  # type: ignore[union-attr]
            ids=[doc.id for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents],  # type: ignore[arg-type]
        )
        logger.info("vector_store.documents_added", count=len(documents))

    async def delete_documents(self, ids: list[str]) -> None:
        self._assert_initialized()
        self._collection.delete(ids=ids)  # type: ignore[union-attr]
        logger.info("vector_store.documents_deleted", count=len(ids))

    async def health_check(self) -> bool:
        try:
            import chromadb

            client = chromadb.HttpClient(host=self._host, port=self._port)
            client.heartbeat()
            return True
        except Exception:
            return False

    def _assert_initialized(self) -> None:
        if self._collection is None:
            raise RuntimeError(
                "ChromaDBAdapter.initialize() must be called before use. "
                "Wire this call into the application lifespan event."
            )
