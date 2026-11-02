"""
The retrieval stage of the query pipeline.

    dense (vector store) ─┐
                          ├─ reciprocal rank fusion ─→ reranker ─→ top k
    lexical (BM25) ───────┘

Dense embeddings match meaning but blur exact terms: a product code, a figure, a
proper noun or a word the embedding model rarely saw can rank below a chunk that
is merely about the same topic. A lexical index matches exactly those terms and
misses paraphrases. Running both and fusing their rankings recovers what either
alone drops. A cross-encoder then reads the query together with each candidate,
which a bi-encoder never does, and puts the best ones first.

Both extra stages are optional. With neither, this is plain dense retrieval and
behaves exactly as the pipeline did before they existed.
"""

from __future__ import annotations

import asyncio

from opentelemetry import trace

from aegis.domain.models.document import Document
from aegis.domain.ports.retrieval import LexicalIndexPort, RerankerPort, RetrieverPort
from aegis.domain.ports.vector_store import VectorStorePort
from aegis.domain.services.rank_fusion import DEFAULT_RRF_K, reciprocal_rank_fusion

tracer = trace.get_tracer("aegis")

DEFAULT_CANDIDATES = 20


class Retriever(RetrieverPort):
    """
    Dense retrieval, optionally fused with lexical search and reranked.

    ``candidates`` is how deep each first-stage retriever looks, and how many
    fused chunks the reranker reads. A larger pool lets a chunk that both first
    stages ranked poorly still be found, at the cost of reranker time, which
    grows linearly with it. It never drops below the ``k`` asked for.
    """

    def __init__(
        self,
        vector_store: VectorStorePort,
        *,
        lexical_index: LexicalIndexPort | None = None,
        reranker: RerankerPort | None = None,
        candidates: int = DEFAULT_CANDIDATES,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._reranker = reranker
        self._candidates = candidates
        self._rrf_k = rrf_k

    @property
    def mode(self) -> str:
        """The stages in use, as the retrieval configuration names them."""
        first = "hybrid" if self._lexical_index is not None else "dense"
        return f"{first}+rerank" if self._reranker is not None else first

    async def retrieve(self, query: str, k: int) -> list[Document]:
        if self._lexical_index is None and self._reranker is None:
            # Nothing to fuse or re-score: ask for exactly k, as before.
            return await self._dense(query, k)

        pool = max(k, self._candidates)
        candidates = await self._first_stage(query, pool)

        if self._reranker is None or not candidates:
            return candidates[:k]

        with tracer.start_as_current_span("retrieve.rerank") as span:
            span.set_attribute("aegis.rerank_candidates", len(candidates))
            return await self._reranker.rerank(query, candidates, k)

    async def _first_stage(self, query: str, pool: int) -> list[Document]:
        if self._lexical_index is None:
            return await self._dense(query, pool)

        # Independent lookups; neither waits for the other.
        dense, lexical = await asyncio.gather(
            self._dense(query, pool), self._lexical(self._lexical_index, query, pool)
        )
        with tracer.start_as_current_span("retrieve.fuse") as span:
            fused = reciprocal_rank_fusion([dense, lexical], k=self._rrf_k)
            span.set_attribute("aegis.dense_hits", len(dense))
            span.set_attribute("aegis.lexical_hits", len(lexical))
            span.set_attribute("aegis.fused", len(fused))
        return fused[:pool]

    async def _dense(self, query: str, pool: int) -> list[Document]:
        with tracer.start_as_current_span("retrieve.dense"):
            return await self._vector_store.similarity_search(query, k=pool)

    @staticmethod
    async def _lexical(index: LexicalIndexPort, query: str, pool: int) -> list[Document]:
        with tracer.start_as_current_span("retrieve.lexical"):
            return await index.search(query, k=pool)
