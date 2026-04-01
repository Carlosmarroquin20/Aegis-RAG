"""
QueryRAG use case — orchestrates the full RAG pipeline.

Execution order is intentional:
  1. SecurityGateway evaluation (no I/O, fast fail)     — LLM01
  2. Vector store retrieval (network I/O)
  3. LLM generation (network I/O, slow)
  4. OutputSanitizer post-processing                    — LLM02

The use case owns no state; all dependencies are injected at construction time.
"""
from __future__ import annotations

import structlog

from aegis.application.dtos.rag_dtos import QueryRequest, QueryResponse, SourceDocument
from aegis.domain.models.query import RawQuery
from aegis.domain.ports.llm_client import LLMClientPort
from aegis.domain.ports.vector_store import VectorStorePort
from aegis.infrastructure.security.output_sanitizer import OutputReflectionError, OutputSanitizer
from aegis.infrastructure.security.security_gateway import SecurityGateway

logger = structlog.get_logger(__name__)


class SecurityViolationError(Exception):
    """Raised when the SecurityGateway blocks a query."""

    def __init__(self, reason: str, query_hash: str, threat_level: str) -> None:
        super().__init__(reason)
        self.query_hash = query_hash
        self.threat_level = threat_level


class QueryRAGUseCase:
    def __init__(
        self,
        vector_store: VectorStorePort,
        llm_client: LLMClientPort,
        security_gateway: SecurityGateway,
        output_sanitizer: OutputSanitizer | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._llm_client = llm_client
        self._gateway = security_gateway
        self._output_sanitizer = output_sanitizer or OutputSanitizer()

    async def execute(self, request: QueryRequest) -> QueryResponse:
        log = logger.bind(top_k=request.top_k)

        # ── Security evaluation: must precede all I/O ──────────────────────────
        raw_query = RawQuery(text=request.query)
        gateway_result = self._gateway.evaluate(raw_query)

        if gateway_result.blocked:
            log.warning(
                "use_case.query_blocked",
                query_hash=gateway_result.assessment.query_hash[:16],
                score=gateway_result.assessment.score,
            )
            raise SecurityViolationError(
                reason=gateway_result.rejection_reason or "Security policy violation.",
                query_hash=gateway_result.assessment.query_hash,
                threat_level=gateway_result.assessment.level.name,
            )

        assert gateway_result.sanitized_query is not None  # guaranteed if not blocked
        sanitized_text = gateway_result.sanitized_query.text
        query_hash = gateway_result.assessment.query_hash

        log = log.bind(query_hash=query_hash[:16])

        # ── Retrieval ──────────────────────────────────────────────────────────
        documents = await self._vector_store.similarity_search(sanitized_text, k=request.top_k)
        log.info("use_case.retrieved", doc_count=len(documents))

        # ── Generation ────────────────────────────────────────────────────────
        raw_answer = await self._llm_client.generate(sanitized_text, documents)

        # ── Output sanitization (LLM02) ────────────────────────────────────────
        # OutputReflectionError is intentionally NOT caught here: it propagates
        # to the route handler which returns HTTP 500 (pipeline failure, not user error).
        sanitized = self._output_sanitizer.sanitize(raw_answer, query_hash=query_hash)
        if sanitized.has_warnings:
            log.warning(
                "use_case.output_warnings",
                reflection=sanitized.reflection_detected,
                pii_types=sanitized.pii_types_detected,
            )

        sources = [
            SourceDocument(
                content_preview=doc.content[:200],
                metadata=doc.metadata,
                relevance_score=doc.relevance_score,
            )
            for doc in documents
        ]

        return QueryResponse(
            answer=sanitized.text,
            sources=sources,
            query_hash=query_hash,
            threat_level=gateway_result.assessment.level.name,
        )
