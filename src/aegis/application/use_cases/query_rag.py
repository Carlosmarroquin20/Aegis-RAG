"""
QueryRAG use case — orchestrates the full RAG pipeline.

Execution order is intentional:
  1. SecurityGateway evaluation (no I/O, fast fail)     — LLM01
  2. Vector store retrieval (network I/O)
  3. LLM generation (network I/O, slow)
  4. OutputSanitizer post-processing                    — LLM05

The use case owns no state; all dependencies are injected at construction time,
and every one of them is a domain port. Nothing here imports from the
infrastructure layer.
"""

from __future__ import annotations

import structlog
from opentelemetry import trace

from aegis.application.dtos.rag_dtos import QueryRequest, QueryResponse, SourceDocument
from aegis.domain.models.query import RawQuery
from aegis.domain.ports.llm_client import LLMClientPort
from aegis.domain.ports.security import OutputGuardPort, QueryGuardPort
from aegis.domain.ports.telemetry import QueryMetricsPort
from aegis.domain.ports.vector_store import VectorStorePort

logger = structlog.get_logger(__name__)

# The OpenTelemetry API is a vendor-neutral facade (a no-op until an SDK is
# installed by the infrastructure layer), in the same category as the logger
# above. Using it here does not couple the use case to a tracing backend.
tracer = trace.get_tracer("aegis")


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
        security_gateway: QueryGuardPort,
        output_sanitizer: OutputGuardPort,
        metrics: QueryMetricsPort,
    ) -> None:
        self._vector_store = vector_store
        self._llm_client = llm_client
        self._gateway = security_gateway
        self._output_sanitizer = output_sanitizer
        self._metrics = metrics

    async def execute(self, request: QueryRequest) -> QueryResponse:
        log = logger.bind(top_k=request.top_k)

        # ── Security evaluation: must precede all I/O ──────────────────────────
        with tracer.start_as_current_span("security.evaluate") as span:
            raw_query = RawQuery(text=request.query)
            gateway_result = self._gateway.evaluate(raw_query)
            span.set_attribute("aegis.threat_level", gateway_result.assessment.level.name)
            span.set_attribute("aegis.score", gateway_result.assessment.score)
            span.set_attribute("aegis.blocked", gateway_result.blocked)

        if gateway_result.blocked:
            log.warning(
                "use_case.query_blocked",
                query_hash=gateway_result.assessment.query_hash[:16],
                score=gateway_result.assessment.score,
            )
            self._metrics.query_blocked(gateway_result.assessment)
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
        self._metrics.query_reached_retrieval()
        with tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("aegis.top_k", request.top_k)
            documents = await self._vector_store.similarity_search(sanitized_text, k=request.top_k)
            span.set_attribute("aegis.doc_count", len(documents))
        log.info("use_case.retrieved", doc_count=len(documents))

        # ── Generation ────────────────────────────────────────────────────────
        with tracer.start_as_current_span("llm.generate") as span:
            span.set_attribute("aegis.context_docs", len(documents))
            raw_answer = await self._llm_client.generate(sanitized_text, documents)

        # ── Output sanitization (LLM05) ────────────────────────────────────────
        # OutputReflectionError is intentionally NOT caught here: it propagates
        # to the route handler which returns HTTP 500 (pipeline failure, not user error).
        # The span therefore records that failure, which is the desired signal.
        with tracer.start_as_current_span("output.sanitize") as span:
            sanitized = self._output_sanitizer.sanitize(raw_answer, query_hash=query_hash)
            span.set_attribute("aegis.reflection_detected", sanitized.reflection_detected)
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
