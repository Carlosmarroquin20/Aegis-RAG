"""
QueryRAG use case — orchestrates the full RAG pipeline.

Execution order is intentional:
  1. SecurityGateway evaluation (no I/O, fast fail)     — LLM01
  2. Tenant token budget check, when the tenant has one — LLM10
  3. Retrieval: dense, optionally fused with keyword search and reranked
  4. LLM generation (network I/O, slow), then the tokens are charged
  5. OutputSanitizer post-processing                    — LLM05
  6. Faithfulness of the answer to its context, when a judge is configured

The use case owns no state; all dependencies are injected at construction time,
and every one of them is a domain port. Nothing here imports from the
infrastructure layer.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from aegis.application.dtos.rag_dtos import (
    FaithfulnessReport,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    StreamDelta,
    StreamEnd,
    StreamStart,
)
from aegis.application.services.retrieval import Retriever
from aegis.application.services.shadow_injection import ShadowInjectionMonitor
from aegis.application.services.streaming import StreamingOutputGuard
from aegis.domain.models.document import Document
from aegis.domain.models.generation import Generation
from aegis.domain.models.query import RawQuery
from aegis.domain.models.tenant import Tenant
from aegis.domain.ports.faithfulness import FaithfulnessJudgePort
from aegis.domain.ports.llm_client import LLMClientPort
from aegis.domain.ports.retrieval import RetrieverPort
from aegis.domain.ports.security import OutputGuardPort, QueryGuardPort
from aegis.domain.ports.telemetry import QueryMetricsPort
from aegis.domain.ports.token_ledger import TokenBudgetExhaustedError, TokenLedgerPort
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


@dataclass(frozen=True, slots=True)
class _Prepared:
    """Everything decided before generation: the checked query and its context."""

    query: str
    query_hash: str
    threat_level: str
    documents: list[Document]
    client: str
    tenant: Tenant | None

    @property
    def budget(self) -> int | None:
        return self.tenant.daily_token_budget if self.tenant is not None else None

    def sources(self) -> list[SourceDocument]:
        return [
            SourceDocument(
                content_preview=doc.content[:200],
                metadata=doc.metadata,
                relevance_score=doc.relevance_score,
            )
            for doc in self.documents
        ]


@dataclass(frozen=True, slots=True)
class QueryStream:
    """
    A query whose security check, budget check and retrieval have already run,
    so any refusal was raised before a response started. ``events`` generates
    the answer: sanitized deltas, then one ``StreamEnd``.
    """

    start: StreamStart
    events: AsyncGenerator[StreamDelta | StreamEnd, None]


class QueryRAGUseCase:
    def __init__(
        self,
        vector_store: VectorStorePort,
        llm_client: LLMClientPort,
        security_gateway: QueryGuardPort,
        output_sanitizer: OutputGuardPort,
        metrics: QueryMetricsPort,
        token_ledger: TokenLedgerPort | None = None,
        retriever: RetrieverPort | None = None,
        faithfulness_judge: FaithfulnessJudgePort | None = None,
        injection_monitor: ShadowInjectionMonitor | None = None,
    ) -> None:
        # Without a retriever, retrieval is dense similarity from vector_store.
        self._retriever = retriever if retriever is not None else Retriever(vector_store)
        self._llm_client = llm_client
        self._gateway = security_gateway
        self._output_sanitizer = output_sanitizer
        self._metrics = metrics
        self._ledger = token_ledger
        self._judge = faithfulness_judge
        self._injection_monitor = injection_monitor

    async def execute(
        self, request: QueryRequest, *, client: str = "unknown", tenant: Tenant | None = None
    ) -> QueryResponse:
        """
        Runs the pipeline for one query. ``client`` attributes the LLM usage to
        the caller and ``tenant`` carries its budget; both come from
        authentication, never from the request body.
        """
        prepared = await self._prepare(request, client=client, tenant=tenant)
        log = logger.bind(query_hash=prepared.query_hash[:16])
        query_hash = prepared.query_hash
        documents = prepared.documents

        # ── Generation ────────────────────────────────────────────────────────
        with tracer.start_as_current_span("llm.generate") as span:
            span.set_attribute("aegis.context_docs", len(documents))
            generation = await self._llm_client.generate(prepared.query, documents)
            _annotate_generation(span, generation)
        await self._account(generation, prepared)

        # ── Output sanitization (LLM05) ────────────────────────────────────────
        # OutputReflectionError is intentionally NOT caught here: it propagates
        # to the route handler which returns HTTP 500 (pipeline failure, not user error).
        # The span therefore records that failure, which is the desired signal.
        with tracer.start_as_current_span("output.sanitize") as span:
            sanitized = self._output_sanitizer.sanitize(generation.text, query_hash=query_hash)
            span.set_attribute("aegis.reflection_detected", sanitized.reflection_detected)
        if sanitized.has_warnings:
            log.warning(
                "use_case.output_warnings",
                reflection=sanitized.reflection_detected,
                pii_types=sanitized.pii_types_detected,
            )

        faithfulness = await self._judge_faithfulness(
            sanitized.text, documents, query_hash, question=prepared.query
        )

        return QueryResponse(
            answer=sanitized.text,
            sources=prepared.sources(),
            query_hash=query_hash,
            threat_level=prepared.threat_level,
            faithfulness=faithfulness,
        )

    async def start_stream(
        self, request: QueryRequest, *, client: str = "unknown", tenant: Tenant | None = None
    ) -> QueryStream:
        """
        Runs everything before generation now (so a blocked query, a spent budget
        or a store outage raises here, while the caller can still answer with a
        status code) and returns the answer as a stream to consume.
        """
        prepared = await self._prepare(request, client=client, tenant=tenant)
        start = StreamStart(
            query_hash=prepared.query_hash,
            threat_level=prepared.threat_level,
            sources=prepared.sources(),
        )
        return QueryStream(start=start, events=self._stream_answer(prepared))

    async def _stream_answer(
        self, prepared: _Prepared
    ) -> AsyncGenerator[StreamDelta | StreamEnd, None]:
        """
        Generates, sanitizes sentence by sentence (StreamingOutputGuard), then
        finishes as ``execute`` does: usage, the full sanitizer pass, faithfulness.

        OutputReflectionError and LLM errors propagate to the caller, which owns
        the decision of what to tell a client that has already seen some text.
        Closing this generator early (a client gone) closes the LLM stream with it.
        """
        log = logger.bind(query_hash=prepared.query_hash[:16])
        guard = StreamingOutputGuard(self._output_sanitizer, query_hash=prepared.query_hash)
        generation: Generation | None = None
        # A span opened and closed by hand: a context-managed one would stay
        # "current" across the yields below, into whatever the caller runs between.
        span = tracer.start_span("llm.generate")
        span.set_attribute("aegis.context_docs", len(prepared.documents))
        span.set_attribute("aegis.stream", True)
        try:
            async with aclosing(
                self._llm_client.stream(prepared.query, prepared.documents)
            ) as events:
                async for event in events:
                    if isinstance(event, Generation):
                        generation = event
                        continue
                    released = guard.feed(event.text)
                    if released:
                        yield StreamDelta(text=released)
                    if guard.exhausted:
                        log.warning("use_case.stream_length_limit", diverged=guard.diverged)
                        break
            if generation is not None:
                _annotate_generation(span, generation)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            span.end()

        if generation is not None:
            await self._account(generation, prepared)

        with tracer.start_as_current_span("output.sanitize") as sanitize_span:
            final, tail = guard.finish()
            sanitize_span.set_attribute("aegis.reflection_detected", final.reflection_detected)
        if tail:
            yield StreamDelta(text=tail)
        if final.has_warnings:
            log.warning(
                "use_case.output_warnings",
                reflection=final.reflection_detected,
                pii_types=final.pii_types_detected,
            )

        faithfulness = await self._judge_faithfulness(
            final.text, prepared.documents, prepared.query_hash, question=prepared.query
        )
        yield StreamEnd(answer=final.text, truncated=final.was_truncated, faithfulness=faithfulness)

    async def _prepare(
        self, request: QueryRequest, *, client: str, tenant: Tenant | None
    ) -> _Prepared:
        """Security check, budget check and retrieval: everything that can refuse."""
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

        # ── Second opinion, recorded only (ADR 0020) ───────────────────────────
        # The raw text, as the classifier was evaluated on it. Scheduled in the
        # background: it cannot block or delay this query.
        if self._injection_monitor is not None:
            self._injection_monitor.observe(
                request.query, query_hash=query_hash, gateway_score=gateway_result.assessment.score
            )

        # ── Token budget (LLM10): before anything that costs ───────────────────
        budget = tenant.daily_token_budget if tenant is not None else None
        if budget is not None:
            await self._check_budget(tenant, budget)  # type: ignore[arg-type]

        # ── Retrieval ──────────────────────────────────────────────────────────
        self._metrics.query_reached_retrieval()
        with tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("aegis.top_k", request.top_k)
            documents = await self._retriever.retrieve(sanitized_text, k=request.top_k)
            span.set_attribute("aegis.doc_count", len(documents))
        log.info("use_case.retrieved", doc_count=len(documents))

        return _Prepared(
            query=sanitized_text,
            query_hash=query_hash,
            threat_level=gateway_result.assessment.level.name,
            documents=documents,
            client=client,
            tenant=tenant,
        )

    async def _account(self, generation: Generation, prepared: _Prepared) -> None:
        """Usage and cost for the generation, and the tenant's budget charge."""
        self._metrics.llm_generation(prepared.client, generation)
        budget = prepared.budget
        if budget is not None and generation.usage is not None:
            assert prepared.tenant is not None
            await self._charge(prepared.tenant, generation.usage.total_tokens)

    async def _judge_faithfulness(
        self, answer: str, documents: list[Document], query_hash: str, *, question: str
    ) -> FaithfulnessReport | None:
        """
        Scores the answer against the chunks it was generated from. A monitoring
        signal, not a gate: an unsupported answer is still returned (with the
        sentences that lack support), and a failing judge costs the score, not
        the answer.
        """
        if self._judge is None:
            return None
        log = logger.bind(query_hash=query_hash[:16])
        with tracer.start_as_current_span("output.faithfulness") as span:
            try:
                assessment = await self._judge.assess(answer, documents, question=question)
            except Exception as exc:
                self._metrics.faithfulness_check_failed()
                log.error("use_case.faithfulness_check_failed", error=repr(exc))
                return None
            span.set_attribute("aegis.faithfulness", assessment.score)
            span.set_attribute("aegis.abstained", assessment.abstained)
        self._metrics.answer_faithfulness(assessment)
        if not assessment.fully_supported:
            log.warning(
                "use_case.answer_unsupported",
                score=round(assessment.score, 3),
                unsupported_claims=len(assessment.unsupported),
            )
        return FaithfulnessReport(
            score=round(assessment.score, 4),
            abstained=assessment.abstained,
            unsupported=list(assessment.unsupported),
        )

    async def _check_budget(self, tenant: Tenant, budget: int) -> None:
        if self._ledger is None:
            return
        try:
            usage = await self._ledger.used_today(tenant.id)
        except Exception as exc:
            # A cost control, like the rate limiter: an unreachable ledger must
            # not take queries down. Counted and logged, never silent.
            self._metrics.token_ledger_unavailable()
            logger.error("token_ledger.unavailable", operation="check", error=repr(exc))
            return
        if usage.tokens >= budget:
            self._metrics.token_budget_exhausted(tenant.id)
            logger.warning(
                "use_case.token_budget_exhausted",
                tenant=tenant.id,
                used=usage.tokens,
                budget=budget,
            )
            raise TokenBudgetExhaustedError(tenant.id, budget, usage.seconds_to_reset)

    async def _charge(self, tenant: Tenant, tokens: int) -> None:
        if self._ledger is None:
            return
        try:
            await self._ledger.charge(tenant.id, tokens)
        except Exception as exc:
            # The answer is already generated; refusing it now would waste the
            # tokens and still not record them.
            self._metrics.token_ledger_unavailable()
            logger.error("token_ledger.unavailable", operation="charge", error=repr(exc))


def _annotate_generation(span: trace.Span, generation: Generation) -> None:
    """
    OpenTelemetry GenAI semantic conventions, so trace backends that understand
    them can chart usage without custom mapping.
    """
    span.set_attribute("gen_ai.provider.name", generation.provider)
    span.set_attribute("gen_ai.request.model", generation.model)
    if generation.usage is not None:
        span.set_attribute("gen_ai.usage.input_tokens", generation.usage.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", generation.usage.output_tokens)
