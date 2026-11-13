"""
Sends simple questions to a small model and the rest to the large one (ADR 0026).

Most questions a knowledge base receives are single lookups, which a small
model answers from the retrieved passage as well as a large one, faster and
for a fraction of the price. ``assess_complexity`` decides per question; this
client acts on the decision. It implements the same ``LLMClientPort``, so the
use case, the readiness probe and the shutdown hook do not change, exactly as
with the fallback client (ADR 0009).

The large side is whatever the deployment already had: one adapter, or the
fallback chain. The small side is one more backend, with its own circuit
breaker. When it fails, the question is answered by the large side instead:
routing saves money, it must not cost availability. When the small model
answers, the answer is final; there is no second opinion from the large one,
which would cost more than routing saves.

Each answer still reports the provider and model that produced it, so token
and cost metrics stay per model, and the route taken is counted separately in
``aegis_llm_routes_total``.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing

import structlog
from opentelemetry import trace

from aegis.domain.models.document import Document
from aegis.domain.models.generation import Generation, GenerationEvent, TextDelta
from aegis.domain.ports.llm_client import LLMClientPort, LLMRateLimitedError, LLMUnavailableError
from aegis.domain.services.query_complexity import ComplexityAssessment, Tier, assess_complexity
from aegis.infrastructure.llm.fallback import Backend, CircuitBreaker
from aegis.infrastructure.observability.metrics import (
    llm_circuit_open,
    llm_route_escalations_total,
    llm_routes_total,
)

logger = structlog.get_logger(__name__)


class RoutingLLMClient(LLMClientPort):
    def __init__(
        self,
        *,
        small: Backend,
        large: LLMClientPort,
        failure_threshold: int = 3,
        reset_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._small = small
        self._large = large
        self._breaker = CircuitBreaker(small.name, failure_threshold, reset_seconds, clock)
        llm_circuit_open.labels(provider=small.name).set(0)

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    def _route(self, query: str, context: list[Document]) -> ComplexityAssessment:
        assessment = assess_complexity(query, context_size=len(context))
        llm_routes_total.labels(tier=assessment.tier.value, reason=assessment.reason).inc()
        span = trace.get_current_span()
        span.set_attribute("aegis.llm_tier", assessment.tier.value)
        span.set_attribute("aegis.llm_route_reason", assessment.reason)
        logger.info(
            "llm.routed",
            tier=assessment.tier.value,
            signals=list(assessment.signals),
        )
        return assessment

    def _small_failed(self, exc: LLMUnavailableError, *, reason: str) -> None:
        rate_limited = isinstance(exc, LLMRateLimitedError)
        self._breaker.record_failure(
            retry_after=getattr(exc, "retry_after", None), rate_limited=rate_limited
        )
        self._escalated(reason, error=str(exc))

    def _escalated(self, reason: str, **detail: object) -> None:
        llm_route_escalations_total.labels(reason=reason).inc()
        logger.warning("llm.route_escalated", small=self._small.name, reason=reason, **detail)

    async def generate(
        self,
        query: str,
        context: list[Document],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Generation:
        if self._route(query, context).tier is Tier.SIMPLE:
            if not self._breaker.allows_call():
                self._escalated("circuit_open")
            else:
                try:
                    generation = await self._small.client.generate(
                        query, context, max_tokens=max_tokens, temperature=temperature
                    )
                except LLMUnavailableError as exc:
                    self._small_failed(exc, reason="small_unavailable")
                except Exception:
                    # A defect, not an outage: raise it, but close out a
                    # half-open trial so the small model is not skipped for good.
                    self._breaker.record_failure()
                    raise
                else:
                    self._breaker.record_success()
                    return generation
        return await self._large.generate(
            query, context, max_tokens=max_tokens, temperature=temperature
        )

    async def stream(
        self,
        query: str,
        context: list[Document],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncGenerator[GenerationEvent, None]:
        """
        Streams from the chosen model. The small model can be abandoned for the
        large one only before it has sent any text; after that the answer is
        its own, and a failure is raised, as in the fallback client.
        """
        if self._route(query, context).tier is Tier.SIMPLE:
            if not self._breaker.allows_call():
                self._escalated("circuit_open")
            else:
                started = False
                # None while the small model is still streaming; a caller that
                # stops early leaves it None (the generator is closed at a yield).
                outcome: str | None = None
                try:
                    async with aclosing(
                        self._small.client.stream(
                            query, context, max_tokens=max_tokens, temperature=temperature
                        )
                    ) as events:
                        async for event in events:
                            if isinstance(event, TextDelta) and event.text:
                                started = True
                            yield event
                    outcome = "answered"
                except LLMUnavailableError as exc:
                    if started:
                        outcome = "failed"
                        self._breaker.record_failure(
                            retry_after=getattr(exc, "retry_after", None),
                            rate_limited=isinstance(exc, LLMRateLimitedError),
                        )
                        raise
                    outcome = "escalated"
                    self._small_failed(exc, reason="small_unavailable")
                except Exception:
                    outcome = "failed"
                    self._breaker.record_failure()
                    raise
                finally:
                    if outcome is None:
                        # The caller stopped early. Text received proves the
                        # model works; none settles a half-open trial as failed.
                        if started:
                            self._breaker.record_success()
                        else:
                            self._breaker.record_failure()
                if outcome == "answered":
                    self._breaker.record_success()
                    return
        async with aclosing(
            self._large.stream(query, context, max_tokens=max_tokens, temperature=temperature)
        ) as events:
            async for event in events:
                yield event

    async def health_check(self) -> bool:
        """
        Ready while the large side is: it answers every complex question and
        stands in for the small model when that one fails.
        """
        return await self._large.health_check()

    async def aclose(self) -> None:
        await self._small.client.aclose()
        await self._large.aclose()
