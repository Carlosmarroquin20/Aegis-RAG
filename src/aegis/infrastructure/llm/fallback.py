"""
Routes generation across LLM backends in priority order, with a circuit breaker
per backend.

A hosted provider is fast but outside our control: it rate limits a free key
after a burst, has outages, and can be slow to recover. Without a fallback each
of those becomes a 503 for every user. With one, the query is answered by the
next backend in line, typically the local Ollama model: slower, but available.

The circuit breaker is what keeps a failing backend from costing every request
its timeout. After ``failure_threshold`` consecutive failures the backend is
skipped for ``reset_seconds``; then a single trial request decides whether it
is back. A rate-limited reply opens the circuit at once, for as long as the
provider asked (Retry-After) when that is longer than the reset period:
calling again sooner would only be refused again and could extend the penalty.

Data locality is the caller's decision, not this module's. Falling back from a
hosted provider to the local model keeps data on the host; the reverse sends it
off the host only when the local model fails. The wiring logs which applies.

State is per process and lives in memory, like the rest of the in-process
caches. Each worker learns a backend's health independently, which costs at
most ``failure_threshold`` failed calls per worker.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import structlog

from aegis.domain.models.document import Document
from aegis.domain.models.generation import Generation
from aegis.domain.ports.llm_client import (
    LLMClientPort,
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from aegis.infrastructure.observability.metrics import llm_circuit_open, llm_fallbacks_total

logger = structlog.get_logger(__name__)


class CircuitState(StrEnum):
    CLOSED = "closed"  # calls flow normally
    OPEN = "open"  # calls are skipped until the cool-down ends
    HALF_OPEN = "half_open"  # one trial call is in flight


@dataclass
class CircuitBreaker:
    """
    Consecutive-failure circuit breaker for one backend.

    Not thread-safe by design: it is used from a single event loop, where the
    only interleaving points are the awaits around the call it guards.
    """

    name: str
    failure_threshold: int = 3
    reset_seconds: float = 30.0
    clock: Callable[[], float] = time.monotonic
    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    consecutive_failures: int = field(default=0, init=False)
    _open_until: float = field(default=0.0, init=False)

    def allows_call(self) -> bool:
        if self.state is CircuitState.CLOSED:
            return True
        if self.state is CircuitState.OPEN and self.clock() >= self._open_until:
            # Cool-down over: let exactly one request find out whether it is back.
            self._set_state(CircuitState.HALF_OPEN)
            return True
        return False

    def record_success(self) -> None:
        self.consecutive_failures = 0
        if self.state is not CircuitState.CLOSED:
            self._set_state(CircuitState.CLOSED)

    def record_failure(
        self, *, retry_after: float | None = None, rate_limited: bool = False
    ) -> None:
        self.consecutive_failures += 1
        if (
            rate_limited
            or self.state is CircuitState.HALF_OPEN
            or self.consecutive_failures >= self.failure_threshold
        ):
            self._open(max(self.reset_seconds, retry_after or 0.0))

    def _open(self, seconds: float) -> None:
        self._open_until = self.clock() + seconds
        self._set_state(CircuitState.OPEN, open_seconds=round(seconds, 1))

    def _set_state(self, state: CircuitState, **detail: object) -> None:
        previous, self.state = self.state, state
        llm_circuit_open.labels(provider=self.name).set(1 if state is CircuitState.OPEN else 0)
        logger.warning(
            "llm.circuit_state_changed",
            provider=self.name,
            previous=previous.value,
            state=state.value,
            consecutive_failures=self.consecutive_failures,
            **detail,
        )


@dataclass(frozen=True, slots=True)
class Backend:
    name: str  # the provider name its Generation reports, for logs and metrics
    client: LLMClientPort


def _failure_reason(exc: LLMUnavailableError) -> str:
    if isinstance(exc, LLMRateLimitedError):
        return "rate_limited"
    if isinstance(exc, LLMTimeoutError):
        return "timeout"
    return "unavailable"


class FallbackLLMClient(LLMClientPort):
    """Tries each backend in order; the first answer wins."""

    def __init__(
        self,
        backends: Sequence[Backend],
        *,
        failure_threshold: int = 3,
        reset_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if len(backends) < 2:
            raise ValueError("a fallback needs at least two backends")
        self._backends = list(backends)
        self._breakers = {
            b.name: CircuitBreaker(b.name, failure_threshold, reset_seconds, clock)
            for b in self._backends
        }
        for backend in self._backends:
            llm_circuit_open.labels(provider=backend.name).set(0)

    def breaker(self, name: str) -> CircuitBreaker:
        return self._breakers[name]

    async def generate(
        self,
        query: str,
        context: list[Document],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Generation:
        primary = self._backends[0].name
        first_problem: str | None = None
        last_error: LLMUnavailableError | None = None

        for backend in self._backends:
            breaker = self._breakers[backend.name]
            if not breaker.allows_call():
                first_problem = first_problem or "circuit_open"
                continue
            try:
                generation = await backend.client.generate(
                    query, context, max_tokens=max_tokens, temperature=temperature
                )
            except LLMUnavailableError as exc:
                reason = _failure_reason(exc)
                breaker.record_failure(
                    retry_after=getattr(exc, "retry_after", None),
                    rate_limited=reason == "rate_limited",
                )
                logger.warning(
                    "llm.backend_failed", provider=backend.name, reason=reason, error=str(exc)
                )
                first_problem = first_problem or reason
                last_error = exc
                continue
            except Exception:
                # A defect, not an outage: do not fall back, but never leave the
                # circuit half-open, or the backend would be skipped for good.
                breaker.record_failure()
                raise

            breaker.record_success()
            if backend.name != primary:
                llm_fallbacks_total.labels(
                    from_provider=primary, to_provider=backend.name, reason=first_problem
                ).inc()
                logger.warning(
                    "llm.fallback_used",
                    from_provider=primary,
                    to_provider=backend.name,
                    reason=first_problem,
                )
            return generation

        if last_error is not None:
            # The last real failure keeps its type, so a timeout still maps to 504.
            raise last_error
        raise LLMUnavailableError("every LLM backend is skipped by an open circuit")

    async def health_check(self) -> bool:
        """Ready while any backend can answer: that is the promise of the fallback."""
        for backend in self._backends:
            if await backend.client.health_check():
                return True
        return False

    async def aclose(self) -> None:
        for backend in self._backends:
            await backend.client.aclose()
