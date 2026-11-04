"""
The injection classifier in shadow mode: scored and recorded, never enforced.

ADR 0019 measured a fine-tuned classifier on the gateway's labelled corpus: it
closes the signature gateway's gaps but refuses about one legitimate query in
six, so it does not block. What the corpus cannot say is how it behaves on a
deployment's real questions. Shadow mode answers that without touching a single
response:

- Every query the gateway allows is also scored by the classifier, in the
  background, after the request has moved on. No answer waits for it.
- A score at or above ``threshold`` is a *flag*: counted, and logged with the
  query hash and the gateway's own score (never the query text, which the
  service does not log anywhere). A sustained flag rate is either attacks the
  gateway misses or the classifier's false positives; the hashes let someone
  with access to the queries tell which.
- Background work is bounded. With ``max_pending`` scores already in flight a
  new query is skipped and counted as ``busy`` rather than queued, so a burst
  of traffic cannot build an unbounded backlog on the CPU.
"""

from __future__ import annotations

import asyncio

import structlog

from aegis.domain.ports.security import InjectionScorerPort
from aegis.domain.ports.telemetry import QueryMetricsPort

logger = structlog.get_logger(__name__)

DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_PENDING = 8


class ShadowInjectionMonitor:
    def __init__(
        self,
        scorer: InjectionScorerPort,
        metrics: QueryMetricsPort,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        max_pending: int = DEFAULT_MAX_PENDING,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if max_pending < 1:
            raise ValueError("max_pending must be at least 1")
        self._scorer = scorer
        self._metrics = metrics
        self._threshold = threshold
        self._max_pending = max_pending
        self._pending: set[asyncio.Task[None]] = set()

    @property
    def pending(self) -> int:
        return len(self._pending)

    def observe(self, text: str, *, query_hash: str, gateway_score: int) -> None:
        """
        Schedules ``text`` to be scored and returns at once. Must be called from
        the event loop; never raises and never delays the caller.
        """
        if len(self._pending) >= self._max_pending:
            self._metrics.injection_shadow_skipped("busy")
            return
        task = asyncio.create_task(self._score(text, query_hash, gateway_score))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        """Waits for every score in flight. For shutdown, and for tests."""
        while self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    async def _score(self, text: str, query_hash: str, gateway_score: int) -> None:
        try:
            score = await self._scorer.score(text)
        except Exception as exc:
            self._metrics.injection_shadow_skipped("error")
            logger.error(
                "security.shadow_classifier_failed", query_hash=query_hash[:16], error=repr(exc)
            )
            return
        flagged = score >= self._threshold
        self._metrics.injection_shadow_scored(score, flagged=flagged)
        if flagged:
            logger.warning(
                "security.shadow_classifier_flagged",
                query_hash=query_hash[:16],
                classifier_score=round(score, 4),
                gateway_score=gateway_score,
            )
