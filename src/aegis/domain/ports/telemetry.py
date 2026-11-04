"""
Telemetry port for the query pipeline.

The use case knows which business events are worth counting — a query was
blocked, a query reached retrieval — but not how they are recorded. Naming the
events here, rather than incrementing vendor counters from the application
layer, keeps the metric names and label sets an infrastructure detail.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aegis.domain.models.faithfulness import FaithfulnessAssessment
from aegis.domain.models.generation import Generation
from aegis.domain.models.security import ThreatAssessment


class QueryMetricsPort(ABC):
    """Records the pipeline outcomes that feed dashboards and alerts."""

    @abstractmethod
    def query_blocked(self, assessment: ThreatAssessment) -> None:
        """The input guard refused a query; ``assessment`` says why."""

    @abstractmethod
    def query_reached_retrieval(self) -> None:
        """A query passed the input guard and is about to hit the vector store."""

    @abstractmethod
    def llm_generation(self, client: str, generation: Generation) -> None:
        """
        The LLM answered a query for ``client``.

        ``client`` is the opaque identifier the authentication layer assigns to
        an API key, never the key itself. Pricing the tokens is the recorder's
        concern: it depends on deployment configuration, not on the pipeline.
        """

    @abstractmethod
    def token_budget_exhausted(self, tenant_id: str) -> None:
        """A query was refused because ``tenant_id`` has used its daily token budget."""

    @abstractmethod
    def token_ledger_unavailable(self) -> None:
        """The token ledger could not be read or charged; the query was served anyway."""

    @abstractmethod
    def answer_faithfulness(self, assessment: FaithfulnessAssessment) -> None:
        """An answer was judged against its retrieved context."""

    @abstractmethod
    def faithfulness_check_failed(self) -> None:
        """The faithfulness judge failed; the answer was served without a score."""

    @abstractmethod
    def injection_shadow_scored(self, score: float, *, flagged: bool) -> None:
        """The shadow classifier scored a query the gateway allowed."""

    @abstractmethod
    def injection_shadow_skipped(self, reason: str) -> None:
        """A query went unscored by the shadow classifier: ``busy`` or ``error``."""
