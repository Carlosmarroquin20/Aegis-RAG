"""
Telemetry port for the query pipeline.

The use case knows which business events are worth counting — a query was
blocked, a query reached retrieval — but not how they are recorded. Naming the
events here, rather than incrementing vendor counters from the application
layer, keeps the metric names and label sets an infrastructure detail.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aegis.domain.models.security import ThreatAssessment


class QueryMetricsPort(ABC):
    """Records the pipeline outcomes that feed dashboards and alerts."""

    @abstractmethod
    def query_blocked(self, assessment: ThreatAssessment) -> None:
        """The input guard refused a query; ``assessment`` says why."""

    @abstractmethod
    def query_reached_retrieval(self) -> None:
        """A query passed the input guard and is about to hit the vector store."""
