"""
Prometheus metrics definitions.

All metrics live in a single module so they are registered exactly once
(Prometheus raises on duplicate registration) and so instrumentation call
sites across the codebase share the same metric identities.

Naming convention follows the Prometheus best-practice guide:
  - Counters end in ``_total``.
  - Histograms use base units (seconds, bytes).
  - Label cardinality is kept bounded: HTTP path uses the matched route
    template (``/api/v1/documents/{id}``), never the raw URL, to avoid
    unbounded series explosion from path parameters.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

from aegis.domain.models.security import ThreatAssessment
from aegis.domain.ports.telemetry import QueryMetricsPort

# A dedicated registry keeps Aegis metrics isolated from the default global one,
# which the prometheus_client library also uses for process/platform collectors.
# Exposing our registry explicitly makes scrape output deterministic.
registry = CollectorRegistry()

# ── HTTP layer ────────────────────────────────────────────────────────────────

http_requests_total = Counter(
    "aegis_http_requests_total",
    "Total HTTP requests processed, partitioned by method, route template, and status.",
    labelnames=("method", "path", "status"),
    registry=registry,
)

# Buckets tuned for RAG latency: embeddings are fast (ms), vector search is
# fast (tens of ms), LLM generation dominates (hundreds of ms to seconds).
http_request_duration_seconds = Histogram(
    "aegis_http_request_duration_seconds",
    "HTTP request processing time in seconds.",
    labelnames=("method", "path"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=registry,
)

# ── Security layer ────────────────────────────────────────────────────────────

security_violations_total = Counter(
    "aegis_security_violations_total",
    "Total queries blocked by the SecurityGateway, partitioned by threat level.",
    labelnames=("threat_level",),
    registry=registry,
)

security_rule_triggers_total = Counter(
    "aegis_security_rule_triggers_total",
    "Total times a specific SecurityGateway rule was triggered (block or suspicious).",
    labelnames=("rule",),
    registry=registry,
)

output_reflections_total = Counter(
    "aegis_output_reflections_total",
    "Total LLM responses blocked by OutputSanitizer reflection detection.",
    registry=registry,
)

auth_failures_total = Counter(
    "aegis_auth_failures_total",
    "Requests refused at authentication: 'rejected' (bad or missing key, 403) or "
    "'throttled' (client address locked out after repeated failures, 429).",
    labelnames=("outcome",),
    registry=registry,
)

rate_limit_backend_errors_total = Counter(
    "aegis_rate_limit_backend_errors_total",
    "Requests for which the rate-limit store could not be consulted.",
    registry=registry,
)

# ── RAG pipeline ──────────────────────────────────────────────────────────────

rag_queries_total = Counter(
    "aegis_rag_queries_total",
    "Total RAG queries that reached the retrieval+generation stage.",
    registry=registry,
)


class PrometheusQueryMetrics(QueryMetricsPort):
    """
    Records query-pipeline outcomes on the counters above.

    Stateless: the counters are module-level singletons bound to ``registry``,
    so any number of instances report into the same series.
    """

    def query_blocked(self, assessment: ThreatAssessment) -> None:
        security_violations_total.labels(threat_level=assessment.level.name).inc()
        for rule in assessment.triggered_rules:
            security_rule_triggers_total.labels(rule=rule).inc()

    def query_reached_retrieval(self) -> None:
        rag_queries_total.inc()
