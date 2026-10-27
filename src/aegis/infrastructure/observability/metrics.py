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

from collections.abc import Mapping
from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from aegis.domain.models.generation import Generation, TokenUsage
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

# ── Runtime ───────────────────────────────────────────────────────────────────

gc_pause_seconds = Histogram(
    "aegis_gc_pause_seconds",
    "Time the Python garbage collector stopped the process, by generation.",
    labelnames=("generation",),
    # A young-generation pass takes microseconds; a full pass over an unfrozen
    # heap took over a second under load, which is the tail this must show.
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=registry,
)

# ── RAG pipeline ──────────────────────────────────────────────────────────────

rag_queries_total = Counter(
    "aegis_rag_queries_total",
    "Total RAG queries that reached the retrieval+generation stage.",
    registry=registry,
)

# ── LLM usage and cost ────────────────────────────────────────────────────────
# ``client`` is the digest prefix APIKeyMiddleware assigns to each configured
# key, so its cardinality is bounded by the number of keys, not by traffic.

llm_tokens_total = Counter(
    "aegis_llm_tokens_total",
    "Tokens consumed by LLM generations, by direction (input or output).",
    ["provider", "model", "client", "direction"],
    registry=registry,
)

llm_cost_usd_total = Counter(
    "aegis_llm_cost_usd_total",
    "Estimated LLM spend in US dollars, from the configured per-token prices.",
    ["provider", "model", "client"],
    registry=registry,
)

llm_usage_unreported_total = Counter(
    "aegis_llm_usage_unreported_total",
    "LLM generations whose backend reported no token usage, so they are not costed.",
    ["provider", "model"],
    registry=registry,
)


llm_fallbacks_total = Counter(
    "aegis_llm_fallbacks_total",
    "Answers served by a fallback LLM backend, by why the primary was not used.",
    ["from_provider", "to_provider", "reason"],
    registry=registry,
)

llm_circuit_open = Gauge(
    "aegis_llm_circuit_open",
    "1 while the circuit breaker for an LLM backend is open and calls skip it.",
    ["provider"],
    registry=registry,
)


@dataclass(frozen=True, slots=True)
class LLMPricing:
    """
    Prices in US dollars per million tokens, as providers publish them.

    Zero by default: a local model costs nothing per token, and a hosted one
    should be costed only from prices someone has configured deliberately.
    """

    input_per_million: float = 0.0
    output_per_million: float = 0.0

    def cost(self, usage: TokenUsage) -> float:
        return (
            usage.input_tokens * self.input_per_million
            + usage.output_tokens * self.output_per_million
        ) / 1_000_000


class PrometheusQueryMetrics(QueryMetricsPort):
    """
    Records query-pipeline outcomes on the counters above.

    Stateless: the counters are module-level singletons bound to ``registry``,
    so any number of instances report into the same series.
    """

    def __init__(self, prices: Mapping[str, LLMPricing] | None = None) -> None:
        # Keyed by Generation.provider. A provider without an entry is costed at
        # zero: with a fallback in place, one deployment can be answered by a
        # hosted model and by a local one, and only the former is billed.
        self._prices = dict(prices or {})

    def query_blocked(self, assessment: ThreatAssessment) -> None:
        security_violations_total.labels(threat_level=assessment.level.name).inc()
        for rule in assessment.triggered_rules:
            security_rule_triggers_total.labels(rule=rule).inc()

    def query_reached_retrieval(self) -> None:
        rag_queries_total.inc()

    def llm_generation(self, client: str, generation: Generation) -> None:
        usage = generation.usage
        if usage is None:
            llm_usage_unreported_total.labels(
                provider=generation.provider, model=generation.model
            ).inc()
            return
        labels = {"provider": generation.provider, "model": generation.model, "client": client}
        llm_tokens_total.labels(**labels, direction="input").inc(usage.input_tokens)
        llm_tokens_total.labels(**labels, direction="output").inc(usage.output_tokens)
        pricing = self._prices.get(generation.provider, LLMPricing())
        llm_cost_usd_total.labels(**labels).inc(pricing.cost(usage))
