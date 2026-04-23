"""
Unit tests for the Prometheus /metrics endpoint.

The endpoint is mounted by the ``health`` router and intentionally public at
the application layer (protected by network-level controls in production).
These tests verify:
  - the endpoint responds 200 with the Prometheus exposition content type,
  - the response body advertises every metric defined in the observability
    module (so renames break tests, not dashboards),
  - increments registered via the metrics API appear in the scrape output.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST

from aegis.interface.api.routes.health import router


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ── Exposition contract ──────────────────────────────────────────────────────


class TestMetricsEndpointExposition:
    def test_returns_200(self) -> None:
        response = _build_client().get("/metrics")
        assert response.status_code == 200

    def test_content_type_is_prometheus_text_format(self) -> None:
        response = _build_client().get("/metrics")
        # CONTENT_TYPE_LATEST is "text/plain; version=0.0.4; charset=utf-8".
        assert response.headers["content-type"] == CONTENT_TYPE_LATEST

    def test_body_is_non_empty(self) -> None:
        body = _build_client().get("/metrics").text
        assert len(body) > 0


# ── Metric catalog — one assertion per metric name ───────────────────────────


class TestMetricsEndpointCatalog:
    """
    A rename of any of these metrics is a breaking change for external
    scrapers and dashboards. The tests below make that cost visible in CI.
    """

    def _body(self) -> str:
        return _build_client().get("/metrics").text

    def test_exposes_http_requests_total(self) -> None:
        assert "aegis_http_requests_total" in self._body()

    def test_exposes_http_request_duration_seconds(self) -> None:
        assert "aegis_http_request_duration_seconds" in self._body()

    def test_exposes_security_violations_total(self) -> None:
        assert "aegis_security_violations_total" in self._body()

    def test_exposes_security_rule_triggers_total(self) -> None:
        assert "aegis_security_rule_triggers_total" in self._body()

    def test_exposes_output_reflections_total(self) -> None:
        assert "aegis_output_reflections_total" in self._body()

    def test_exposes_rag_queries_total(self) -> None:
        assert "aegis_rag_queries_total" in self._body()


# ── Recorded values reach the exposition output ──────────────────────────────


class TestMetricsEndpointValuePropagation:
    def test_security_violation_increment_appears_in_body(
        self, reset_prometheus_registry: None
    ) -> None:
        from aegis.infrastructure.observability.metrics import security_violations_total

        security_violations_total.labels(threat_level="BLOCKED").inc()
        security_violations_total.labels(threat_level="BLOCKED").inc()

        body = _build_client().get("/metrics").text
        # Exposition lines look like:
        #   aegis_security_violations_total{threat_level="BLOCKED"} 2.0
        assert 'aegis_security_violations_total{threat_level="BLOCKED"} 2.0' in body

    def test_rag_query_counter_appears_in_body(
        self, reset_prometheus_registry: None
    ) -> None:
        from aegis.infrastructure.observability.metrics import rag_queries_total

        rag_queries_total.inc()

        body = _build_client().get("/metrics").text
        assert "aegis_rag_queries_total 1.0" in body

    def test_rule_triggers_are_counted_per_rule(
        self, reset_prometheus_registry: None
    ) -> None:
        from aegis.infrastructure.observability.metrics import security_rule_triggers_total

        security_rule_triggers_total.labels(rule="instruction_override").inc()
        security_rule_triggers_total.labels(rule="persona_hijack").inc()
        security_rule_triggers_total.labels(rule="persona_hijack").inc()

        body = _build_client().get("/metrics").text
        assert (
            'aegis_security_rule_triggers_total{rule="instruction_override"} 1.0' in body
        )
        assert 'aegis_security_rule_triggers_total{rule="persona_hijack"} 2.0' in body
