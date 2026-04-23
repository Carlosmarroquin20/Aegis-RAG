"""
Unit tests for the observability & security middleware stack.

Covers middleware that previously had no direct coverage:
  - RequestIDMiddleware        → correlation IDs + propagation + structlog context
  - SecurityHeadersMiddleware  → OWASP-aligned response headers
  - AccessLogMiddleware        → X-Response-Time header + Prometheus metrics

Each middleware is mounted on a minimal FastAPI app with a single test route
to keep the assertions focused and fast.
"""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from aegis.interface.api.middleware.security_middleware import (
    AccessLogMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_app(middleware_cls: type[BaseHTTPMiddleware]) -> TestClient:
    """
    Builds a single-middleware FastAPI app with two test routes.

    ``/ping``             — static path, returns {"pong": true}.
    ``/items/{item_id}``  — parameterized path; used to verify metric label
                            templating (``/items/{item_id}`` should be the
                            label value, never the raw URL with the id).
    """
    app = FastAPI()
    app.add_middleware(middleware_cls)

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"pong": True}

    @app.get("/items/{item_id}")
    async def get_item(item_id: str) -> dict[str, str]:
        return {"item": item_id}

    return TestClient(app)


# ── RequestIDMiddleware ──────────────────────────────────────────────────────


class TestRequestIDMiddleware:
    def test_generates_uuid_when_header_absent(self) -> None:
        client = _build_app(RequestIDMiddleware)
        response = client.get("/ping")
        request_id = response.headers.get("X-Request-ID", "")
        # UUID4 hex is 32 lowercase hex chars (no dashes — we use .hex).
        assert re.fullmatch(r"[0-9a-f]{32}", request_id) is not None

    def test_reuses_caller_provided_id(self) -> None:
        client = _build_app(RequestIDMiddleware)
        upstream_id = "req-abc-123-from-gateway"
        response = client.get("/ping", headers={"X-Request-ID": upstream_id})
        assert response.headers["X-Request-ID"] == upstream_id

    def test_each_request_gets_distinct_id_by_default(self) -> None:
        client = _build_app(RequestIDMiddleware)
        first = client.get("/ping").headers["X-Request-ID"]
        second = client.get("/ping").headers["X-Request-ID"]
        assert first != second

    def test_id_is_present_even_on_404_responses(self) -> None:
        """Middleware must run for unmatched routes so IDs appear in 404 logs."""
        client = _build_app(RequestIDMiddleware)
        response = client.get("/this-route-does-not-exist")
        assert response.status_code == 404
        assert "X-Request-ID" in response.headers


# ── SecurityHeadersMiddleware ────────────────────────────────────────────────


class TestSecurityHeadersMiddleware:
    @pytest.fixture()
    def response_headers(self) -> dict[str, str]:
        client = _build_app(SecurityHeadersMiddleware)
        return dict(client.get("/ping").headers)

    def test_sets_nosniff(self, response_headers: dict[str, str]) -> None:
        assert response_headers["x-content-type-options"] == "nosniff"

    def test_sets_frame_deny(self, response_headers: dict[str, str]) -> None:
        assert response_headers["x-frame-options"] == "DENY"

    def test_disables_legacy_xss_auditor(self, response_headers: dict[str, str]) -> None:
        """Modern best practice: rely on CSP, not the deprecated XSS auditor."""
        assert response_headers["x-xss-protection"] == "0"

    def test_sets_hsts_with_long_max_age(self, response_headers: dict[str, str]) -> None:
        hsts = response_headers["strict-transport-security"]
        assert "max-age=63072000" in hsts
        assert "includeSubDomains" in hsts

    def test_sets_cache_control_no_store(self, response_headers: dict[str, str]) -> None:
        assert response_headers["cache-control"] == "no-store"

    def test_sets_restrictive_csp(self, response_headers: dict[str, str]) -> None:
        csp = response_headers["content-security-policy"]
        assert "default-src 'none'" in csp
        assert "frame-ancestors 'none'" in csp

    def test_sets_referrer_policy(self, response_headers: dict[str, str]) -> None:
        assert response_headers["referrer-policy"] == "strict-origin-when-cross-origin"

    def test_sets_permissions_policy(self, response_headers: dict[str, str]) -> None:
        policy = response_headers["permissions-policy"]
        assert "geolocation=()" in policy
        assert "camera=()" in policy
        assert "microphone=()" in policy

    def test_headers_present_on_error_responses(self) -> None:
        """Security headers must apply to 404s and other non-200 responses too."""
        client = _build_app(SecurityHeadersMiddleware)
        response = client.get("/does-not-exist")
        assert response.status_code == 404
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"


# ── AccessLogMiddleware ──────────────────────────────────────────────────────


class TestAccessLogMiddleware:
    def test_sets_response_time_header(self) -> None:
        client = _build_app(AccessLogMiddleware)
        response = client.get("/ping")
        response_time = response.headers.get("X-Response-Time", "")
        assert response_time.endswith("ms")
        # Strip "ms" and ensure the prefix is a positive float.
        assert float(response_time[:-2]) >= 0.0

    def test_records_http_requests_total_counter(
        self, reset_prometheus_registry: None
    ) -> None:
        from aegis.infrastructure.observability.metrics import registry

        client = _build_app(AccessLogMiddleware)
        client.get("/ping")

        value = registry.get_sample_value(
            "aegis_http_requests_total",
            {"method": "GET", "path": "/ping", "status": "200"},
        )
        assert value == 1.0

    def test_records_latency_histogram(self, reset_prometheus_registry: None) -> None:
        from aegis.infrastructure.observability.metrics import registry

        client = _build_app(AccessLogMiddleware)
        client.get("/ping")

        # The histogram emits a _count sample that increments per observation.
        count = registry.get_sample_value(
            "aegis_http_request_duration_seconds_count",
            {"method": "GET", "path": "/ping"},
        )
        assert count == 1.0

    def test_uses_route_template_not_raw_path(
        self, reset_prometheus_registry: None
    ) -> None:
        """
        Path parameters must collapse into the route template to keep
        Prometheus label cardinality bounded.
        """
        from aegis.infrastructure.observability.metrics import registry

        client = _build_app(AccessLogMiddleware)
        client.get("/items/abc")
        client.get("/items/xyz")

        # Two requests, same template → one label combination with count=2.
        templated = registry.get_sample_value(
            "aegis_http_requests_total",
            {"method": "GET", "path": "/items/{item_id}", "status": "200"},
        )
        assert templated == 2.0

        # The raw paths must never appear as label values.
        raw_abc = registry.get_sample_value(
            "aegis_http_requests_total",
            {"method": "GET", "path": "/items/abc", "status": "200"},
        )
        raw_xyz = registry.get_sample_value(
            "aegis_http_requests_total",
            {"method": "GET", "path": "/items/xyz", "status": "200"},
        )
        assert raw_abc is None
        assert raw_xyz is None

    def test_counts_status_codes_separately(
        self, reset_prometheus_registry: None
    ) -> None:
        from aegis.infrastructure.observability.metrics import registry

        client = _build_app(AccessLogMiddleware)
        client.get("/ping")                 # 200
        client.get("/does-not-exist")       # 404

        ok = registry.get_sample_value(
            "aegis_http_requests_total",
            {"method": "GET", "path": "/ping", "status": "200"},
        )
        assert ok == 1.0

        # For 404s, the path label falls back to the raw URL because no route
        # is matched. That is expected and acceptable for error counts.
        not_found = registry.get_sample_value(
            "aegis_http_requests_total",
            {"method": "GET", "path": "/does-not-exist", "status": "404"},
        )
        assert not_found == 1.0


# ── Smoke test: middleware composition ───────────────────────────────────────


class TestMiddlewareComposition:
    """
    The real production stack layers all three of these middleware.
    Verify they don't interfere with one another.
    """

    def test_stacking_all_three_middleware(self) -> None:
        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)
        app.add_middleware(AccessLogMiddleware)
        app.add_middleware(SecurityHeadersMiddleware)

        @app.get("/ping")
        async def ping() -> dict[str, bool]:
            return {"pong": True}

        response = TestClient(app).get("/ping")

        assert response.status_code == 200
        assert response.json() == {"pong": True}
        # Each middleware contributed its signature header:
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time" in response.headers
        assert response.headers["x-content-type-options"] == "nosniff"
