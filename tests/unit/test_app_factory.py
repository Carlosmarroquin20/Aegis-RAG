"""
Unit tests for ``create_app()`` — behaviour that only exists once the whole
application is assembled (settings-driven switches, cross-middleware effects).

The lifespan is never entered (``TestClient`` is used without a context manager),
so no vector store or LLM is contacted.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from aegis.config import Settings
from aegis.interface.api import main
from aegis.interface.api.dependencies import get_query_use_case

_KEY = "unit-test-key"


def _client(monkeypatch: pytest.MonkeyPatch, *, debug: bool) -> TestClient:
    settings = Settings(debug=debug, valid_api_keys=_KEY, log_level="CRITICAL")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    return TestClient(main.create_app(), raise_server_exceptions=False)


class TestApiDocsExposure:
    @pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc"])
    def test_schema_and_docs_are_not_served_in_production(
        self, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        """These paths skip authentication, so in production they must not exist."""
        client = _client(monkeypatch, debug=False)
        assert client.get(path).status_code == 404

    @pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc"])
    def test_schema_and_docs_are_served_in_debug(
        self, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        client = _client(monkeypatch, debug=True)
        assert client.get(path).status_code == 200


class _ExplodingUseCase:
    async def execute(self, _request: object) -> None:
        raise RuntimeError("db password is hunter2")


class TestUnhandledRouteErrors:
    """
    A route that raises must still produce a response that went through the whole
    middleware stack. Asserted on the real application because the property is
    about how the layers compose, not about any single middleware.
    """

    @pytest.fixture()
    def response(
        self, monkeypatch: pytest.MonkeyPatch, reset_prometheus_registry: None
    ) -> Response:
        self.client = _client(monkeypatch, debug=False)
        self.client.app.dependency_overrides[get_query_use_case] = _ExplodingUseCase  # type: ignore[attr-defined]
        return self.client.post(
            "/api/v1/query", json={"query": "hello"}, headers={"X-API-Key": _KEY}
        )

    def test_returns_generic_500_without_leaking_the_error(self, response: Response) -> None:
        assert response.status_code == 500
        assert "hunter2" not in response.text
        assert response.json() == {"detail": main.INTERNAL_ERROR_DETAIL}

    def test_500_carries_security_headers_and_request_id(self, response: Response) -> None:
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Cache-Control"] == "no-store"
        assert "X-Request-ID" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

    def test_500_is_counted_in_request_metrics(self, response: Response) -> None:
        """The AegisHighErrorRate alert is computed from this series."""
        exposition = self.client.get("/metrics").text
        expected = 'aegis_http_requests_total{method="POST",path="/api/v1/query",status="500"} 1.0'
        assert expected in exposition
