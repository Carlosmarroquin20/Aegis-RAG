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
from aegis.domain.ports.llm_client import LLMTimeoutError, LLMUnavailableError
from aegis.domain.ports.vector_store import VectorStoreUnavailableError
from aegis.interface.api import main
from aegis.interface.api.dependencies import get_query_use_case
from aegis.interface.api.error_handlers import (
    DEPENDENCY_TIMEOUT_DETAIL,
    DEPENDENCY_UNAVAILABLE_DETAIL,
)

_KEY = "unit-test-key"


def _client(monkeypatch: pytest.MonkeyPatch, *, debug: bool) -> TestClient:
    settings = Settings(debug=debug, valid_api_keys=_KEY, log_level="CRITICAL")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    # The real factory is a process-wide singleton; sharing it would let failed
    # requests in one test count towards a lockout in another.
    monkeypatch.setattr(main, "get_auth_failure_limiter", lambda: None)
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


class TestDocsContentSecurityPolicy:
    """
    The strict API policy blanks out Swagger UI, so the docs pages need their own.
    That relaxation must not leak onto API responses or into production.
    """

    @pytest.mark.parametrize("path", ["/docs", "/redoc"])
    def test_docs_pages_allow_their_cdn_bundle_in_debug(
        self, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        csp = _client(monkeypatch, debug=True).get(path).headers["Content-Security-Policy"]
        assert "script-src 'unsafe-inline' https://cdn.jsdelivr.net" in csp
        assert "connect-src 'self'" in csp
        # Still locked down where it does not need to be open.
        assert csp.startswith("default-src 'none'")
        assert "frame-ancestors 'none'" in csp

    @pytest.mark.parametrize("path", ["/openapi.json", "/health", "/metrics"])
    def test_api_responses_keep_the_strict_policy_in_debug(
        self, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        response = _client(monkeypatch, debug=True).get(path, headers={"X-API-Key": _KEY})
        assert response.headers["Content-Security-Policy"] == (
            "default-src 'none'; frame-ancestors 'none'"
        )

    @pytest.mark.parametrize("path", ["/docs", "/redoc"])
    def test_docs_paths_keep_the_strict_policy_in_production(
        self, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        response = _client(monkeypatch, debug=False).get(path)
        assert response.status_code == 404
        assert "unsafe-inline" not in response.headers["Content-Security-Policy"]


class TestRequestBodyLimit:
    """How the body limit composes with the rest of the stack on the real app."""

    def test_oversized_request_gets_a_413_that_went_through_the_stack(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(main, "MAX_REQUEST_BODY_BYTES", 32)
        response = _client(monkeypatch, debug=False).post(
            "/api/v1/query", content=b"x" * 64, headers={"X-API-Key": _KEY}
        )
        assert response.status_code == 413
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Request-ID" in response.headers

    def test_unauthenticated_oversized_request_is_rejected_by_auth_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a valid key the caller should learn nothing about body limits."""
        monkeypatch.setattr(main, "MAX_REQUEST_BODY_BYTES", 32)
        response = _client(monkeypatch, debug=False).post("/api/v1/query", content=b"x" * 64)
        assert response.status_code == 403

    def test_limit_leaves_room_for_a_file_of_the_maximum_size(self) -> None:
        from aegis.interface.api.routes import documents

        assert main.MAX_REQUEST_BODY_BYTES > documents.MAX_UPLOAD_BYTES


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


class TestDependencyFailures:
    """Outages of the vector store or the LLM are not internal errors."""

    @staticmethod
    def _query(monkeypatch: pytest.MonkeyPatch, error: Exception) -> Response:
        class FailingUseCase:
            async def execute(self, _request: object) -> None:
                raise error

        client = _client(monkeypatch, debug=False)
        client.app.dependency_overrides[get_query_use_case] = FailingUseCase  # type: ignore[attr-defined]
        return client.post("/api/v1/query", json={"query": "hello"}, headers={"X-API-Key": _KEY})

    @pytest.mark.parametrize(
        "error",
        [
            VectorStoreUnavailableError("chroma:8000 refused"),
            LLMUnavailableError("ollama:11434 refused"),
        ],
        ids=["vector-store", "llm"],
    )
    def test_unavailable_dependency_returns_retryable_503(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        response = self._query(monkeypatch, error)
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "5"
        assert response.json() == {"detail": DEPENDENCY_UNAVAILABLE_DETAIL}

    def test_llm_timeout_returns_504(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = self._query(monkeypatch, LLMTimeoutError("no answer after 120s"))
        assert response.status_code == 504
        assert response.json() == {"detail": DEPENDENCY_TIMEOUT_DETAIL}

    def test_response_does_not_disclose_the_backend_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = self._query(monkeypatch, LLMUnavailableError("ollama:11434 refused"))
        assert "ollama" not in response.text
        assert "11434" not in response.text

    def test_response_still_passes_through_the_middleware_stack(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = self._query(monkeypatch, VectorStoreUnavailableError("down"))
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Request-ID" in response.headers
