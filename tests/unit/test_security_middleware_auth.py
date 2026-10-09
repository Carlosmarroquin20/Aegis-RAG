"""
Unit tests for the authentication and rate-limiting middleware.

These complement test_middleware.py (which covers RequestID / AccessLog /
SecurityHeaders) by exercising the APIKeyMiddleware and RateLimitMiddleware
dispatch paths against a minimal app.
"""

from __future__ import annotations

import hashlib
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegis.config import Settings
from aegis.infrastructure.security.rate_limiter import (
    InMemoryRateLimitStore,
    RateLimiter,
    RateLimitPolicy,
)
from aegis.interface.api.middleware.security_middleware import (
    APIKeyMiddleware,
    RateLimitMiddleware,
)

_KEY = "secret-key"


def _settings() -> Settings:
    return Settings(valid_api_keys=_KEY, api_key_header="X-API-Key")


def _app(
    *, with_rate_limit: bool = False, limit: int = 5, burst: int = 0, window: int = 3600
) -> FastAPI:
    app = FastAPI()
    if with_rate_limit:
        limiter = RateLimiter(RateLimitPolicy(limit, window, burst_allowance=burst))
        app.add_middleware(RateLimitMiddleware, rate_limiter=limiter)
    app.add_middleware(APIKeyMiddleware, settings=_settings())

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"pong": True}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


# ── APIKeyMiddleware ─────────────────────────────────────────────────────────


class TestAPIKeyMiddleware:
    def test_missing_key_is_forbidden(self) -> None:
        response = TestClient(_app()).get("/ping")
        assert response.status_code == 403

    def test_invalid_key_is_forbidden(self) -> None:
        response = TestClient(_app()).get("/ping", headers={"X-API-Key": "wrong"})
        assert response.status_code == 403

    def test_valid_key_is_allowed(self) -> None:
        response = TestClient(_app()).get("/ping", headers={"X-API-Key": _KEY})
        assert response.status_code == 200
        assert response.json() == {"pong": True}

    @pytest.mark.parametrize("candidate", [_KEY[:-1], _KEY + "x", _KEY.upper(), " " + _KEY])
    def test_near_miss_keys_are_forbidden(self, candidate: str) -> None:
        response = TestClient(_app()).get("/ping", headers={"X-API-Key": candidate})
        assert response.status_code == 403

    def test_every_configured_key_is_accepted(self) -> None:
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, settings=Settings(valid_api_keys="alpha, beta,gamma"))

        @app.get("/ping")
        async def ping() -> dict[str, bool]:
            return {"pong": True}

        client = TestClient(app)
        for key in ("alpha", "beta", "gamma"):
            assert client.get("/ping", headers={"X-API-Key": key}).status_code == 200

    @pytest.mark.parametrize("candidate", ["", "anything"])
    def test_fails_closed_when_no_keys_are_configured(self, candidate: str) -> None:
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, settings=Settings(valid_api_keys=""))

        @app.get("/ping")
        async def ping() -> dict[str, bool]:
            return {"pong": True}

        response = TestClient(app).get("/ping", headers={"X-API-Key": candidate})
        assert response.status_code == 403

    def test_public_path_bypasses_auth(self) -> None:
        # /health is in _PUBLIC_PATHS, so no key is required.
        response = TestClient(_app()).get("/health")
        assert response.status_code == 200


# ── RateLimitMiddleware ──────────────────────────────────────────────────────


class TestRateLimitMiddleware:
    def test_allowed_requests_expose_quota_headers(self) -> None:
        client = TestClient(_app(with_rate_limit=True, limit=5))
        response = client.get("/ping", headers={"X-API-Key": _KEY})
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

    def test_requests_over_limit_are_throttled(self) -> None:
        client = TestClient(_app(with_rate_limit=True, limit=2))
        headers = {"X-API-Key": _KEY}
        for _ in range(2):
            assert client.get("/ping", headers=headers).status_code == 200
        blocked = client.get("/ping", headers=headers)
        assert blocked.status_code == 429
        assert "Retry-After" in blocked.headers

    def test_limit_header_includes_burst_and_agrees_with_remaining(self) -> None:
        """Limit must be the effective ceiling, or Remaining can exceed it."""
        client = TestClient(_app(with_rate_limit=True, limit=5, burst=3))
        response = client.get("/ping", headers={"X-API-Key": _KEY})
        assert response.headers["X-RateLimit-Limit"] == "8"
        assert response.headers["X-RateLimit-Remaining"] == "7"

    def test_reset_header_is_a_unix_timestamp_one_window_ahead(self) -> None:
        window = 120
        client = TestClient(_app(with_rate_limit=True, window=window))
        before = time.time()
        response = client.get("/ping", headers={"X-API-Key": _KEY})
        reset = int(response.headers["X-RateLimit-Reset"])
        assert before + window - 1 <= reset <= time.time() + window + 1

    def test_retry_after_is_never_zero(self) -> None:
        """A sub-second remainder must round up, not truncate to an immediate retry."""
        client = TestClient(_app(with_rate_limit=True, limit=1, window=1))
        headers = {"X-API-Key": _KEY}
        client.get("/ping", headers=headers)
        blocked = client.get("/ping", headers=headers)
        assert blocked.status_code == 429
        assert int(blocked.headers["Retry-After"]) >= 1

    def test_rate_limit_bucket_is_keyed_by_a_digest_not_the_api_key(self) -> None:
        """The store's keyspace (Redis) and the limiter's logs must never hold the secret."""
        store = InMemoryRateLimitStore()
        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware, rate_limiter=RateLimiter(RateLimitPolicy(5, 60), store)
        )
        app.add_middleware(APIKeyMiddleware, settings=_settings())

        @app.get("/ping")
        async def ping() -> dict[str, bool]:
            return {"pong": True}

        assert TestClient(app).get("/ping", headers={"X-API-Key": _KEY}).status_code == 200

        (bucket,) = store._buckets.keys()
        assert _KEY not in bucket
        assert bucket == hashlib.sha256(_KEY.encode()).hexdigest()[:16]

    def test_public_path_is_not_rate_limited(self) -> None:
        client = TestClient(_app(with_rate_limit=True, limit=1))
        # Many hits on a public path never trip the limiter.
        for _ in range(5):
            assert client.get("/health").status_code == 200


class _UnreachableStore(InMemoryRateLimitStore):
    def record_and_get(
        self, key: str, window_seconds: int, max_entries: int | None = None
    ) -> list[float]:
        raise ConnectionError("redis:6379 refused")


class TestRateLimitBackendFailure:
    """The limiter's store (Redis) going away must not surface as a 500."""

    @staticmethod
    def _client(*, fail_open: bool) -> TestClient:
        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            rate_limiter=RateLimiter(RateLimitPolicy(5, 60), _UnreachableStore()),
            fail_open=fail_open,
        )
        app.add_middleware(APIKeyMiddleware, settings=_settings())

        @app.get("/ping")
        async def ping() -> dict[str, bool]:
            return {"pong": True}

        return TestClient(app)

    def test_fail_open_serves_the_request_unthrottled(self) -> None:
        response = self._client(fail_open=True).get("/ping", headers={"X-API-Key": _KEY})
        assert response.status_code == 200
        # No quota was computed, so none is advertised.
        assert "X-RateLimit-Remaining" not in response.headers

    def test_fail_open_still_requires_authentication(self) -> None:
        response = self._client(fail_open=True).get("/ping", headers={"X-API-Key": "wrong"})
        assert response.status_code == 403

    def test_fail_closed_refuses_with_retryable_503(self) -> None:
        response = self._client(fail_open=False).get("/ping", headers={"X-API-Key": _KEY})
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "5"
        assert "redis" not in response.text

    @pytest.mark.parametrize("fail_open", [True, False])
    def test_every_failure_is_counted(
        self, fail_open: bool, reset_prometheus_registry: None
    ) -> None:
        from aegis.infrastructure.observability.metrics import rate_limit_backend_errors_total

        client = self._client(fail_open=fail_open)
        for _ in range(3):
            client.get("/ping", headers={"X-API-Key": _KEY})
        assert rate_limit_backend_errors_total._value.get() == 3
