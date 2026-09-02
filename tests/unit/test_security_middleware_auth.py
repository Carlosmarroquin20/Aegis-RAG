"""
Unit tests for the authentication and rate-limiting middleware.

These complement test_middleware.py (which covers RequestID / AccessLog /
SecurityHeaders) by exercising the APIKeyMiddleware and RateLimitMiddleware
dispatch paths against a minimal app.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegis.config import Settings
from aegis.infrastructure.security.rate_limiter import RateLimiter, RateLimitPolicy
from aegis.interface.api.middleware.security_middleware import (
    APIKeyMiddleware,
    RateLimitMiddleware,
)

_KEY = "secret-key"


def _settings() -> Settings:
    return Settings(valid_api_keys=_KEY, api_key_header="X-API-Key")


def _app(*, with_rate_limit: bool = False, limit: int = 5) -> FastAPI:
    app = FastAPI()
    if with_rate_limit:
        limiter = RateLimiter(RateLimitPolicy(limit, 3600, burst_allowance=0))
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

    def test_public_path_is_not_rate_limited(self) -> None:
        client = TestClient(_app(with_rate_limit=True, limit=1))
        # Many hits on a public path never trip the limiter.
        for _ in range(5):
            assert client.get("/health").status_code == 200
