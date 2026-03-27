"""
ASGI Security Middleware Stack.

Responsibilities separated by concern:
  - APIKeyMiddleware   : authenticates the caller (identity layer)
  - RateLimitMiddleware: enforces per-key request quotas (throttle layer)

These run before any route handler is invoked. Placing them in middleware
(rather than FastAPI dependencies) ensures they execute even for 404/405 responses,
preventing information leakage about undiscovered endpoints.
"""
from __future__ import annotations

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from aegis.config import Settings
from aegis.infrastructure.security.rate_limiter import RateLimiter

logger = structlog.get_logger(__name__)

# Paths excluded from API key enforcement (public endpoints).
_PUBLIC_PATHS: frozenset[str] = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates the X-API-Key header against the configured key set.

    Security notes:
    - Uses a constant-time set lookup (frozenset.__contains__) rather than
      a linear search or ORM query to avoid timing side-channels.
    - Returns 403 (not 401) for missing keys to avoid revealing that
      authentication via API key is expected on this endpoint.
    - The key is stored in request.state for downstream use (e.g., rate limiter).
    """

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        super().__init__(app)
        self._header_name = settings.api_key_header
        self._valid_keys = settings.api_keys_set

    async def dispatch(self, request: Request, call_next: object) -> Response:
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)  # type: ignore[operator]

        api_key = request.headers.get(self._header_name, "")
        if not api_key or api_key not in self._valid_keys:
            logger.warning(
                "auth.rejected",
                path=request.url.path,
                ip=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden: valid API key required."},
            )

        # Attach the validated key to request state for the rate limiter.
        request.state.api_key = api_key
        return await call_next(request)  # type: ignore[operator]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter enforced per validated API key.
    Must run AFTER APIKeyMiddleware so request.state.api_key is populated.
    """

    def __init__(self, app: ASGIApp, rate_limiter: RateLimiter) -> None:
        super().__init__(app)
        self._limiter = rate_limiter

    async def dispatch(self, request: Request, call_next: object) -> Response:
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)  # type: ignore[operator]

        api_key: str = getattr(request.state, "api_key", "anonymous")
        result = self._limiter.check_and_record(api_key)

        # Always attach rate-limit headers, even on allowed requests.
        # This gives clients visibility into their quota without requiring an extra call.
        headers = {
            "X-RateLimit-Limit": str(
                self._limiter._policy.requests_per_window  # noqa: SLF001
            ),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(int(result.reset_at + time.time() - time.monotonic())),
        }

        if not result.allowed:
            headers["Retry-After"] = str(int(result.retry_after or 1))
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down."},
                headers=headers,
            )

        response: Response = await call_next(request)  # type: ignore[operator]
        for header_name, header_value in headers.items():
            response.headers[header_name] = header_value
        return response
