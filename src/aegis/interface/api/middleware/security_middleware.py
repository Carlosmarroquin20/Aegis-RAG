"""
ASGI Security & Observability Middleware Stack.

Responsibilities separated by concern:
  - RequestIDMiddleware       : assigns a unique correlation ID to every request
  - AccessLogMiddleware       : structured access logging with response timing
  - SecurityHeadersMiddleware : defence-in-depth HTTP headers on every response
  - APIKeyMiddleware          : authenticates the caller (identity layer)
  - RateLimitMiddleware       : enforces per-key request quotas (throttle layer)
  - ErrorBoundaryMiddleware   : turns unhandled route errors into a 500 response
                                that still travels back through the stack above

These run before any route handler is invoked. Placing them in middleware
(rather than FastAPI dependencies) ensures they execute even for 404/405 responses,
preventing information leakage about undiscovered endpoints.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import time
import uuid

import structlog
from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from aegis.config import Settings
from aegis.infrastructure.observability.metrics import (
    http_request_duration_seconds,
    http_requests_total,
    rate_limit_backend_errors_total,
)
from aegis.infrastructure.security.rate_limiter import RateLimiter

logger = structlog.get_logger(__name__)

# Paths excluded from API key enforcement and rate limiting (public endpoints).
# /metrics is scraped by Prometheus on the internal network; protect it via
# network-level ACLs (e.g., a private listener) rather than the API key.
_PUBLIC_PATHS: frozenset[str] = frozenset(
    {"/health", "/ready", "/metrics", "/docs", "/openapi.json", "/redoc"}
)

_API_CSP = "default-src 'none'; frame-ancestors 'none'"

# Paths that serve the interactive documentation pages (HTML, not JSON).
_DOCS_PATHS: frozenset[str] = frozenset({"/docs", "/redoc"})

# What FastAPI's bundled Swagger UI and ReDoc pages actually request: the bundle
# and stylesheet from jsDelivr, an inline bootstrap script, Google Fonts (ReDoc),
# a favicon, the schema from this origin, and a blob: worker (ReDoc search).
# 'unsafe-inline' is confined to these two debug-only pages.
_DOCS_CSP = "; ".join(
    [
        "default-src 'none'",
        "script-src 'unsafe-inline' https://cdn.jsdelivr.net",
        "style-src 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
        "font-src https://fonts.gstatic.com",
        "img-src data: https://fastapi.tiangolo.com https://cdn.redoc.ly",
        "connect-src 'self'",
        "worker-src blob:",
        "frame-ancestors 'none'",
    ]
)

INTERNAL_ERROR_DETAIL = "An internal error occurred. Check server logs."

# Shape of a caller-supplied correlation ID we are willing to propagate. Covers
# UUIDs (with or without dashes), ULIDs and W3C trace IDs. ASCII is spelled out
# because \w would also admit Unicode letters.
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._\-]{1,64}")

# 64 bits of the key digest: ample to keep clients apart, useless for recovering
# or verifying a key offline.
_CLIENT_ID_HEX_CHARS = 16


def _digest(api_key: str) -> bytes:
    return hashlib.sha256(api_key.encode()).digest()


class ErrorBoundaryMiddleware(BaseHTTPMiddleware):
    """
    Converts an unhandled exception from the routes into a generic 500 response.

    Must be the INNERMOST middleware. Starlette routes a catch-all
    ``Exception`` handler through ServerErrorMiddleware, which sits outside
    every user middleware; a response built there never passes back through the
    stack. Without this boundary a 500 would therefore ship without security
    headers or an ``X-Request-ID``, and would be missing from the access log and
    from ``aegis_http_requests_total`` — hiding failures from the 5xx alert.

    HTTPException and request-validation errors are unaffected: Starlette
    resolves those inside the router, before they can reach this layer.
    """

    async def dispatch(self, request: Request, call_next: object) -> Response:
        try:
            return await call_next(request)  # type: ignore[no-any-return,operator]
        except Exception:
            logger.exception("unhandled_exception", method=request.method, path=request.url.path)
            # Never expose internal exception details to the caller.
            return JSONResponse(status_code=500, content={"detail": INTERNAL_ERROR_DETAIL})


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Assigns a unique ID to every request for end-to-end log correlation.

    If the caller provides a well-formed X-Request-ID header (e.g., from an API
    gateway), it is reused; otherwise a UUID4 is generated. The ID is:
      1. Bound to structlog context vars so every log line includes it.
      2. Echoed back in the X-Request-ID response header for client-side tracing.

    The inbound value is untrusted and ends up in every log line, a span
    attribute and a response header, so it is accepted only if it looks like an
    identifier. Anything else (oversized, whitespace, control or non-ASCII
    characters) is replaced rather than rejected: a bad correlation header is
    not a reason to fail the request.
    """

    async def dispatch(self, request: Request, call_next: object) -> Response:
        inbound = request.headers.get("X-Request-ID", "")
        request_id = inbound if _REQUEST_ID_PATTERN.fullmatch(inbound) else uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        # Tie request_id to the active server span (no-op when tracing is off).
        trace.get_current_span().set_attribute("aegis.request_id", request_id)

        try:
            response: Response = await call_next(request)  # type: ignore[operator]
        finally:
            # Unbind even if the stack below raises, so the ID cannot bleed into
            # log lines emitted later from the same context.
            structlog.contextvars.unbind_contextvars("request_id")

        response.headers["X-Request-ID"] = request_id
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """
    Structured access log + Prometheus instrumentation.

    Each request produces:
      - one structured log line with method, path, status, duration_ms, IP
      - one increment of ``aegis_http_requests_total{method,path,status}``
      - one observation on ``aegis_http_request_duration_seconds{method,path}``

    The ``X-Response-Time`` header is also set so callers can measure latency
    from their side. The ``path`` label uses the matched route template
    (e.g. ``/api/v1/documents/{doc_id}``) when available; this prevents label
    cardinality explosion from path parameters.
    """

    async def dispatch(self, request: Request, call_next: object) -> Response:
        start = time.monotonic()
        response: Response = await call_next(request)  # type: ignore[operator]
        duration_s = time.monotonic() - start
        duration_ms = round(duration_s * 1000, 2)

        response.headers["X-Response-Time"] = f"{duration_ms}ms"

        # Prefer the matched route template over the raw URL path to keep
        # Prometheus label cardinality bounded.
        route = request.scope.get("route")
        metric_path = getattr(route, "path", None) or request.url.path

        http_requests_total.labels(
            method=request.method,
            path=metric_path,
            status=str(response.status_code),
        ).inc()
        http_request_duration_seconds.labels(
            method=request.method,
            path=metric_path,
        ).observe(duration_s)

        logger.info(
            "http.access",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
            client_ip=request.client.host if request.client else "unknown",
        )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Injects defence-in-depth HTTP headers into every response.

    These headers mitigate common browser-side attacks (XSS, clickjacking,
    MIME sniffing) and signal HTTPS enforcement to compliant user-agents.

    Every response gets ``default-src 'none'``: this is a JSON API, so nothing it
    returns should ever load a sub-resource. The one exception is the interactive
    documentation, which is an HTML page that pulls its bundle from a CDN. When
    ``docs_enabled`` is set (debug only) those two paths get a policy scoped to
    exactly what Swagger UI and ReDoc load; under the strict policy they render
    as a blank page.
    """

    def __init__(self, app: ASGIApp, docs_enabled: bool = False) -> None:
        super().__init__(app)
        self._docs_enabled = docs_enabled

    async def dispatch(self, request: Request, call_next: object) -> Response:
        response: Response = await call_next(request)  # type: ignore[operator]
        relax_for_docs = self._docs_enabled and request.url.path in _DOCS_PATHS
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "0"  # Modern best practice: rely on CSP instead
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = _DOCS_CSP if relax_for_docs else _API_CSP
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates the X-API-Key header against the configured key set.

    Security notes:
    - Keys are compared as SHA-256 digests with hmac.compare_digest, against
      every configured key and without an early exit. A set lookup is not
      constant-time: it hashes the candidate and then runs a short-circuiting
      string comparison on the bucket hit. Digesting first also equalizes the
      length, which compare_digest would otherwise leak.
    - Returns 403 (not 401) for missing keys to avoid revealing that
      authentication via API key is expected on this endpoint.
    - Downstream layers receive ``request.state.client_id``, a truncated digest
      of the key, never the key itself. It is a stable per-key identifier that
      is safe to use as a rate-limit bucket name and to write to logs.
    """

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        super().__init__(app)
        self._header_name = settings.api_key_header
        self._valid_key_digests = tuple(_digest(key) for key in settings.api_keys_set)

    async def dispatch(self, request: Request, call_next: object) -> Response:
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)  # type: ignore[no-any-return,operator]

        candidate = _digest(request.headers.get(self._header_name, ""))
        authenticated = False
        for valid in self._valid_key_digests:
            # Bitwise OR, not `or`/any(): every key is compared on every request.
            authenticated |= hmac.compare_digest(candidate, valid)

        if not authenticated:
            logger.warning(
                "auth.rejected",
                path=request.url.path,
                ip=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden: valid API key required."},
            )

        request.state.client_id = candidate.hex()[:_CLIENT_ID_HEX_CHARS]
        return await call_next(request)  # type: ignore[no-any-return,operator]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter enforced per validated API key.
    Must run AFTER APIKeyMiddleware so request.state.client_id is populated.

    ``fail_open`` decides what happens when the limiter's store is unreachable
    (see ``Settings.rate_limit_fail_open`` for the trade-off): serve the request
    without throttling, or refuse it with a 503. An unhandled store error would
    otherwise escape as a 500 on every request for as long as the store is down.
    """

    def __init__(self, app: ASGIApp, rate_limiter: RateLimiter, fail_open: bool = True) -> None:
        super().__init__(app)
        self._limiter = rate_limiter
        self._fail_open = fail_open

    async def dispatch(self, request: Request, call_next: object) -> Response:
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)  # type: ignore[no-any-return,operator]

        client_id: str = getattr(request.state, "client_id", "anonymous")
        try:
            result = self._limiter.check_and_record(client_id)
        except Exception as exc:
            rate_limit_backend_errors_total.inc()
            logger.error(
                "rate_limit.backend_unavailable",
                fail_open=self._fail_open,
                error=repr(exc),
            )
            if self._fail_open:
                return await call_next(request)  # type: ignore[no-any-return,operator]
            return JSONResponse(
                status_code=503,
                content={"detail": "Service temporarily unavailable. Retry shortly."},
                headers={"Retry-After": "5"},
            )

        # Always attach rate-limit headers, even on allowed requests.
        # This gives clients visibility into their quota without requiring an extra call.
        # Limit is the effective ceiling (baseline + burst) so that it agrees with
        # Remaining; reset_at is already a Unix wall-clock timestamp.
        headers = {
            "X-RateLimit-Limit": str(self._limiter.limit),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(math.ceil(result.reset_at)),
        }

        if not result.allowed:
            # Round up: truncating a sub-second wait to "0" tells the client to
            # retry immediately, straight into another 429.
            headers["Retry-After"] = str(max(1, math.ceil(result.retry_after or 0)))
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down."},
                headers=headers,
            )

        response: Response = await call_next(request)  # type: ignore[operator]
        for header_name, header_value in headers.items():
            response.headers[header_name] = header_value
        return response
