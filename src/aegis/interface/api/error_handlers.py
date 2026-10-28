"""
Exception handlers that translate dependency failures into HTTP semantics.

A failing vector store or LLM is not a bug in this service, and reporting it as
a 500 tells the caller the wrong thing: 5xx dashboards blame the API, clients do
not know the request is safe to retry, and load balancers cannot distinguish a
broken instance from a healthy one waiting on a slow model.

The adapters raise the domain-level errors below; this module is the single
place where they become status codes:

  VectorStoreUnavailableError → 503
  LLMUnavailableError         → 503
  LLMRateLimitedError         → 503, with the provider's own Retry-After when it gave one
  LLMTimeoutError             → 504  (the upstream was reached but did not answer)
  TokenBudgetExhaustedError   → 429, Retry-After until the budget resets at UTC midnight

Handlers for specific exception types are resolved inside the router, so these
responses travel back through the full middleware stack.
"""

from __future__ import annotations

import math

import structlog
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from aegis.domain.ports.llm_client import (
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from aegis.domain.ports.token_ledger import TokenBudgetExhaustedError
from aegis.domain.ports.vector_store import VectorStoreUnavailableError

logger = structlog.get_logger(__name__)

# How long a client should wait before retrying. Short on purpose: these are
# transient conditions (a restart, a model still loading), not maintenance windows.
_RETRY_AFTER_SECONDS = "5"
# A provider's quota window can be long; a client told to wait an hour would
# rather hear "try again in a minute" and get a fresh answer then.
_MAX_RETRY_AFTER_SECONDS = 60

DEPENDENCY_UNAVAILABLE_DETAIL = "A backend dependency is temporarily unavailable. Retry shortly."
DEPENDENCY_TIMEOUT_DETAIL = "The language model did not respond in time."
BUDGET_EXHAUSTED_DETAIL = "Daily token budget exhausted for this tenant."


def register_dependency_error_handlers(app: FastAPI) -> None:
    """Installs the handlers on ``app``. Called once from the application factory."""

    @app.exception_handler(TokenBudgetExhaustedError)
    async def token_budget_exhausted(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, TokenBudgetExhaustedError)
        # Unlike a provider's quota, this one resets at a known time, so the
        # client is told exactly when rather than to poll.
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": BUDGET_EXHAUSTED_DETAIL},
            headers={"Retry-After": str(max(1, math.ceil(exc.retry_after)))},
        )

    @app.exception_handler(VectorStoreUnavailableError)
    @app.exception_handler(LLMUnavailableError)
    async def dependency_unavailable(request: Request, exc: Exception) -> JSONResponse:
        timed_out = isinstance(exc, LLMTimeoutError)
        # The cause goes to the log only. The response stays generic so it does
        # not disclose hostnames, ports or vendor error text.
        logger.error(
            "dependency.unavailable",
            dependency="llm" if isinstance(exc, LLMUnavailableError) else "vector_store",
            timed_out=timed_out,
            path=request.url.path,
            error=str(exc),
        )
        if timed_out:
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={"detail": DEPENDENCY_TIMEOUT_DETAIL},
            )
        retry_after = _RETRY_AFTER_SECONDS
        if isinstance(exc, LLMRateLimitedError) and exc.retry_after is not None:
            retry_after = str(min(_MAX_RETRY_AFTER_SECONDS, max(1, math.ceil(exc.retry_after))))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": DEPENDENCY_UNAVAILABLE_DETAIL},
            headers={"Retry-After": retry_after},
        )
