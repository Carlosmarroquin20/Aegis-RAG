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
  LLMTimeoutError             → 504  (the upstream was reached but did not answer)

Handlers for specific exception types are resolved inside the router, so these
responses travel back through the full middleware stack.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from aegis.domain.ports.llm_client import LLMTimeoutError, LLMUnavailableError
from aegis.domain.ports.vector_store import VectorStoreUnavailableError

logger = structlog.get_logger(__name__)

# How long a client should wait before retrying. Short on purpose: these are
# transient conditions (a restart, a model still loading), not maintenance windows.
_RETRY_AFTER_SECONDS = "5"

DEPENDENCY_UNAVAILABLE_DETAIL = "A backend dependency is temporarily unavailable. Retry shortly."
DEPENDENCY_TIMEOUT_DETAIL = "The language model did not respond in time."


def register_dependency_error_handlers(app: FastAPI) -> None:
    """Installs the handlers on ``app``. Called once from the application factory."""

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
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": DEPENDENCY_UNAVAILABLE_DETAIL},
            headers={"Retry-After": _RETRY_AFTER_SECONDS},
        )
