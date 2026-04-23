"""
FastAPI application entry point.

Wiring order matters:
  1. Logging is configured before anything else so startup errors are captured.
  2. Middleware is added in reverse execution order (last added = first to run).
     Stack order (request →): RequestID → AccessLog → SecurityHeaders → RateLimit → APIKey → Routes
  3. The lifespan context manager handles adapter initialization and cleanup.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aegis.config import get_settings
from aegis.interface.api.dependencies import (
    get_chromadb_adapter,
    get_ollama_adapter,
    get_parser_registry,
    get_rate_limiter,
    get_security_gateway,
)
from aegis.interface.api.middleware.security_middleware import (
    AccessLogMiddleware,
    APIKeyMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
)
from aegis.interface.api.routes import documents, health, query


def _configure_logging(settings_obj: object) -> None:
    """
    Configures structlog for either machine-readable JSON (production)
    or human-readable console output (development).
    """
    from aegis.config import Settings

    cfg: Settings = settings_obj  # type: ignore[assignment]

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if cfg.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()  # type: ignore[assignment]

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(cfg.log_level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Handles resource initialization on startup and cleanup on shutdown.
    Adapter initialization is deferred to this context so it benefits from
    async I/O and so failures surface clearly during startup, not mid-request.
    """
    logger = structlog.get_logger(__name__)
    cfg = get_settings()

    logger.info("startup.begin", app=cfg.app_name, version=cfg.app_version)

    # Warm up the security gateway (pre-compiles regex patterns).
    get_security_gateway(cfg)

    # Pre-load the parser registry (deferred imports in parsers run at first get).
    get_parser_registry()

    # Initialize the vector store (creates collection if absent).
    vector_store = get_chromadb_adapter(cfg)
    await vector_store.initialize()

    logger.info("startup.complete")
    yield

    # Graceful shutdown: release the LLM HTTP client connection pool.
    logger.info("shutdown.begin")
    llm = get_ollama_adapter(cfg)
    await llm.aclose()
    logger.info("shutdown.complete")


def create_app() -> FastAPI:
    cfg = get_settings()
    _configure_logging(cfg)

    app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        description=(
            "Hardened Retrieval-Augmented Generation API with OWASP LLM Top 10 controls. "
            "All queries are evaluated by the SecurityGateway before reaching the RAG pipeline."
        ),
        docs_url="/docs" if cfg.debug else None,  # Disable Swagger UI in production.
        redoc_url="/redoc" if cfg.debug else None,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # In production, set CORS_ALLOWED_ORIGINS to the exact frontend domains.
    # Debug mode permits all origins for local development convenience.
    origins = ["*"] if cfg.debug else cfg.cors_allowed_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", cfg.api_key_header],
    )

    # ── Security & Observability Middleware Stack ─────────────────────────────
    # Middleware executes in reverse registration order (last added = first to run).
    # Stack (request →): RequestID → AccessLog → SecurityHeaders → RateLimit → APIKey → Routes
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, rate_limiter=get_rate_limiter(cfg))
    app.add_middleware(APIKeyMiddleware, settings=cfg)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(documents.router)

    # ── Global Exception Handlers ─────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        structlog.get_logger(__name__).error(
            "unhandled_exception",
            path=request.url.path,
            exc_type=type(exc).__name__,
        )
        # Never expose internal exception details in production responses.
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Check server logs."},
        )

    return app


# Expose the app instance for uvicorn: `uvicorn aegis.interface.api.main:app`
app = create_app()
