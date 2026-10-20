"""
Health, readiness, and observability endpoints.

/health   — shallow liveness check (no external calls); used by load balancers.
/ready    — deep readiness check (verifies vector store + LLM connectivity).
/metrics  — Prometheus exposition format for scraping.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, Response
from prometheus_client.exposition import choose_encoder
from pydantic import BaseModel

from aegis.infrastructure.observability.metrics import registry
from aegis.interface.api.dependencies import get_llm_client, get_vector_store

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["ops"])


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    components: dict[str, bool]


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    from aegis.config import get_settings

    cfg = get_settings()
    return HealthResponse(status="ok", version=cfg.app_version)


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def ready() -> JSONResponse:
    """
    Checks connectivity to all external dependencies.
    Returns HTTP 503 if any component is unavailable so that orchestrators
    (Kubernetes, ECS) hold traffic until the service is fully operational.
    """
    vector_store_ok = await get_vector_store().health_check()
    llm_ok = await get_llm_client().health_check()

    components = {"vector_store": vector_store_ok, "llm": llm_ok}
    all_ok = all(components.values())

    if not all_ok:
        logger.warning("readiness.degraded", components=components)

    return JSONResponse(
        status_code=status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ReadinessResponse(
            status="ready" if all_ok else "degraded",
            components=components,
        ).model_dump(),
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics endpoint",
    include_in_schema=False,
    response_class=Response,
)
async def metrics(request: Request) -> Response:
    """
    Exposes counters and histograms in the format the scraper asks for.

    OpenMetrics when the Accept header requests it, as Prometheus does by
    default, and the classic text format otherwise. OpenMetrics carries each
    counter's creation time, which lets Prometheus count the increments a new
    series made before its first scrape; without it the first queries after a
    restart vanish from increase(), and the cost panels under-report.

    This endpoint is public at the application layer (no API key) so that
    scrape agents do not require rotating credentials; protect it at the
    network layer (e.g., a private listener or ingress ACL).
    """
    encoder, content_type = choose_encoder(request.headers.get("accept", ""))
    return Response(content=encoder(registry), media_type=content_type)
