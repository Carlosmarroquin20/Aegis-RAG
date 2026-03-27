"""
Health and readiness endpoints.

/health  — shallow liveness check (no external calls); used by load balancers.
/ready   — deep readiness check (verifies vector store + LLM connectivity).
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from aegis.interface.api.dependencies import get_chromadb_adapter, get_ollama_adapter

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
    vector_store_ok = await get_chromadb_adapter().health_check()
    llm_ok = await get_ollama_adapter().health_check()

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
