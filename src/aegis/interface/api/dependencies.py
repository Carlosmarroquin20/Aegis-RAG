"""
FastAPI dependency injection wiring.

This module owns the object graph construction. Swapping adapters (e.g., from
ChromaDB to pgvector) only requires a change here — no other layer is affected.

All expensive objects (embedding model, HTTP clients) are singletons cached via
lru_cache(maxsize=1). FastAPI calls dependency factories per-request by default;
the cache ensures the underlying heavy object is built only once per process.
"""

from __future__ import annotations

from functools import cache, lru_cache
from typing import Any
from urllib.parse import urlsplit

from starlette.requests import Request

from aegis.application.use_cases.ingest_documents import IngestDocumentsUseCase
from aegis.application.use_cases.query_rag import QueryRAGUseCase
from aegis.config import get_settings
from aegis.domain.models.tenant import Tenant
from aegis.domain.ports.llm_client import LLMClientPort
from aegis.domain.ports.vector_store import VectorStorePort
from aegis.infrastructure.llm.fallback import Backend, FallbackLLMClient
from aegis.infrastructure.llm.ollama_adapter import OllamaAdapter
from aegis.infrastructure.llm.openai_compatible_adapter import OpenAICompatibleAdapter
from aegis.infrastructure.observability.metrics import LLMPricing, PrometheusQueryMetrics
from aegis.infrastructure.parsers.parser_registry import ParserRegistry
from aegis.infrastructure.security.output_sanitizer import OutputSanitizer
from aegis.infrastructure.security.rate_limiter import (
    InMemoryRateLimitStore,
    RateLimiter,
    RateLimitPolicy,
    RateLimitStore,
    RedisRateLimitStore,
)
from aegis.infrastructure.security.security_gateway import SecurityGateway
from aegis.infrastructure.tenancy.registry import TenantRegistry
from aegis.infrastructure.vector_stores.chromadb_adapter import ChromaDBAdapter


@lru_cache(maxsize=1)
def get_security_gateway() -> SecurityGateway:
    cfg = get_settings()
    return SecurityGateway(strict_mode=cfg.security_strict_mode)


@lru_cache(maxsize=1)
def get_output_sanitizer() -> OutputSanitizer:
    return OutputSanitizer(strip_html=True, block_reflection=True)


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    """The deployment-wide limit, for tenants that do not set their own."""
    return get_tenant_rate_limiter(get_settings().rate_limit_requests)


@cache
def get_tenant_rate_limiter(requests_per_window: int) -> RateLimiter:
    """
    A limiter admitting ``requests_per_window`` per window, one per distinct
    limit in the tenants file. Window and burst stay deployment-wide; every
    limiter shares the one store, so the Redis window still spans replicas.
    """
    cfg = get_settings()
    policy = RateLimitPolicy(
        requests_per_window=requests_per_window,
        window_seconds=cfg.rate_limit_window_seconds,
        burst_allowance=cfg.rate_limit_burst,
    )
    return RateLimiter(policy=policy, store=get_rate_limit_store())


@lru_cache(maxsize=1)
def get_rate_limit_store() -> RateLimitStore:
    """
    The sliding-window store, shared by every limiter in the process.

    Swapping the persistence backend is a one-line change here; nothing in the
    RateLimiter, the middleware, or the routes is aware of which store is used.
    """
    cfg = get_settings()
    if cfg.rate_limit_backend == "redis":
        return RedisRateLimitStore(cfg.redis_url)
    return InMemoryRateLimitStore()


@lru_cache(maxsize=1)
def get_auth_failure_limiter() -> RateLimiter | None:
    """
    Limiter for rejected API keys, bucketed by client address. None when disabled.

    It uses the same store as the request limiter, so with the Redis backend a
    lockout holds across workers and replicas instead of being per-process.
    """
    cfg = get_settings()
    if cfg.auth_failure_limit <= 0:
        return None
    policy = RateLimitPolicy(
        requests_per_window=cfg.auth_failure_limit,
        window_seconds=cfg.auth_failure_window_seconds,
    )
    return RateLimiter(policy=policy, store=get_rate_limit_store())


@lru_cache(maxsize=1)
def get_tenant_registry() -> TenantRegistry:
    """
    The tenants and their keys: from TENANTS_FILE when set, otherwise one tenant
    per key in VALID_API_KEYS on the configured collection. Built at app creation,
    so an invalid tenants file stops the process before it serves anything.
    """
    cfg = get_settings()
    if cfg.tenants_file:
        return TenantRegistry.from_file(cfg.tenants_file)
    return TenantRegistry.from_api_keys(cfg.api_keys_set, cfg.chroma_collection)


@lru_cache(maxsize=1)
def get_parser_registry() -> ParserRegistry:
    """
    Instantiates and caches the ParserRegistry which holds all parser adapters.
    Parsers are stateless but some (e.g., DOCX) import heavy libraries on first use;
    caching avoids repeated module-level initialization.
    """
    return ParserRegistry()


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClientPort:
    """
    The LLM backend, selected by LLM_PROVIDER.

    This is the only place that knows which adapter is in use. The use case, the
    readiness probe and the shutdown hook all work through the port.
    """
    cfg = get_settings()
    primary = _llm_backend(cfg.llm_provider)
    if not cfg.llm_fallback_provider:
        return primary.client
    return FallbackLLMClient(
        [primary, _llm_backend(cfg.llm_fallback_provider)],
        failure_threshold=cfg.llm_circuit_failure_threshold,
        reset_seconds=cfg.llm_circuit_reset_seconds,
    )


def _llm_backend(provider: str) -> Backend:
    """One adapter, named the way its answers will report their provider."""
    cfg = get_settings()
    if provider == "openai_compatible":
        adapter = OpenAICompatibleAdapter(
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key.get_secret_value(),
            model=cfg.llm_model,
            timeout_seconds=cfg.llm_timeout_seconds,
        )
        return Backend(
            name=urlsplit(cfg.llm_base_url).hostname or "openai_compatible", client=adapter
        )
    ollama = OllamaAdapter(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        timeout_seconds=cfg.ollama_timeout_seconds,
    )
    return Backend(name="ollama", client=ollama)


@lru_cache(maxsize=1)
def get_embedding_function() -> Any:
    """
    The SentenceTransformers model, loaded once and shared by every collection:
    tenants differ in what they store, not in how text is embedded.
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    cfg = get_settings()
    return SentenceTransformerEmbeddingFunction(
        model_name=cfg.embedding_model,
        device=cfg.embedding_device,
    )


@cache
def get_collection_store(collection: str) -> ChromaDBAdapter:
    """
    The adapter for one collection, one per collection for the process's life.
    It is NOT initialized here — that happens in the lifespan handler so
    initialization errors surface at startup, not mid-request. The cache is
    bounded by the tenants file: only configured collections are ever asked for.
    """
    cfg = get_settings()
    return ChromaDBAdapter(
        host=cfg.chroma_host,
        port=cfg.chroma_port,
        collection_name=collection,
        embedding_function=get_embedding_function(),
        embedding_model=cfg.embedding_model,
    )


def get_chromadb_adapter() -> ChromaDBAdapter:
    """The adapter for CHROMA_COLLECTION, the collection of a single-team deployment."""
    return get_collection_store(get_settings().chroma_collection)


def get_tenant_vector_stores() -> list[ChromaDBAdapter]:
    """One adapter per collection any tenant uses, for startup and readiness."""
    collections = sorted({t.collection for t in get_tenant_registry().tenants})
    return [get_collection_store(c) for c in collections]


def request_tenant(request: Request) -> Tenant:
    """
    The tenant APIKeyMiddleware resolved for this request.

    There is deliberately no fallback to a default collection: a route reached
    without a tenant is a wiring defect, and serving it from some collection
    would hand one tenant's documents to a caller nobody authenticated as it.
    """
    tenant: Tenant | None = getattr(request.state, "tenant", None)
    if tenant is None:
        raise LookupError("no tenant on the request; is APIKeyMiddleware installed?")
    return tenant


def get_vector_store(request: Request) -> VectorStorePort:
    """
    The calling tenant's vector store, through its port. Routes and use cases
    depend on this, never on the concrete adapter, so replacing ChromaDB means
    changing this module only. Each tenant reaches its own collection and
    nothing else, so a list, a delete or a search cannot cross tenants.
    """
    return get_collection_store(request_tenant(request).collection)


def _hosted_llm_prices() -> dict[str, LLMPricing]:
    """Prices for the hosted provider, keyed the way its adapter names itself."""
    cfg = get_settings()
    if not cfg.llm_base_url:
        return {}
    host = urlsplit(cfg.llm_base_url).hostname or ""
    return {
        host: LLMPricing(
            input_per_million=cfg.llm_input_price_per_mtok,
            output_per_million=cfg.llm_output_price_per_mtok,
        )
    }


def get_query_use_case(request: Request) -> QueryRAGUseCase:
    """FastAPI dependency factory for the query pipeline, bound to the caller's tenant."""
    return QueryRAGUseCase(
        vector_store=get_vector_store(request),
        llm_client=get_llm_client(),
        security_gateway=get_security_gateway(),
        output_sanitizer=get_output_sanitizer(),
        metrics=PrometheusQueryMetrics(_hosted_llm_prices()),
    )


def get_ingest_use_case(request: Request) -> IngestDocumentsUseCase:
    """FastAPI dependency factory for the ingestion pipeline, into the caller's collection."""
    return build_ingest_use_case(request_tenant(request).collection)


def build_ingest_use_case(collection: str) -> IngestDocumentsUseCase:
    """The ingestion pipeline for one collection; also used by the seed script."""
    return IngestDocumentsUseCase(
        vector_store=get_collection_store(collection),
        parser_registry=get_parser_registry(),
        collection=collection,
    )
