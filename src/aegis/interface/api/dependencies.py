"""
FastAPI dependency injection wiring.

This module owns the object graph construction. Swapping adapters (e.g., from
ChromaDB to pgvector) only requires a change here — no other layer is affected.

All expensive objects (embedding model, HTTP clients) are singletons cached via
lru_cache(maxsize=1). FastAPI calls dependency factories per-request by default;
the cache ensures the underlying heavy object is built only once per process.
"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlsplit

from aegis.application.use_cases.ingest_documents import IngestDocumentsUseCase
from aegis.application.use_cases.query_rag import QueryRAGUseCase
from aegis.config import get_settings
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
    cfg = get_settings()
    policy = RateLimitPolicy(
        requests_per_window=cfg.rate_limit_requests,
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
def get_chromadb_adapter() -> ChromaDBAdapter:
    """
    Builds the ChromaDB adapter with a SentenceTransformers embedding function.
    The adapter is NOT initialized here — that happens in the lifespan handler
    so initialization errors surface at startup, not mid-request.
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    cfg = get_settings()
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=cfg.embedding_model,
        device=cfg.embedding_device,
    )
    return ChromaDBAdapter(
        host=cfg.chroma_host,
        port=cfg.chroma_port,
        collection_name=cfg.chroma_collection,
        embedding_function=embedding_fn,
        embedding_model=cfg.embedding_model,
    )


def get_vector_store() -> VectorStorePort:
    """
    The vector store as the rest of the application sees it: through its port.
    Routes and use cases depend on this, never on the concrete adapter, so
    replacing ChromaDB means changing this one return statement.
    """
    return get_chromadb_adapter()


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


def get_query_use_case() -> QueryRAGUseCase:
    """FastAPI dependency factory for the query pipeline."""
    return QueryRAGUseCase(
        vector_store=get_vector_store(),
        llm_client=get_llm_client(),
        security_gateway=get_security_gateway(),
        output_sanitizer=get_output_sanitizer(),
        metrics=PrometheusQueryMetrics(_hosted_llm_prices()),
    )


def get_ingest_use_case() -> IngestDocumentsUseCase:
    """FastAPI dependency factory for the ingestion pipeline."""
    cfg = get_settings()
    return IngestDocumentsUseCase(
        vector_store=get_vector_store(),
        parser_registry=get_parser_registry(),
        collection=cfg.chroma_collection,
    )
