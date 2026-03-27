"""
FastAPI dependency injection wiring.

This module owns the object graph construction. Swapping adapters (e.g., from
ChromaDB to pgvector) only requires a change here — no other layer is affected.
"""
from __future__ import annotations

from functools import lru_cache

from aegis.application.use_cases.query_rag import QueryRAGUseCase
from aegis.config import Settings, get_settings
from aegis.infrastructure.llm.ollama_adapter import OllamaAdapter
from aegis.infrastructure.security.rate_limiter import RateLimitPolicy, RateLimiter
from aegis.infrastructure.security.security_gateway import SecurityGateway
from aegis.infrastructure.vector_stores.chromadb_adapter import ChromaDBAdapter


@lru_cache(maxsize=1)
def get_security_gateway(settings: Settings | None = None) -> SecurityGateway:
    cfg = settings or get_settings()
    return SecurityGateway(strict_mode=cfg.security_strict_mode)


@lru_cache(maxsize=1)
def get_rate_limiter(settings: Settings | None = None) -> RateLimiter:
    cfg = settings or get_settings()
    policy = RateLimitPolicy(
        requests_per_window=cfg.rate_limit_requests,
        window_seconds=cfg.rate_limit_window_seconds,
        burst_allowance=cfg.rate_limit_burst,
    )
    return RateLimiter(policy=policy)


def get_ollama_adapter(settings: Settings | None = None) -> OllamaAdapter:
    cfg = settings or get_settings()
    return OllamaAdapter(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        timeout_seconds=cfg.ollama_timeout_seconds,
    )


def get_chromadb_adapter(settings: Settings | None = None) -> ChromaDBAdapter:
    """
    Builds the ChromaDB adapter with a SentenceTransformers embedding function.
    The adapter is NOT initialized here — that happens in the lifespan handler.
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    cfg = settings or get_settings()
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=cfg.embedding_model,
        device=cfg.embedding_device,
    )
    return ChromaDBAdapter(
        host=cfg.chroma_host,
        port=cfg.chroma_port,
        collection_name=cfg.chroma_collection,
        embedding_function=embedding_fn,
    )


def get_query_use_case() -> QueryRAGUseCase:
    """
    FastAPI dependency factory.
    Adapters are module-level singletons accessed via the cached getters above.
    Re-creating them per-request would be wasteful (embedding models are large).
    """
    return QueryRAGUseCase(
        vector_store=get_chromadb_adapter(),
        llm_client=get_ollama_adapter(),
        security_gateway=get_security_gateway(),
    )
