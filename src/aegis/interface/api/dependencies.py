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

from aegis.application.use_cases.ingest_documents import IngestDocumentsUseCase
from aegis.application.use_cases.query_rag import QueryRAGUseCase
from aegis.config import Settings, get_settings
from aegis.infrastructure.llm.ollama_adapter import OllamaAdapter
from aegis.infrastructure.parsers.parser_registry import ParserRegistry
from aegis.infrastructure.security.output_sanitizer import OutputSanitizer
from aegis.infrastructure.security.rate_limiter import RateLimiter, RateLimitPolicy
from aegis.infrastructure.security.security_gateway import SecurityGateway
from aegis.infrastructure.vector_stores.chromadb_adapter import ChromaDBAdapter


@lru_cache(maxsize=1)
def get_security_gateway(settings: Settings | None = None) -> SecurityGateway:
    cfg = settings or get_settings()
    return SecurityGateway(strict_mode=cfg.security_strict_mode)


@lru_cache(maxsize=1)
def get_output_sanitizer() -> OutputSanitizer:
    return OutputSanitizer(strip_html=True, block_reflection=True)


@lru_cache(maxsize=1)
def get_rate_limiter(settings: Settings | None = None) -> RateLimiter:
    cfg = settings or get_settings()
    policy = RateLimitPolicy(
        requests_per_window=cfg.rate_limit_requests,
        window_seconds=cfg.rate_limit_window_seconds,
        burst_allowance=cfg.rate_limit_burst,
    )
    return RateLimiter(policy=policy)


@lru_cache(maxsize=1)
def get_parser_registry() -> ParserRegistry:
    """
    Instantiates and caches the ParserRegistry which holds all parser adapters.
    Parsers are stateless but some (e.g., DOCX) import heavy libraries on first use;
    caching avoids repeated module-level initialization.
    """
    return ParserRegistry()


@lru_cache(maxsize=1)
def get_ollama_adapter(settings: Settings | None = None) -> OllamaAdapter:
    cfg = settings or get_settings()
    return OllamaAdapter(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        timeout_seconds=cfg.ollama_timeout_seconds,
    )


@lru_cache(maxsize=1)
def get_chromadb_adapter(settings: Settings | None = None) -> ChromaDBAdapter:
    """
    Builds the ChromaDB adapter with a SentenceTransformers embedding function.
    The adapter is NOT initialized here — that happens in the lifespan handler
    so initialization errors surface at startup, not mid-request.
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
    """FastAPI dependency factory for the query pipeline."""
    return QueryRAGUseCase(
        vector_store=get_chromadb_adapter(),
        llm_client=get_ollama_adapter(),
        security_gateway=get_security_gateway(),
        output_sanitizer=get_output_sanitizer(),
    )


def get_ingest_use_case() -> IngestDocumentsUseCase:
    """FastAPI dependency factory for the ingestion pipeline."""
    cfg = get_settings()
    return IngestDocumentsUseCase(
        vector_store=get_chromadb_adapter(),
        parser_registry=get_parser_registry(),
        default_collection=cfg.chroma_collection,
    )
