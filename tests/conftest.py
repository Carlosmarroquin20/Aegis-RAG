"""
Shared pytest fixtures for the Aegis-RAG test suite.

This module centralizes the test doubles and helpers that multiple test files
need: fake port implementations, a minimal settings object, and FastAPI app
builders that wire the full middleware stack against in-memory adapters.

Design notes:
  - Fakes implement the domain ports directly (not Mock objects) so they stay
    in sync with the port contract at type-check time.
  - Fixtures return fresh instances per test (pytest's default scope) to
    guarantee isolation; the Prometheus registry is the one exception — it is
    process-global by design and is reset between tests where it matters.
  - This file must NOT import anything that performs network or filesystem I/O
    at import time; fixtures handle lazy setup.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegis.application.dtos.rag_dtos import QueryRequest
from aegis.config import Settings
from aegis.domain.models.document import Document
from aegis.domain.ports.llm_client import LLMClientPort
from aegis.domain.ports.vector_store import VectorStorePort
from aegis.infrastructure.parsers.parser_registry import ParserRegistry
from aegis.infrastructure.security.output_sanitizer import OutputSanitizer
from aegis.infrastructure.security.rate_limiter import RateLimiter, RateLimitPolicy
from aegis.infrastructure.security.security_gateway import SecurityGateway

# ── Constants ────────────────────────────────────────────────────────────────

TEST_API_KEY = "test-api-key-unit-suite"
"""API key used by the test TestClient and middleware fixtures."""

TEST_API_KEY_HEADER = "X-API-Key"


# ── Test doubles (port implementations) ──────────────────────────────────────


class FakeVectorStore(VectorStorePort):
    """
    In-memory VectorStorePort implementation.

    Records every upsert/delete for assertions, returns a configurable list
    of documents from similarity_search. Tests should mutate ``stored`` or
    ``next_search_results`` directly when they need specific retrieval output.
    """

    def __init__(
        self,
        *,
        healthy: bool = True,
        next_search_results: list[Document] | None = None,
    ) -> None:
        self.stored: list[Document] = []
        self.deleted_ids: list[str] = []
        self.next_search_results: list[Document] = next_search_results or []
        self._healthy = healthy

    async def similarity_search(
        self, query: str, k: int = 5, score_threshold: float = 0.0
    ) -> list[Document]:
        return list(self.next_search_results[:k])

    async def add_documents(self, documents: list[Document]) -> None:
        self.stored.extend(documents)

    async def delete_documents(self, ids: list[str]) -> None:
        self.deleted_ids.extend(ids)
        self.stored = [d for d in self.stored if d.id not in ids]

    async def health_check(self) -> bool:
        return self._healthy


class FakeLLMClient(LLMClientPort):
    """
    In-memory LLMClientPort implementation.

    Returns ``fixed_response`` on every ``generate`` call and records the
    arguments for assertions. Useful for testing orchestration logic without
    the latency and non-determinism of a real LLM.
    """

    def __init__(
        self,
        *,
        fixed_response: str = "The answer is grounded in the provided context.",
        healthy: bool = True,
    ) -> None:
        self.fixed_response = fixed_response
        self._healthy = healthy
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        query: str,
        context: list[Document],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        self.calls.append(
            {
                "query": query,
                "context_len": len(context),
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return self.fixed_response

    async def health_check(self) -> bool:
        return self._healthy


# ── Core fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def fake_vector_store() -> FakeVectorStore:
    """Fresh in-memory vector store per test."""
    return FakeVectorStore()


@pytest.fixture()
def fake_llm_client() -> FakeLLMClient:
    """Fresh in-memory LLM client per test."""
    return FakeLLMClient()


@pytest.fixture()
def parser_registry() -> ParserRegistry:
    """Real ParserRegistry — stateless and safe to reuse."""
    return ParserRegistry()


@pytest.fixture()
def security_gateway_strict() -> SecurityGateway:
    """SecurityGateway in production-like strict mode (blocks SUSPICIOUS too)."""
    return SecurityGateway(strict_mode=True)


@pytest.fixture()
def security_gateway_permissive() -> SecurityGateway:
    """SecurityGateway in permissive mode (only blocks BLOCKED level)."""
    return SecurityGateway(strict_mode=False)


@pytest.fixture()
def output_sanitizer() -> OutputSanitizer:
    """OutputSanitizer with production defaults (HTML strip, reflection blocking)."""
    return OutputSanitizer(strip_html=True, block_reflection=True)


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    """RateLimiter with a generous policy so tests do not trip quotas accidentally."""
    return RateLimiter(
        policy=RateLimitPolicy(
            requests_per_window=1000,
            window_seconds=60,
            burst_allowance=100,
        )
    )


# ── Sample data fixtures ─────────────────────────────────────────────────────


@pytest.fixture()
def sample_document() -> Document:
    """A single Document ready for retrieval-path tests."""
    return Document(
        id="doc-1",
        content="Aegis-RAG routes every query through a security gateway before retrieval.",
        metadata={"source": "readme.md", "chunk_index": "0"},
        relevance_score=0.92,
    )


@pytest.fixture()
def sample_documents(sample_document: Document) -> list[Document]:
    """A small corpus of three Documents for top-k retrieval tests."""
    return [
        sample_document,
        Document(
            id="doc-2",
            content="The SecurityGateway applies 11 signature rules and entropy analysis.",
            metadata={"source": "readme.md", "chunk_index": "1"},
            relevance_score=0.81,
        ),
        Document(
            id="doc-3",
            content="Prometheus metrics are exposed on the /metrics endpoint.",
            metadata={"source": "readme.md", "chunk_index": "2"},
            relevance_score=0.74,
        ),
    ]


@pytest.fixture()
def sample_query_request() -> QueryRequest:
    """A valid QueryRequest for happy-path tests."""
    return QueryRequest(query="What is Aegis-RAG?", top_k=3)


# ── Settings + app fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def test_settings() -> Settings:
    """
    Settings instance configured for tests.

    Overrides only what matters for deterministic behavior: a known API key,
    console logging (so captured output is human-readable), and a DEBUG flag
    that keeps /docs enabled in case an app fixture needs it.
    """
    return Settings(
        debug=True,
        valid_api_keys=TEST_API_KEY,
        log_format="console",
        log_level="WARNING",
    )


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    """Pre-built header dict with the test API key, ready for TestClient calls."""
    return {TEST_API_KEY_HEADER: TEST_API_KEY}


@pytest.fixture()
def reset_prometheus_registry() -> Iterator[None]:
    """
    Resets the Aegis Prometheus registry before and after a test.

    Use this for tests that assert on metric values — otherwise counters
    from previous tests (same process) contaminate the observation.

    Labeled metrics store per-label-combo trackers in ``_metrics``; unlabeled
    metrics hold their count directly in ``_value``. Reset both paths.
    """
    from aegis.infrastructure.observability import metrics as metrics_module

    def _reset() -> None:
        for collector in list(metrics_module.registry._collector_to_names.keys()):  # noqa: SLF001
            # Labeled metrics: drop every per-label child tracker.
            if hasattr(collector, "_metrics"):
                collector._metrics.clear()  # noqa: SLF001
            # Unlabeled counters/gauges: zero out the backing value.
            value = getattr(collector, "_value", None)
            if value is not None and hasattr(value, "set"):
                value.set(0)

    _reset()
    yield
    _reset()


# ── FastAPI app builders ─────────────────────────────────────────────────────


def build_minimal_app(router: Any, overrides: dict[Any, Any] | None = None) -> TestClient:
    """
    Builds a FastAPI app with a single router and optional dependency overrides.

    Intended for route-layer unit tests that need to isolate a handler from
    the rest of the wiring. For tests that need the full middleware stack,
    use the ``test_app`` / ``test_client`` fixtures instead.
    """
    app = FastAPI()
    app.include_router(router)
    if overrides:
        for dep, impl in overrides.items():
            app.dependency_overrides[dep] = impl
    return TestClient(app)
