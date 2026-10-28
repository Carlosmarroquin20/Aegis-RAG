"""
Two tenants on one deployment must not reach each other's documents.

Runs the real application, middleware stack and routes included, with an
in-memory store per collection, and tries every way one tenant could touch the
other's data: listing, searching through a query, deleting by id. Each tenant
reaches only the collection its key resolves to, so none of them can.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from aegis.config import Settings
from aegis.domain.models.document import Document
from aegis.domain.models.generation import Generation
from aegis.infrastructure.security.rate_limiter import RateLimiter, RateLimitPolicy
from aegis.infrastructure.tenancy.registry import TenantRegistry, key_digest
from aegis.interface.api import dependencies, main
from tests.conftest import FakeLLMClient, FakeVectorStore

_SUPPORT_KEY = "support-test-key"
_BILLING_KEY = "billing-test-key"


class _SearchableStore(FakeVectorStore):
    """Search returns what was stored, so a query shows which collection it read."""

    async def similarity_search(
        self, query: str, k: int = 5, score_threshold: float = 0.0
    ) -> list[Document]:
        return self.stored[:k]


class _ContextRecordingLLM(FakeLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[list[str]] = []

    async def generate(self, query: str, context: list[Document], **kwargs: Any) -> Generation:
        self.contexts.append([d.metadata.get("source", "") for d in context])
        return await super().generate(query, context, **kwargs)


@pytest.fixture()
def stores() -> dict[str, _SearchableStore]:
    return {"support-docs": _SearchableStore(), "billing-docs": _SearchableStore()}


@pytest.fixture()
def llm() -> _ContextRecordingLLM:
    return _ContextRecordingLLM()


@pytest.fixture()
def client(
    monkeypatch: pytest.MonkeyPatch,
    stores: dict[str, _SearchableStore],
    llm: _ContextRecordingLLM,
    reset_prometheus_registry: None,
) -> TestClient:
    tenants = TenantRegistry.from_mapping(
        {
            "tenant": [
                {
                    "id": "support",
                    "collection": "support-docs",
                    "api_key_sha256": [key_digest(_SUPPORT_KEY).hex()],
                },
                {
                    "id": "billing",
                    "collection": "billing-docs",
                    "api_key_sha256": [key_digest(_BILLING_KEY).hex()],
                },
            ]
        }
    )
    settings = Settings(log_level="CRITICAL")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "get_tenant_registry", lambda: tenants)
    monkeypatch.setattr(main, "get_auth_failure_limiter", lambda: None)
    monkeypatch.setattr(main, "get_rate_limiter", lambda: RateLimiter(RateLimitPolicy(1000, 60)))
    monkeypatch.setattr(dependencies, "get_collection_store", lambda name: stores[name])
    monkeypatch.setattr(dependencies, "get_llm_client", lambda: llm)
    return TestClient(main.create_app())


def _upload(client: TestClient, key: str, name: str, text: str) -> None:
    response = client.post(
        "/api/v1/documents",
        files={"file": (name, text.encode(), "text/plain")},
        headers={"X-API-Key": key},
    )
    assert response.status_code == 201, response.text


def _list(client: TestClient, key: str) -> list[dict[str, Any]]:
    response = client.get("/api/v1/documents", headers={"X-API-Key": key})
    assert response.status_code == 200
    return response.json()  # type: ignore[no-any-return]


class TestTenantIsolation:
    def test_an_upload_lands_in_the_uploaders_collection_only(
        self, client: TestClient, stores: dict[str, _SearchableStore]
    ) -> None:
        _upload(client, _SUPPORT_KEY, "returns.txt", "Returns are accepted within 30 days.")

        assert stores["support-docs"].stored
        assert stores["billing-docs"].stored == []

    def test_a_tenant_lists_only_its_own_documents(self, client: TestClient) -> None:
        _upload(client, _SUPPORT_KEY, "returns.txt", "Returns are accepted within 30 days.")
        _upload(client, _BILLING_KEY, "invoices.txt", "Invoices are issued with your NIT.")

        assert {d["source"] for d in _list(client, _SUPPORT_KEY)} == {"returns.txt"}
        assert {d["source"] for d in _list(client, _BILLING_KEY)} == {"invoices.txt"}

    def test_a_query_is_answered_from_the_callers_documents_only(
        self, client: TestClient, llm: _ContextRecordingLLM
    ) -> None:
        _upload(client, _SUPPORT_KEY, "returns.txt", "Returns are accepted within 30 days.")
        _upload(client, _BILLING_KEY, "invoices.txt", "Invoices are issued with your NIT.")

        response = client.post(
            "/api/v1/query",
            json={"query": "How are invoices issued?"},
            headers={"X-API-Key": _SUPPORT_KEY},
        )

        assert response.status_code == 200
        assert llm.contexts == [["returns.txt"]]

    def test_a_tenant_cannot_delete_another_tenants_document(
        self, client: TestClient, stores: dict[str, _SearchableStore]
    ) -> None:
        """Knowing the id is not enough: the delete runs against the caller's collection."""
        _upload(client, _BILLING_KEY, "invoices.txt", "Invoices are issued with your NIT.")
        billing_id = _list(client, _BILLING_KEY)[0]["id"]

        response = client.delete(
            f"/api/v1/documents/{billing_id}", headers={"X-API-Key": _SUPPORT_KEY}
        )

        assert response.status_code == 200  # deleting an unknown id is a no-op
        assert [d["id"] for d in _list(client, _BILLING_KEY)] == [billing_id]
        assert stores["support-docs"].deleted_ids == [billing_id]
        assert stores["billing-docs"].deleted_ids == []


class TestNoTenantNoStore:
    def test_a_request_without_a_tenant_gets_no_store_at_all(self) -> None:
        """No default collection to fall back to: that would serve someone's documents."""
        from starlette.requests import Request

        request = Request({"type": "http", "state": {}})
        with pytest.raises(LookupError, match="no tenant"):
            dependencies.get_vector_store(request)
