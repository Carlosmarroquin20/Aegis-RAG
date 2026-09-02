"""
Unit tests for the health/readiness endpoints.

/ready calls the vector-store and LLM adapters directly, so they are
monkeypatched with fakes that report a configurable health state.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegis.interface.api.routes import health


class _FakeAdapter:
    def __init__(self, *, healthy: bool) -> None:
        self._healthy = healthy

    async def health_check(self) -> bool:
        return self._healthy


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(health.router)
    return TestClient(app)


class TestHealth:
    def test_health_is_ok(self) -> None:
        resp = _client().get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestReady:
    def test_ready_when_all_components_healthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health, "get_chromadb_adapter", lambda: _FakeAdapter(healthy=True))
        monkeypatch.setattr(health, "get_ollama_adapter", lambda: _FakeAdapter(healthy=True))
        resp = _client().get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_degraded_when_a_component_is_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health, "get_chromadb_adapter", lambda: _FakeAdapter(healthy=True))
        monkeypatch.setattr(health, "get_ollama_adapter", lambda: _FakeAdapter(healthy=False))
        resp = _client().get("/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["components"]["llm"] is False
