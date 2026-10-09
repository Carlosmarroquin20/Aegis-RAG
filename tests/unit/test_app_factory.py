"""
Unit tests for ``create_app()`` — behaviour that only exists once the whole
application is assembled (settings-driven switches, cross-middleware effects).

The lifespan is never entered (``TestClient`` is used without a context manager),
so no vector store or LLM is contacted.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aegis.config import Settings
from aegis.interface.api import main

_KEY = "unit-test-key"


def _client(monkeypatch: pytest.MonkeyPatch, *, debug: bool) -> TestClient:
    settings = Settings(debug=debug, valid_api_keys=_KEY, log_level="CRITICAL")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    return TestClient(main.create_app(), raise_server_exceptions=False)


class TestApiDocsExposure:
    @pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc"])
    def test_schema_and_docs_are_not_served_in_production(
        self, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        """These paths skip authentication, so in production they must not exist."""
        client = _client(monkeypatch, debug=False)
        assert client.get(path).status_code == 404

    @pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc"])
    def test_schema_and_docs_are_served_in_debug(
        self, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        client = _client(monkeypatch, debug=True)
        assert client.get(path).status_code == 200
