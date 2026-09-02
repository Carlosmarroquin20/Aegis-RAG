"""
Unit tests for the document management routes.

The ingest handler is isolated via a dependency override; the list/delete
handlers call get_chromadb_adapter() directly, so those are monkeypatched.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegis.application.dtos.ingestion_dtos import IngestResponse
from aegis.application.use_cases.ingest_documents import (
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from aegis.domain.ports.document_parser import ParseError
from aegis.interface.api.dependencies import get_ingest_use_case
from aegis.interface.api.routes import documents


class _FakeIngestUseCase:
    def __init__(self, *, response: IngestResponse | None = None, exc: Exception | None = None):
        self._response = response
        self._exc = exc

    async def execute(self, uploaded: Any, request: Any) -> IngestResponse:
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response


def _client(use_case: _FakeIngestUseCase) -> TestClient:
    app = FastAPI()
    app.include_router(documents.router)
    app.dependency_overrides[get_ingest_use_case] = lambda: use_case
    return TestClient(app)


# ── POST /api/v1/documents ───────────────────────────────────────────────────


class TestIngestDocument:
    def test_successful_ingest_returns_201(self) -> None:
        response = IngestResponse(
            source="notes.txt",
            chunks_created=3,
            duplicates_skipped=0,
            warnings=[],
            collection="aegis_documents",
        )
        client = _client(_FakeIngestUseCase(response=response))
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 201
        assert resp.json()["chunks_created"] == 3

    def test_unsupported_type_returns_415(self) -> None:
        client = _client(_FakeIngestUseCase(exc=UnsupportedFileTypeError("no parser")))
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("weird.xyz", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 415

    def test_parse_error_returns_400(self) -> None:
        client = _client(_FakeIngestUseCase(exc=ParseError("bad.pdf", "corrupt")))
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("bad.pdf", b"%PDF-broken", "application/pdf")},
        )
        assert resp.status_code == 400

    def test_file_too_large_from_use_case_returns_413(self) -> None:
        client = _client(_FakeIngestUseCase(exc=FileTooLargeError("too big")))
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("big.txt", b"x", "text/plain")},
        )
        assert resp.status_code == 413


# ── GET / DELETE /api/v1/documents ───────────────────────────────────────────


class _FakeCollection:
    def get(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "ids": ["chunk-1"],
            "documents": ["This is the chunk content."],
            "metadatas": [{"source": "readme.md", "chunk_index": "0"}],
        }


class _FakeAdapter:
    def __init__(self, *, initialized: bool = True) -> None:
        self._collection = _FakeCollection() if initialized else None
        self.deleted: list[str] = []

    async def delete_documents(self, ids: list[str]) -> None:
        self.deleted.extend(ids)


def _routed_client() -> TestClient:
    app = FastAPI()
    app.include_router(documents.router)
    return TestClient(app)


class TestListDocuments:
    def test_lists_indexed_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(documents, "get_chromadb_adapter", lambda: _FakeAdapter())
        resp = _routed_client().get("/api/v1/documents")
        assert resp.status_code == 200
        body = resp.json()
        assert body[0]["id"] == "chunk-1"
        assert body[0]["source"] == "readme.md"

    def test_uninitialized_store_returns_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            documents, "get_chromadb_adapter", lambda: _FakeAdapter(initialized=False)
        )
        resp = _routed_client().get("/api/v1/documents")
        assert resp.status_code == 503


class TestDeleteDocuments:
    def test_delete_single(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _FakeAdapter()
        monkeypatch.setattr(documents, "get_chromadb_adapter", lambda: adapter)
        resp = _routed_client().delete("/api/v1/documents/chunk-1")
        assert resp.status_code == 200
        assert resp.json() == {"deleted_ids": ["chunk-1"], "count": 1}
        assert adapter.deleted == ["chunk-1"]

    def test_bulk_delete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _FakeAdapter()
        monkeypatch.setattr(documents, "get_chromadb_adapter", lambda: adapter)
        resp = _routed_client().request("DELETE", "/api/v1/documents", json=["a", "b", "c"])
        assert resp.status_code == 200
        assert resp.json()["count"] == 3

    def test_bulk_delete_empty_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(documents, "get_chromadb_adapter", lambda: _FakeAdapter())
        resp = _routed_client().request("DELETE", "/api/v1/documents", json=[])
        assert resp.status_code == 400

    def test_bulk_delete_over_limit_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(documents, "get_chromadb_adapter", lambda: _FakeAdapter())
        resp = _routed_client().request(
            "DELETE", "/api/v1/documents", json=[str(i) for i in range(501)]
        )
        assert resp.status_code == 400
