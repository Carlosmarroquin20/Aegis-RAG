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
from aegis.domain.models.document import Document
from aegis.domain.ports.document_parser import ParseError
from aegis.domain.ports.vector_store import VectorStoreUnavailableError
from aegis.interface.api.dependencies import get_ingest_use_case, get_vector_store
from aegis.interface.api.routes import documents
from tests.conftest import FakeVectorStore


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

    def test_oversized_upload_is_rejected_before_reaching_the_use_case(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(documents, "_MAX_UPLOAD_BYTES", 16)
        # No canned response: the fake raises if the route calls it at all.
        client = _client(_FakeIngestUseCase())
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("big.txt", b"x" * 17, "text/plain")},
        )
        assert resp.status_code == 413

    def test_upload_exactly_at_the_limit_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(documents, "_MAX_UPLOAD_BYTES", 16)
        response = IngestResponse(
            source="ok.txt", chunks_created=1, duplicates_skipped=0, collection="c"
        )
        client = _client(_FakeIngestUseCase(response=response))
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("ok.txt", b"x" * 16, "text/plain")},
        )
        assert resp.status_code == 201

    def test_read_is_bounded_when_the_part_size_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a recorded size the route must still not buffer the whole part."""
        from starlette.datastructures import UploadFile

        requested: list[int] = []
        original_init = UploadFile.__init__
        original_read = UploadFile.read

        def init_without_size(self: UploadFile, *args: Any, **kwargs: Any) -> None:
            # UploadFile only tracks its size when it starts from a number.
            kwargs["size"] = None
            original_init(self, *args, **kwargs)

        async def spy_read(self: UploadFile, size: int = -1) -> bytes:
            requested.append(size)
            return await original_read(self, size)

        monkeypatch.setattr(documents, "_MAX_UPLOAD_BYTES", 16)
        monkeypatch.setattr(UploadFile, "__init__", init_without_size)
        monkeypatch.setattr(UploadFile, "read", spy_read)

        client = _client(_FakeIngestUseCase())
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("big.txt", b"x" * 64, "text/plain")},
        )
        assert resp.status_code == 413
        assert requested == [17]

    def test_file_too_large_from_use_case_returns_413(self) -> None:
        client = _client(_FakeIngestUseCase(exc=FileTooLargeError("too big")))
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("big.txt", b"x", "text/plain")},
        )
        assert resp.status_code == 413


# ── GET / DELETE /api/v1/documents ───────────────────────────────────────────


_CHUNK = Document(
    id="chunk-1",
    content="This is the chunk content.",
    metadata={"source": "readme.md", "chunk_index": "0"},
)


class _UninitializedStore(FakeVectorStore):
    async def list_documents(self, limit: int = 20, offset: int = 0) -> list[Document]:
        raise VectorStoreUnavailableError("not initialized")


def _store_client(store: FakeVectorStore) -> TestClient:
    app = FastAPI()
    app.include_router(documents.router)
    app.dependency_overrides[get_vector_store] = lambda: store
    return TestClient(app)


class TestListDocuments:
    def test_lists_indexed_chunks(self) -> None:
        store = FakeVectorStore()
        store.stored.append(_CHUNK)
        resp = _store_client(store).get("/api/v1/documents")
        assert resp.status_code == 200
        assert resp.json() == [
            {
                "id": "chunk-1",
                "source": "readme.md",
                "chunk_index": "0",
                "page": None,
                "section": None,
                "content_preview": "This is the chunk content.",
            }
        ]

    def test_preview_is_truncated(self) -> None:
        store = FakeVectorStore()
        store.stored.append(Document(id="long", content="x" * 500))
        body = _store_client(store).get("/api/v1/documents").json()
        assert len(body[0]["content_preview"]) == 120

    def test_pagination_is_forwarded_to_the_store(self) -> None:
        store = FakeVectorStore()
        store.stored.extend(Document(id=f"c{i}", content="text") for i in range(5))
        body = _store_client(store).get("/api/v1/documents?limit=2&offset=3").json()
        assert [item["id"] for item in body] == ["c3", "c4"]

    def test_uninitialized_store_returns_503(self) -> None:
        resp = _store_client(_UninitializedStore()).get("/api/v1/documents")
        assert resp.status_code == 503


class TestDeleteDocuments:
    def test_delete_single(self) -> None:
        store = FakeVectorStore()
        resp = _store_client(store).delete("/api/v1/documents/chunk-1")
        assert resp.status_code == 200
        assert resp.json() == {"deleted_ids": ["chunk-1"], "count": 1}
        assert store.deleted_ids == ["chunk-1"]

    def test_bulk_delete(self) -> None:
        store = FakeVectorStore()
        resp = _store_client(store).request("DELETE", "/api/v1/documents", json=["a", "b", "c"])
        assert resp.status_code == 200
        assert resp.json()["count"] == 3
        assert store.deleted_ids == ["a", "b", "c"]

    def test_bulk_delete_empty_is_rejected(self) -> None:
        resp = _store_client(FakeVectorStore()).request("DELETE", "/api/v1/documents", json=[])
        assert resp.status_code == 400

    def test_bulk_delete_over_limit_is_rejected(self) -> None:
        resp = _store_client(FakeVectorStore()).request(
            "DELETE", "/api/v1/documents", json=[str(i) for i in range(501)]
        )
        assert resp.status_code == 400
