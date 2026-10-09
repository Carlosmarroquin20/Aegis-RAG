"""
Unit tests for ChromaDBAdapter.

The vendor client is replaced by small recording fakes assigned to the adapter's
handles, so neither ``chromadb`` nor a running server is needed. What is under
test is the adapter's own behaviour: response mapping, the not-initialized
guard, and — because the client is synchronous — that no call runs on the event
loop thread.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from aegis.domain.models.document import Document
from aegis.domain.ports.vector_store import VectorStoreUnavailableError
from aegis.infrastructure.vector_stores.chromadb_adapter import ChromaDBAdapter


class _FakeCollection:
    """Records the calling thread and arguments of every operation."""

    def __init__(self) -> None:
        self.threads: list[int] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.query_result: dict[str, Any] = {
            "ids": [["a", "b"]],
            "documents": [["alpha", "beta"]],
            "metadatas": [[{"source": "x.txt", "page": 3}, None]],
            "distances": [[0.2, 1.6]],
        }
        self.get_result: dict[str, Any] = {"ids": [], "documents": [], "metadatas": []}

    def _record(self, name: str, kwargs: dict[str, Any]) -> None:
        self.threads.append(threading.get_ident())
        self.calls.append((name, kwargs))

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self._record("query", kwargs)
        return self.query_result

    def upsert(self, **kwargs: Any) -> None:
        self._record("upsert", kwargs)

    def delete(self, **kwargs: Any) -> None:
        self._record("delete", kwargs)

    def get(self, **kwargs: Any) -> dict[str, Any]:
        self._record("get", kwargs)
        return self.get_result


class _FakeClient:
    def __init__(self, *, alive: bool = True) -> None:
        self._alive = alive
        self.threads: list[int] = []

    def heartbeat(self) -> int:
        self.threads.append(threading.get_ident())
        if not self._alive:
            raise ConnectionError("chroma unreachable")
        return 1


def _adapter(
    collection: _FakeCollection | None = None, client: _FakeClient | None = None
) -> ChromaDBAdapter:
    adapter = ChromaDBAdapter(
        host="chroma", port=8000, collection_name="docs", embedding_function=object()
    )
    adapter._collection = collection
    adapter._client = client
    return adapter


class TestSimilaritySearch:
    async def test_maps_results_and_converts_distance_to_similarity(self) -> None:
        results = await _adapter(_FakeCollection()).similarity_search("q", k=2)

        assert results == [
            Document(
                id="a",
                content="alpha",
                metadata={"source": "x.txt", "page": "3"},
                relevance_score=0.9,
            ),
            Document(id="b", content="beta", metadata={}, relevance_score=0.2),
        ]

    async def test_applies_score_threshold(self) -> None:
        results = await _adapter(_FakeCollection()).similarity_search("q", k=2, score_threshold=0.5)
        assert [doc.id for doc in results] == ["a"]

    async def test_tolerates_null_metadata(self) -> None:
        """Chroma returns None, not {}, for records stored without metadata."""
        results = await _adapter(_FakeCollection()).similarity_search("q")
        assert results[1].metadata == {}

    async def test_forwards_query_arguments(self) -> None:
        collection = _FakeCollection()
        await _adapter(collection).similarity_search("what is x?", k=7)
        name, kwargs = collection.calls[0]
        assert name == "query"
        assert kwargs["query_texts"] == ["what is x?"]
        assert kwargs["n_results"] == 7


class TestWritesAndLookups:
    async def test_add_documents_upserts_parallel_lists(self) -> None:
        collection = _FakeCollection()
        docs = [
            Document(id="1", content="one", metadata={"source": "a"}),
            Document(id="2", content="two", metadata={"source": "b"}),
        ]
        await _adapter(collection).add_documents(docs)

        assert collection.calls == [
            (
                "upsert",
                {
                    "ids": ["1", "2"],
                    "documents": ["one", "two"],
                    "metadatas": [{"source": "a"}, {"source": "b"}],
                },
            )
        ]

    async def test_delete_documents_forwards_ids(self) -> None:
        collection = _FakeCollection()
        await _adapter(collection).delete_documents(["1", "2"])
        assert collection.calls == [("delete", {"ids": ["1", "2"]})]

    async def test_existing_ids_returns_only_ids_present_in_the_store(self) -> None:
        collection = _FakeCollection()
        collection.get_result = {"ids": ["1"]}
        found = await _adapter(collection).existing_ids(["1", "2"])

        assert found == {"1"}
        assert collection.calls == [("get", {"ids": ["1", "2"], "include": []})]

    async def test_list_documents_maps_page_and_skips_empty_records(self) -> None:
        collection = _FakeCollection()
        collection.get_result = {
            "ids": ["1", "2", "3"],
            "documents": ["one", None, "three"],
            "metadatas": [{"source": "a"}, None, None],
        }
        page = await _adapter(collection).list_documents(limit=3, offset=6)

        assert [(d.id, d.content, d.metadata) for d in page] == [
            ("1", "one", {"source": "a"}),
            ("3", "three", {}),
        ]
        assert collection.calls[0][1]["limit"] == 3
        assert collection.calls[0][1]["offset"] == 6


class TestNotInitialized:
    @pytest.mark.parametrize(
        "operation",
        [
            lambda a: a.similarity_search("q"),
            lambda a: a.add_documents([Document(id="1", content="x")]),
            lambda a: a.delete_documents(["1"]),
            lambda a: a.existing_ids(["1"]),
            lambda a: a.list_documents(),
        ],
        ids=["search", "add", "delete", "existing_ids", "list"],
    )
    async def test_operations_raise_the_domain_error(self, operation: Any) -> None:
        with pytest.raises(VectorStoreUnavailableError):
            await operation(_adapter())

    async def test_health_check_is_false_rather_than_raising(self) -> None:
        assert await _adapter().health_check() is False


class TestHealthCheck:
    async def test_true_when_the_server_answers(self) -> None:
        assert await _adapter(client=_FakeClient()).health_check() is True

    async def test_false_when_the_heartbeat_fails(self) -> None:
        assert await _adapter(client=_FakeClient(alive=False)).health_check() is False

    async def test_reuses_the_startup_client_across_probes(self) -> None:
        client = _FakeClient()
        adapter = _adapter(client=client)
        for _ in range(3):
            await adapter.health_check()
        assert len(client.threads) == 3


class TestBlockingCallsLeaveTheEventLoop:
    """
    The client is synchronous and a query also runs the embedding model. If any
    of these ran on the loop thread, one slow request would freeze all others.
    """

    async def test_collection_operations_run_on_a_worker_thread(self) -> None:
        collection = _FakeCollection()
        adapter = _adapter(collection)

        await adapter.similarity_search("q")
        await adapter.add_documents([Document(id="1", content="x")])
        await adapter.delete_documents(["1"])
        await adapter.existing_ids(["1"])
        await adapter.list_documents()

        assert len(collection.threads) == 5
        assert threading.get_ident() not in collection.threads

    async def test_health_probe_runs_on_a_worker_thread(self) -> None:
        client = _FakeClient()
        await _adapter(client=client).health_check()
        assert client.threads != [threading.get_ident()]


class TestClientFailuresAreTranslated:
    """Whatever the vendor client raises, callers only ever see the port's error."""

    @pytest.mark.parametrize(
        "error",
        [ConnectionError("refused"), TimeoutError("slow"), ValueError("bad response")],
        ids=["connection", "timeout", "value"],
    )
    async def test_failed_operation_raises_the_domain_error(self, error: Exception) -> None:
        class Failing(_FakeCollection):
            def query(self, **kwargs: Any) -> dict[str, Any]:
                raise error

        with pytest.raises(VectorStoreUnavailableError) as excinfo:
            await _adapter(Failing()).similarity_search("q")
        # asyncio re-creates TimeoutError when relaying it from the worker thread,
        # so compare by type rather than identity.
        assert type(excinfo.value.__cause__) is type(error)
