"""Tests for the Retriever: dense, hybrid and reranked retrieval through the ports."""

from __future__ import annotations

import pytest

from aegis.application.services.retrieval import Retriever
from aegis.domain.models.document import Document
from aegis.domain.ports.retrieval import LexicalIndexPort, RerankerPort
from tests.conftest import FakeVectorStore


def _doc(doc_id: str, score: float = 0.5) -> Document:
    return Document(id=doc_id, content=f"content of {doc_id}", relevance_score=score)


def _ids(documents: list[Document]) -> list[str]:
    return [d.id for d in documents]


class _RecordingStore(FakeVectorStore):
    def __init__(self, results: list[Document]) -> None:
        super().__init__(next_search_results=results)
        self.asked_k: list[int] = []

    async def similarity_search(
        self, query: str, k: int = 5, score_threshold: float = 0.0
    ) -> list[Document]:
        self.asked_k.append(k)
        return await super().similarity_search(query, k, score_threshold)


class _FixedLexical(LexicalIndexPort):
    def __init__(self, results: list[Document]) -> None:
        self.results = results
        self.asked_k: list[int] = []

    async def search(self, query: str, k: int) -> list[Document]:
        self.asked_k.append(k)
        return self.results[:k]


class _ReversingReranker(RerankerPort):
    """Puts the candidates in reverse order, so its effect is visible."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    async def rerank(self, query: str, documents: list[Document], k: int) -> list[Document]:
        self.seen.append(_ids(documents))
        return [d.model_copy(update={"relevance_score": 0.9}) for d in reversed(documents)][:k]


class TestDense:
    async def test_asks_the_store_for_exactly_k(self) -> None:
        store = _RecordingStore([_doc("a"), _doc("b"), _doc("c")])

        results = await Retriever(store, candidates=50).retrieve("q", k=2)

        assert _ids(results) == ["a", "b"]
        assert store.asked_k == [2]

    async def test_keeps_the_stores_scores(self) -> None:
        store = _RecordingStore([_doc("a", score=0.87)])

        results = await Retriever(store).retrieve("q", k=1)

        assert results[0].relevance_score == 0.87

    def test_mode(self) -> None:
        assert Retriever(FakeVectorStore()).mode == "dense"


class TestHybrid:
    async def test_fuses_dense_and_lexical(self) -> None:
        store = _RecordingStore([_doc("a"), _doc("b")])
        lexical = _FixedLexical([_doc("c"), _doc("b")])

        results = await Retriever(store, lexical_index=lexical).retrieve("q", k=3)

        # b is in both lists, so it comes first; a and c follow.
        assert _ids(results)[0] == "b"
        assert set(_ids(results)) == {"a", "b", "c"}

    async def test_a_lexical_only_hit_is_retrieved(self) -> None:
        """The point of hybrid: a chunk dense retrieval missed can still be found."""
        store = _RecordingStore([_doc("a")])
        lexical = _FixedLexical([_doc("exact-term")])

        results = await Retriever(store, lexical_index=lexical).retrieve("q", k=2)

        assert "exact-term" in _ids(results)

    async def test_both_stages_look_candidates_deep(self) -> None:
        store = _RecordingStore([])
        lexical = _FixedLexical([])

        await Retriever(store, lexical_index=lexical, candidates=20).retrieve("q", k=5)

        assert store.asked_k == [20]
        assert lexical.asked_k == [20]

    async def test_never_looks_shallower_than_k(self) -> None:
        store = _RecordingStore([])
        lexical = _FixedLexical([])

        await Retriever(store, lexical_index=lexical, candidates=3).retrieve("q", k=8)

        assert store.asked_k == [8]
        assert lexical.asked_k == [8]

    async def test_returns_at_most_k(self) -> None:
        store = _RecordingStore([_doc(f"d{i}") for i in range(10)])
        lexical = _FixedLexical([_doc(f"l{i}") for i in range(10)])

        results = await Retriever(store, lexical_index=lexical).retrieve("q", k=4)

        assert len(results) == 4

    async def test_scores_are_in_range(self) -> None:
        store = _RecordingStore([_doc("a"), _doc("b")])
        lexical = _FixedLexical([_doc("a")])

        results = await Retriever(store, lexical_index=lexical).retrieve("q", k=2)

        assert results[0].relevance_score == 1.0
        assert all(0.0 <= d.relevance_score <= 1.0 for d in results)

    def test_mode(self) -> None:
        retriever = Retriever(FakeVectorStore(), lexical_index=_FixedLexical([]))

        assert retriever.mode == "hybrid"


class TestRerank:
    async def test_reranks_the_fused_candidates(self) -> None:
        store = _RecordingStore([_doc("a"), _doc("b")])
        lexical = _FixedLexical([_doc("b"), _doc("c")])
        reranker = _ReversingReranker()

        results = await Retriever(store, lexical_index=lexical, reranker=reranker).retrieve(
            "q", k=2
        )

        assert reranker.seen == [["b", "a", "c"]]
        assert _ids(results) == ["c", "a"]
        assert all(d.relevance_score == 0.9 for d in results)

    async def test_dense_plus_rerank_reads_candidates_deep(self) -> None:
        store = _RecordingStore([_doc(f"d{i}") for i in range(30)])
        reranker = _ReversingReranker()

        results = await Retriever(store, reranker=reranker, candidates=10).retrieve("q", k=3)

        assert store.asked_k == [10]
        assert len(reranker.seen[0]) == 10
        assert _ids(results) == ["d9", "d8", "d7"]

    async def test_nothing_to_rerank_skips_the_reranker(self) -> None:
        reranker = _ReversingReranker()

        results = await Retriever(_RecordingStore([]), reranker=reranker).retrieve("q", k=3)

        assert results == []
        assert reranker.seen == []

    def test_modes(self) -> None:
        store = FakeVectorStore()
        reranker = _ReversingReranker()

        assert Retriever(store, reranker=reranker).mode == "dense+rerank"
        assert (
            Retriever(store, lexical_index=_FixedLexical([]), reranker=reranker).mode
            == "hybrid+rerank"
        )


def test_candidates_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Retriever(FakeVectorStore(), candidates=0)
