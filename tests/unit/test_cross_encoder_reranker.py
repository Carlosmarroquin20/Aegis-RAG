"""
Tests for the cross-encoder reranker adapter.

The model itself is replaced by a stub: these tests pin what the adapter does
with its scores (order, cut, range) and how it loads, not what a given model
thinks. How much the real model improves retrieval is measured by
scripts/evaluate_retrieval.py in CI.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Sequence
from typing import Any

import pytest

from aegis.domain.models.document import Document
from aegis.infrastructure.retrieval import cross_encoder
from aegis.infrastructure.retrieval.cross_encoder import CrossEncoderReranker


def _doc(doc_id: str) -> Document:
    return Document(id=doc_id, content=f"content of {doc_id}", relevance_score=0.3)


class _StubModel:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], **_: Any) -> Sequence[float]:
        self.calls.append(pairs)
        return [self.scores[text.removeprefix("content of ")] for _, text in pairs]


@pytest.fixture()
def builds(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records every model build; the built model is set per test via ``_install``."""
    calls: list[str] = []
    monkeypatch.setattr(CrossEncoderReranker, "_build", lambda self: calls.append("built"))
    return calls


def _install(reranker: CrossEncoderReranker, model: _StubModel) -> None:
    reranker._model = model


class TestRerank:
    async def test_orders_by_the_models_score_and_cuts_to_k(self) -> None:
        reranker = CrossEncoderReranker()
        _install(reranker, _StubModel({"a": 0.1, "b": 0.9, "c": 0.5}))

        results = await reranker.rerank("q", [_doc("a"), _doc("b"), _doc("c")], k=2)

        assert [d.id for d in results] == ["b", "c"]
        assert [d.relevance_score for d in results] == [0.9, 0.5]

    async def test_reads_the_query_with_each_document(self) -> None:
        reranker = CrossEncoderReranker()
        model = _StubModel({"a": 0.2})
        _install(reranker, model)

        await reranker.rerank("¿cuánto dura la garantía?", [_doc("a")], k=1)

        assert model.calls == [[("¿cuánto dura la garantía?", "content of a")]]

    async def test_raw_logits_are_squashed_into_range_without_reordering(self) -> None:
        reranker = CrossEncoderReranker()
        _install(reranker, _StubModel({"a": -7.5, "b": 3.2, "c": 0.0}))

        results = await reranker.rerank("q", [_doc("a"), _doc("b"), _doc("c")], k=3)

        assert [d.id for d in results] == ["b", "c", "a"]
        assert all(0.0 <= d.relevance_score <= 1.0 for d in results)
        assert results[1].relevance_score == 0.5

    async def test_extreme_logits_do_not_overflow(self) -> None:
        reranker = CrossEncoderReranker()
        _install(reranker, _StubModel({"a": -1000.0, "b": 1000.0}))

        results = await reranker.rerank("q", [_doc("a"), _doc("b")], k=2)

        assert [d.relevance_score for d in results] == [1.0, 0.0]

    async def test_equal_scores_keep_the_first_stage_order(self) -> None:
        reranker = CrossEncoderReranker()
        _install(reranker, _StubModel({"a": 0.5, "b": 0.5, "c": 0.5}))

        results = await reranker.rerank("q", [_doc("c"), _doc("a"), _doc("b")], k=3)

        assert [d.id for d in results] == ["c", "a", "b"]

    async def test_nothing_to_rerank_does_not_load_the_model(self, builds: list[str]) -> None:
        reranker = CrossEncoderReranker()

        assert await reranker.rerank("q", [], k=3) == []
        assert await reranker.rerank("q", [_doc("a")], k=0) == []
        assert builds == []


class TestLoading:
    async def test_loads_once_however_many_callers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        builds: list[str] = []

        def build(self: CrossEncoderReranker) -> _StubModel:
            builds.append(self.model_name)
            return _StubModel({"a": 0.4})

        monkeypatch.setattr(CrossEncoderReranker, "_build", build)
        reranker = CrossEncoderReranker("some/model")

        await asyncio.gather(reranker.load(), reranker.load(), reranker.rerank("q", [_doc("a")], 1))

        assert builds == ["some/model"]

    def test_never_trusts_remote_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A reranker that must run code shipped with its weights is not loaded."""
        seen: dict[str, Any] = {}

        class _FakeCrossEncoder:
            def __init__(self, name: str, **kwargs: Any) -> None:
                seen.update(kwargs, name=name)

        # A stand-in module: the real one imports torch, which this test does not need.
        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            types.SimpleNamespace(CrossEncoder=_FakeCrossEncoder),
        )

        CrossEncoderReranker("some/model", device="cpu", max_length=256)._build()

        assert seen == {
            "name": "some/model",
            "device": "cpu",
            "max_length": 256,
            "trust_remote_code": False,
        }

    def test_the_default_is_a_multilingual_model(self) -> None:
        assert "mmarco" in cross_encoder.DEFAULT_RERANKER_MODEL
        assert CrossEncoderReranker().model_name == cross_encoder.DEFAULT_RERANKER_MODEL
