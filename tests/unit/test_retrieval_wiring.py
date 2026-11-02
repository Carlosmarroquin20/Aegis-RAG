"""RETRIEVAL_MODE decides which stages the API wires into each tenant's retriever."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from aegis.config import Settings
from aegis.infrastructure.retrieval.cross_encoder import CrossEncoderReranker
from aegis.interface.api import dependencies
from tests.conftest import FakeVectorStore


@pytest.fixture()
def stores(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, FakeVectorStore]]:
    by_collection = {"support-docs": FakeVectorStore(), "billing-docs": FakeVectorStore()}
    monkeypatch.setattr(dependencies, "get_collection_store", lambda name: by_collection[name])
    dependencies.get_lexical_index.cache_clear()
    dependencies.get_reranker.cache_clear()
    yield by_collection
    dependencies.get_lexical_index.cache_clear()
    dependencies.get_reranker.cache_clear()


def _use(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> None:
    settings = Settings(_env_file=None, **overrides)  # type: ignore[call-arg]
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)


@pytest.mark.parametrize("mode", ["dense", "hybrid", "dense+rerank", "hybrid+rerank"])
def test_the_mode_selects_the_stages(
    monkeypatch: pytest.MonkeyPatch, stores: dict[str, FakeVectorStore], mode: str
) -> None:
    _use(monkeypatch, retrieval_mode=mode)

    assert dependencies.build_retriever("support-docs").mode == mode


def test_hybrid_is_the_default(
    monkeypatch: pytest.MonkeyPatch, stores: dict[str, FakeVectorStore]
) -> None:
    _use(monkeypatch)

    assert dependencies.build_retriever("support-docs").mode == "hybrid"


def test_no_reranker_is_built_unless_a_rerank_mode_is_set(
    monkeypatch: pytest.MonkeyPatch, stores: dict[str, FakeVectorStore]
) -> None:
    _use(monkeypatch, retrieval_mode="hybrid")

    assert dependencies.get_reranker() is None


def test_the_reranker_is_built_once_with_the_configured_model(
    monkeypatch: pytest.MonkeyPatch, stores: dict[str, FakeVectorStore]
) -> None:
    _use(monkeypatch, retrieval_mode="hybrid+rerank", reranker_model="org/reranker")

    reranker = dependencies.get_reranker()

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "org/reranker"
    assert dependencies.get_reranker() is reranker


async def test_each_collection_has_its_own_keyword_index(
    monkeypatch: pytest.MonkeyPatch, stores: dict[str, FakeVectorStore]
) -> None:
    from aegis.domain.models.document import Document

    _use(monkeypatch, retrieval_mode="hybrid")
    await stores["billing-docs"].add_documents(
        [Document(id="b1", content="La factura se emite con tu NIT.")]
    )

    support = dependencies.get_lexical_index("support-docs")
    billing = dependencies.get_lexical_index("billing-docs")

    assert support is not billing
    assert support is dependencies.get_lexical_index("support-docs")
    assert await support.search("NIT", k=5) == []
    assert [d.id for d in await billing.search("NIT", k=5)] == ["b1"]
