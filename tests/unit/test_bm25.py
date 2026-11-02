"""Tests for the BM25 keyword index and its store-backed wrapper."""

from __future__ import annotations

import asyncio

import pytest

from aegis.domain.models.document import Document
from aegis.infrastructure.retrieval.bm25 import BM25Index, StoreBackedLexicalIndex, tokenize
from tests.conftest import FakeVectorStore


def _doc(doc_id: str, content: str) -> Document:
    return Document(id=doc_id, content=content, metadata={"source": f"{doc_id}.md"})


_CORPUS = [
    _doc("returns", "Las devoluciones se aceptan dentro de 30 días con el ticket de compra."),
    _doc("shipping", "Los envíos a domicilio tardan de 2 a 5 días hábiles."),
    _doc("loyalty", "Cada punto del programa de lealtad vale un quetzal en tu próxima compra."),
    _doc("warranty", "La garantía de los colchones cubre defectos de fabricación por 10 años."),
]


class TestTokenize:
    def test_folds_case_and_accents(self) -> None:
        assert tokenize("Garantía DÍAS") == tokenize("garantia dias")

    def test_removes_stop_words(self) -> None:
        assert tokenize("el colchón de la tienda") == ["colchon", "tienda"]

    @pytest.mark.parametrize(
        ("plural", "singular"),
        [
            ("devoluciones", "devolución"),
            ("colchones", "colchón"),
            ("días", "día"),
            ("luces", "luz"),
            ("mujeres", "mujer"),
            ("ciudades", "ciudad"),
            ("clientes", "cliente"),
            ("returns", "return"),
        ],
    )
    def test_plural_and_singular_share_a_term(self, plural: str, singular: str) -> None:
        assert tokenize(plural) == tokenize(singular)

    def test_short_words_and_double_s_are_left_alone(self) -> None:
        assert tokenize("mas class") == ["mas", "class"]

    def test_keeps_numbers(self) -> None:
        assert "30" in tokenize("30 días")


class TestBM25Index:
    def test_finds_the_document_sharing_the_rare_term(self) -> None:
        index = BM25Index(_CORPUS)

        results = index.search("¿Qué cubre la garantía?", k=3)

        assert results[0].id == "warranty"

    def test_matches_across_plural_and_accent(self) -> None:
        index = BM25Index(_CORPUS)

        assert index.search("devolucion colchon", k=1)[0].id in {"returns", "warranty"}
        assert index.search("devolución", k=1)[0].id == "returns"

    def test_documents_without_a_shared_term_are_not_returned(self) -> None:
        index = BM25Index(_CORPUS)

        assert [d.id for d in index.search("garantía", k=10)] == ["warranty"]
        assert index.search("bitcoin", k=10) == []

    def test_scores_are_relative_to_the_best_match(self) -> None:
        index = BM25Index(_CORPUS)

        results = index.search("compra días", k=4)

        assert results[0].relevance_score == 1.0
        assert all(0.0 < d.relevance_score <= 1.0 for d in results)

    def test_a_rare_term_outweighs_a_common_one(self) -> None:
        index = BM25Index(_CORPUS)

        # "días" is in two documents, "ticket" in one.
        assert index.search("días ticket", k=1)[0].id == "returns"

    def test_repeating_a_query_term_does_not_change_the_ranking(self) -> None:
        index = BM25Index(_CORPUS)

        once = [d.id for d in index.search("compra lealtad", k=4)]
        repeated = [d.id for d in index.search("compra compra compra lealtad", k=4)]

        assert once == repeated

    def test_k_limits_the_results(self) -> None:
        assert len(BM25Index(_CORPUS).search("días compra", k=1)) == 1

    def test_empty_index_and_nonpositive_k(self) -> None:
        assert BM25Index([]).search("anything", k=5) == []
        assert BM25Index(_CORPUS).search("garantía", k=0) == []

    def test_metadata_is_preserved(self) -> None:
        result = BM25Index(_CORPUS).search("garantía", k=1)[0]

        assert result.metadata == {"source": "warranty.md"}


class _CountingStore(FakeVectorStore):
    def __init__(self, documents: list[Document]) -> None:
        super().__init__()
        self.stored = list(documents)
        self.list_calls = 0

    async def list_documents(self, limit: int = 20, offset: int = 0) -> list[Document]:
        self.list_calls += 1
        return await super().list_documents(limit, offset)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestStoreBackedLexicalIndex:
    async def test_searches_the_stores_chunks(self) -> None:
        index = StoreBackedLexicalIndex(_CountingStore(_CORPUS))

        assert (await index.search("garantía", k=3))[0].id == "warranty"

    async def test_reuses_the_index_while_the_store_is_unchanged(self) -> None:
        store = _CountingStore(_CORPUS)
        index = StoreBackedLexicalIndex(store, page_size=100)

        await index.search("garantía", k=3)
        await index.search("envíos", k=3)

        assert store.list_calls == 1

    async def test_an_added_chunk_is_found_on_the_next_query(self) -> None:
        store = _CountingStore(_CORPUS)
        index = StoreBackedLexicalIndex(store)
        assert await index.search("factura", k=3) == []

        await store.add_documents([_doc("billing", "La factura se emite con tu NIT.")])

        assert (await index.search("factura", k=3))[0].id == "billing"

    async def test_a_deleted_chunk_disappears_on_the_next_query(self) -> None:
        store = _CountingStore(_CORPUS)
        index = StoreBackedLexicalIndex(store)
        assert await index.search("garantía", k=3)

        await store.delete_documents(["warranty"])

        assert await index.search("garantía", k=3) == []

    async def test_a_same_size_change_is_picked_up_after_max_age(self) -> None:
        store = _CountingStore(_CORPUS)
        clock = _Clock()
        index = StoreBackedLexicalIndex(store, max_age_seconds=60, clock=clock)
        await index.search("garantía", k=3)

        # One chunk replaced by another: the count does not move.
        await store.delete_documents(["warranty"])
        await store.add_documents([_doc("billing", "La factura se emite con tu NIT.")])
        clock.now += 30
        assert await index.search("factura", k=3) == []  # still within max age

        clock.now += 31
        assert (await index.search("factura", k=3))[0].id == "billing"

    async def test_loads_every_page_by_the_stores_count(self) -> None:
        documents = [_doc(f"d{i}", f"documento número {i} palabra{i}") for i in range(7)]
        store = _CountingStore(documents)
        index = StoreBackedLexicalIndex(store, page_size=3)

        results = await index.search("palabra6", k=1)

        assert results[0].id == "d6"
        assert store.list_calls == 3

    async def test_a_short_page_does_not_end_loading_early(self) -> None:
        """A page can come back short when records without text are skipped."""

        class _ShortPages(_CountingStore):
            async def list_documents(self, limit: int = 20, offset: int = 0) -> list[Document]:
                page = await super().list_documents(limit, offset)
                return page[:-1] if offset == 0 else page

        documents = [_doc(f"d{i}", f"palabra{i}") for i in range(4)]
        index = StoreBackedLexicalIndex(_ShortPages(documents), page_size=2)

        assert (await index.search("palabra3", k=1))[0].id == "d3"

    async def test_concurrent_queries_rebuild_once(self) -> None:
        store = _CountingStore(_CORPUS)
        index = StoreBackedLexicalIndex(store, page_size=100)

        await asyncio.gather(*(index.search("garantía", k=3) for _ in range(10)))

        assert store.list_calls == 1

    async def test_an_empty_store_answers_nothing(self) -> None:
        index = StoreBackedLexicalIndex(_CountingStore([]))

        assert await index.search("garantía", k=3) == []

    def test_page_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            StoreBackedLexicalIndex(_CountingStore([]), page_size=0)
