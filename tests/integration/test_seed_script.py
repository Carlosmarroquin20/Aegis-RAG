"""
Integration tests for scripts/seed_documents.py.

The script sits outside the package and is not exercised by the API tests, which
is how it stayed broken for a while without anything noticing. These run its
``seed`` coroutine against the real parsers and use case with an in-memory
vector store, and also import the module the way ``python scripts/...`` does.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from aegis.application.use_cases.ingest_documents import IngestDocumentsUseCase
from aegis.infrastructure.parsers.parser_registry import ParserRegistry
from tests.conftest import FakeVectorStore

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "seed_documents.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("seed_documents", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: dataclasses resolve their own module through
    # sys.modules while the class body is being processed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture()
def use_case(store: FakeVectorStore) -> IngestDocumentsUseCase:
    return IngestDocumentsUseCase(
        vector_store=store, parser_registry=ParserRegistry(), collection="seed-test"
    )


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "remote.md").write_text(
        "# Remote work\n\nEmployees may work remotely two days per week with approval.\n",
        encoding="utf-8",
    )
    (tmp_path / "handbook.txt").write_text(
        "The onboarding handbook explains accounts, equipment and security training.\n",
        encoding="utf-8",
    )
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    return tmp_path


class TestSeed:
    async def test_indexes_supported_files_recursively(
        self,
        script: ModuleType,
        use_case: IngestDocumentsUseCase,
        store: FakeVectorStore,
        corpus: Path,
    ) -> None:
        summary = await script.seed(corpus, use_case)

        assert summary.files_indexed == 2
        assert summary.chunks_created == len(store.stored) > 0
        assert summary.failures == []

    async def test_source_is_the_path_relative_to_the_seed_directory(
        self,
        script: ModuleType,
        use_case: IngestDocumentsUseCase,
        store: FakeVectorStore,
        corpus: Path,
    ) -> None:
        await script.seed(corpus, use_case)
        assert {doc.metadata["source"] for doc in store.stored} == {
            "handbook.txt",
            "policies/remote.md",
        }

    async def test_unsupported_extensions_are_not_attempted(
        self,
        script: ModuleType,
        use_case: IngestDocumentsUseCase,
        store: FakeVectorStore,
        corpus: Path,
    ) -> None:
        await script.seed(corpus, use_case)
        assert all(not doc.metadata["source"].endswith(".png") for doc in store.stored)

    async def test_seeding_twice_adds_nothing(
        self,
        script: ModuleType,
        use_case: IngestDocumentsUseCase,
        store: FakeVectorStore,
        corpus: Path,
    ) -> None:
        first = await script.seed(corpus, use_case)
        stored = len(store.stored)
        second = await script.seed(corpus, use_case)

        assert second.chunks_created == 0
        assert second.duplicates_skipped == first.chunks_created
        assert len(store.stored) == stored

    async def test_unreadable_file_is_reported_and_the_rest_still_load(
        self, script: ModuleType, use_case: IngestDocumentsUseCase, corpus: Path
    ) -> None:
        (corpus / "broken.pdf").write_bytes(b"%PDF-1.7 this is not a real pdf")

        summary = await script.seed(corpus, use_case)

        assert summary.files_indexed == 2
        assert [source for source, _ in summary.failures] == ["broken.pdf"]

    async def test_empty_directory_yields_an_empty_summary(
        self, script: ModuleType, use_case: IngestDocumentsUseCase, tmp_path: Path
    ) -> None:
        summary = await script.seed(tmp_path, use_case)
        assert (summary.files_indexed, summary.chunks_created, summary.failures) == (0, 0, [])
