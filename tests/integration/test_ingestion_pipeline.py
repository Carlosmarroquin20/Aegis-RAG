"""
Integration tests for the full ingestion pipeline.

Tests the IngestDocumentsUseCase end-to-end with a mock vector store,
verifying that parsing → chunking → deduplication → upsert work correctly
together without requiring a live ChromaDB or filesystem.
"""

from __future__ import annotations

import pytest

from aegis.application.dtos.ingestion_dtos import IngestRequest, UploadedFile
from aegis.application.use_cases.ingest_documents import (
    FileTooLargeError,
    IngestDocumentsUseCase,
    UnsupportedFileTypeError,
)
from aegis.infrastructure.parsers.parser_registry import ParserRegistry
from tests.conftest import FakeVectorStore

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture()
def registry() -> ParserRegistry:
    return ParserRegistry()


@pytest.fixture()
def use_case(vector_store: FakeVectorStore, registry: ParserRegistry) -> IngestDocumentsUseCase:
    return IngestDocumentsUseCase(
        vector_store=vector_store,
        parser_registry=registry,
        default_collection="test_collection",
    )


@pytest.fixture()
def default_request() -> IngestRequest:
    return IngestRequest(chunk_size=256, overlap=32)


# ── Plain-Text Ingestion ───────────────────────────────────────────────────────


class TestTxtIngestion:
    async def test_txt_file_ingested_successfully(
        self,
        use_case: IngestDocumentsUseCase,
        vector_store: FakeVectorStore,
        default_request: IngestRequest,
    ) -> None:
        content = "This is a test document. " * 20  # > one chunk
        uploaded = UploadedFile(filename="test.txt", content=content.encode())

        result = await use_case.execute(uploaded, default_request)

        assert result.chunks_created > 0
        assert result.source == "test.txt"
        assert result.collection == "test_collection"
        assert len(vector_store.stored) == result.chunks_created

    async def test_chunks_have_correct_source_metadata(
        self,
        use_case: IngestDocumentsUseCase,
        vector_store: FakeVectorStore,
        default_request: IngestRequest,
    ) -> None:
        content = "This is sentence one. This is sentence two. " * 10
        uploaded = UploadedFile(filename="policy.txt", content=content.encode())

        await use_case.execute(uploaded, default_request)

        assert all(d.metadata["source"] == "policy.txt" for d in vector_store.stored)

    async def test_empty_txt_raises_parse_error(
        self,
        use_case: IngestDocumentsUseCase,
        default_request: IngestRequest,
    ) -> None:
        from aegis.domain.ports.document_parser import ParseError

        uploaded = UploadedFile(filename="empty.txt", content=b"   \n  ")
        with pytest.raises(ParseError):
            await use_case.execute(uploaded, default_request)


# ── Markdown Ingestion ─────────────────────────────────────────────────────────


class TestMarkdownIngestion:
    async def test_markdown_file_ingested(
        self,
        use_case: IngestDocumentsUseCase,
        vector_store: FakeVectorStore,
        default_request: IngestRequest,
    ) -> None:
        md_content = (
            "# Title\n\nThis is a paragraph with **bold** text.\n\n## Section\n\nMore content here."
        )
        uploaded = UploadedFile(filename="readme.md", content=md_content.encode())

        result = await use_case.execute(uploaded, default_request)

        assert result.chunks_created >= 1
        # Markdown tags should not appear in the indexed text.
        for doc in vector_store.stored:
            assert "<h1>" not in doc.content
            assert "**" not in doc.content


# ── Security: File Size ────────────────────────────────────────────────────────


class TestFileSizeEnforcement:
    async def test_oversized_file_rejected(
        self,
        use_case: IngestDocumentsUseCase,
        default_request: IngestRequest,
    ) -> None:
        # Simulate a file slightly over the 50 MB limit.
        oversized = b"x" * (50 * 1024 * 1024 + 1)
        uploaded = UploadedFile(filename="huge.txt", content=oversized)

        with pytest.raises(FileTooLargeError):
            await use_case.execute(uploaded, default_request)


# ── Security: Unsupported File Types ──────────────────────────────────────────


class TestUnsupportedFileTypes:
    async def test_executable_file_rejected(
        self,
        use_case: IngestDocumentsUseCase,
        default_request: IngestRequest,
    ) -> None:
        # PE header magic bytes (\x4d\x5a = MZ) — Windows executable.
        exe_content = b"\x4d\x5a" + b"\x00" * 100
        uploaded = UploadedFile(
            filename="malware.exe",
            content=exe_content,
            declared_mime_type="application/octet-stream",
        )

        with pytest.raises(UnsupportedFileTypeError):
            await use_case.execute(uploaded, default_request)


# ── Deduplication ─────────────────────────────────────────────────────────────


class TestDeduplication:
    async def test_re_ingesting_the_same_file_creates_no_new_chunks(
        self,
        use_case: IngestDocumentsUseCase,
        vector_store: FakeVectorStore,
        default_request: IngestRequest,
    ) -> None:
        content = b"Stable document content for deduplication testing. " * 5
        uploaded = UploadedFile(filename="stable.txt", content=content)

        first = await use_case.execute(uploaded, default_request)
        stored_after_first = len(vector_store.stored)
        second = await use_case.execute(uploaded, default_request)

        assert first.chunks_created > 0
        assert first.duplicates_skipped == 0
        assert second.chunks_created == 0
        assert second.duplicates_skipped == first.chunks_created
        assert len(vector_store.stored) == stored_after_first

    async def test_same_content_under_another_filename_is_not_a_duplicate(
        self,
        use_case: IngestDocumentsUseCase,
        default_request: IngestRequest,
    ) -> None:
        """Chunk IDs are derived from source + content, so provenance is preserved."""
        content = b"Stable document content for deduplication testing. " * 5

        await use_case.execute(UploadedFile(filename="a.txt", content=content), default_request)
        other = await use_case.execute(
            UploadedFile(filename="b.txt", content=content), default_request
        )

        assert other.chunks_created > 0
        assert other.duplicates_skipped == 0

    async def test_failed_lookup_falls_back_to_upserting_everything(
        self,
        registry: ParserRegistry,
        default_request: IngestRequest,
    ) -> None:
        """Deduplication is an optimisation; losing it must not fail the ingest."""

        class LookupFails(FakeVectorStore):
            async def existing_ids(self, ids: list[str]) -> set[str]:
                raise ConnectionError("store unreachable")

        store = LookupFails()
        use_case = IngestDocumentsUseCase(
            vector_store=store, parser_registry=registry, default_collection="c"
        )
        content = b"Stable document content for deduplication testing. " * 5

        result = await use_case.execute(
            UploadedFile(filename="stable.txt", content=content), default_request
        )

        assert result.chunks_created == len(store.stored) > 0


# ── Custom Collection ──────────────────────────────────────────────────────────


class TestCollectionRouting:
    async def test_custom_collection_in_response(
        self,
        use_case: IngestDocumentsUseCase,
        default_request: IngestRequest,
    ) -> None:
        request = IngestRequest(collection="hr_policies", chunk_size=256, overlap=32)
        uploaded = UploadedFile(filename="hr.txt", content=b"HR policy document content. " * 10)

        result = await use_case.execute(uploaded, request)
        assert result.collection == "hr_policies"
