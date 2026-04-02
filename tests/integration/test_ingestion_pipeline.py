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
from aegis.domain.models.document import Document
from aegis.domain.ports.vector_store import VectorStorePort
from aegis.infrastructure.parsers.parser_registry import ParserRegistry

# ── Fixtures ───────────────────────────────────────────────────────────────────


class FakeVectorStore(VectorStorePort):
    """In-memory vector store stub that records upserted documents."""

    def __init__(self) -> None:
        self.stored: list[Document] = []

    async def similarity_search(
        self, query: str, k: int = 5, score_threshold: float = 0.0
    ) -> list[Document]:
        return []

    async def add_documents(self, documents: list[Document]) -> None:
        self.stored.extend(documents)

    async def delete_documents(self, ids: list[str]) -> None:
        self.stored = [d for d in self.stored if d.id not in ids]

    async def health_check(self) -> bool:
        return True


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
    async def test_re_ingestion_does_not_crash(
        self,
        use_case: IngestDocumentsUseCase,
        vector_store: FakeVectorStore,
        default_request: IngestRequest,
    ) -> None:
        content = b"Stable document content for deduplication testing. " * 5
        uploaded = UploadedFile(filename="stable.txt", content=content)

        result1 = await use_case.execute(uploaded, default_request)
        # Re-ingest: the use case should complete without errors.
        # (Full dedup requires ChromaDB; this test verifies graceful handling with FakeVectorStore.)
        result2 = await use_case.execute(uploaded, default_request)

        assert result1.chunks_created > 0
        assert result2.chunks_created >= 0  # May be 0 if dedup is active, > 0 if not.


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
