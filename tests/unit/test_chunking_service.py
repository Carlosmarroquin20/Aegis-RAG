"""
Unit tests for ChunkingService.

Validates chunking behavior under normal conditions, boundary cases,
and adversarial inputs (empty content, single oversized sentence, etc.).
"""

from __future__ import annotations

import pytest

from aegis.domain.models.ingestion import RawDocument
from aegis.domain.services.chunking_service import ChunkingService


@pytest.fixture()
def chunker() -> ChunkingService:
    return ChunkingService(chunk_size=200, overlap=30, min_chunk_size=20)


# ── Core Behavior ──────────────────────────────────────────────────────────────


class TestChunkingBasics:
    def test_short_document_produces_single_chunk(self, chunker: ChunkingService) -> None:
        doc = RawDocument(content="This is a short document.", source="test.txt")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "This is a short document."

    def test_long_document_produces_multiple_chunks(self, chunker: ChunkingService) -> None:
        # 10 paragraphs of 50 chars each; total > chunk_size=200 → must split.
        paragraphs = [f"This is paragraph number {i:02d}. It has some text." for i in range(10)]
        content = "\n\n".join(paragraphs)
        doc = RawDocument(content=content, source="long.txt")
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_chunk_size_respected(self, chunker: ChunkingService) -> None:
        # Each chunk should not vastly exceed chunk_size
        # (sentence boundary may cause minor overflow).
        doc = RawDocument(content=" ".join(["word"] * 500), source="words.txt")
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            # Allow up to 2x chunk_size for single-sentence overflow.
            assert len(chunk.content) <= chunker.chunk_size * 2

    def test_overlap_text_present_in_consecutive_chunks(self, chunker: ChunkingService) -> None:
        """The overlap region of chunk N should appear at the start of chunk N+1."""
        long_text = " ".join([f"sentence{i}." for i in range(50)])
        doc = RawDocument(content=long_text, source="overlap_test.txt")
        chunks = chunker.chunk(doc)
        if len(chunks) < 2:
            pytest.skip("Document too short to produce multiple chunks with this config.")

        # The end of chunk[0]'s body should appear somewhere in chunk[1].
        body0 = chunks[0].content
        tail = body0[-chunker.overlap :]
        # Tail must be present in chunk[1] (possibly with minor whitespace variation).
        assert tail.strip()[:10] in chunks[1].content


# ── Metadata Propagation ───────────────────────────────────────────────────────


class TestMetadataPropagation:
    def test_source_propagated_to_chunks(self, chunker: ChunkingService) -> None:
        doc = RawDocument(content="Some content here.", source="report.pdf", page_number=3)
        chunks = chunker.chunk(doc)
        assert all(c.metadata["source"] == "report.pdf" for c in chunks)

    def test_page_number_in_metadata(self, chunker: ChunkingService) -> None:
        doc = RawDocument(content="Page content.", source="book.pdf", page_number=7)
        chunks = chunker.chunk(doc)
        assert all(c.metadata["page"] == "7" for c in chunks)

    def test_section_in_metadata(self, chunker: ChunkingService) -> None:
        doc = RawDocument(content="Section body.", source="doc.docx", section="Introduction")
        chunks = chunker.chunk(doc)
        assert all(c.metadata["section"] == "Introduction" for c in chunks)

    def test_custom_metadata_preserved(self, chunker: ChunkingService) -> None:
        doc = RawDocument(
            content="Content.",
            source="file.txt",
            metadata={"author": "Alice", "department": "legal"},
        )
        chunks = chunker.chunk(doc)
        assert chunks[0].metadata["author"] == "Alice"
        assert chunks[0].metadata["department"] == "legal"

    def test_chunk_index_is_sequential(self, chunker: ChunkingService) -> None:
        paragraphs = ["Paragraph {:02d}: " + "word " * 30 for _ in range(20)]
        content = "\n\n".join(p.format(i) for i, p in enumerate(paragraphs))
        doc = RawDocument(content=content, source="seq.txt")
        chunks = chunker.chunk(doc)
        indices = [int(c.metadata["chunk_index"]) for c in chunks]
        assert indices == list(range(len(chunks)))


# ── Content-Addressed IDs ──────────────────────────────────────────────────────


class TestContentAddressedIDs:
    def test_same_content_same_id(self, chunker: ChunkingService) -> None:
        doc = RawDocument(content="Deterministic content.", source="a.txt")
        run1 = chunker.chunk(doc)
        run2 = chunker.chunk(doc)
        assert run1[0].id == run2[0].id

    def test_different_content_different_id(self, chunker: ChunkingService) -> None:
        doc_a = RawDocument(content="Content A.", source="a.txt")
        doc_b = RawDocument(content="Content B.", source="a.txt")
        assert chunker.chunk(doc_a)[0].id != chunker.chunk(doc_b)[0].id

    def test_same_content_different_source_different_id(self, chunker: ChunkingService) -> None:
        doc_a = RawDocument(content="Same text.", source="file_a.txt")
        doc_b = RawDocument(content="Same text.", source="file_b.txt")
        # Source is included in the hash seed to distinguish duplicate content across files.
        assert chunker.chunk(doc_a)[0].id != chunker.chunk(doc_b)[0].id


# ── Edge Cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_min_chunk_size_filters_tiny_chunks(self) -> None:
        chunker = ChunkingService(chunk_size=200, overlap=10, min_chunk_size=50)
        # Content that would produce tiny chunks after splitting.
        doc = RawDocument(
            content="Hi.\n\nHi.\n\nThis paragraph is long enough to survive the filter.",
            source="tiny.txt",
        )
        chunks = chunker.chunk(doc)
        assert all(len(c.content) >= chunker.min_chunk_size for c in chunks)

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be smaller"):
            ChunkingService(chunk_size=100, overlap=100)

    def test_single_very_long_word_handled(self, chunker: ChunkingService) -> None:
        # A single token longer than chunk_size should not crash the chunker.
        long_word = "a" * 1000
        doc = RawDocument(content=long_word, source="longword.txt")
        chunks = chunker.chunk(doc)
        # May produce 0 chunks if below min_chunk_size after word-split, or several chunks.
        reconstructed = "".join(c.content for c in chunks)
        assert long_word[:100] in reconstructed or len(chunks) > 0
