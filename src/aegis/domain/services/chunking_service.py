"""
Sentence-aware chunking service.

Strategy (in priority order):
  1. Split on paragraph boundaries (blank lines) — natural semantic units.
  2. If a paragraph exceeds chunk_size, split on sentence boundaries (regex-based,
     no NLTK dependency) to avoid cutting mid-thought.
  3. If a single sentence still exceeds chunk_size, fall back to word-boundary splits.

Overlap is applied by prepending the tail of the previous chunk to each new chunk,
preserving cross-boundary context critical for retrieval accuracy.

This lives in the domain layer because chunking is a core business concern —
it directly affects retrieval quality and is independent of any infrastructure.
"""

from __future__ import annotations

import hashlib
import re

import structlog

from aegis.domain.models.document import Document
from aegis.domain.models.ingestion import RawDocument

logger = structlog.get_logger(__name__)

# Sentence boundary: period/!/?  followed by one or more spaces and an uppercase letter,
# OR a newline. Avoids splitting on abbreviations like "U.S." or "Dr. Smith".
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)")
_WHITESPACE_RUN = re.compile(r"\s{3,}")


class ChunkingService:
    """
    Converts a RawDocument into a list of overlapping Document chunks.

    Args:
        chunk_size:    Target character count per chunk. Not a hard limit —
                       sentence boundaries take precedence to avoid mid-sentence splits.
        overlap:       Characters from the end of the previous chunk to prepend
                       to the next one. Typical: 10–15% of chunk_size.
        min_chunk_size: Chunks shorter than this are discarded (degenerate content).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        min_chunk_size: int = 50,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, raw_doc: RawDocument) -> list[Document]:
        """
        Produces Document objects ready for upsert into the vector store.
        Each chunk inherits the source document's metadata and gets a stable
        content-addressed ID (SHA-256 of content) for idempotent upserts.
        """
        text = _normalize_whitespace(raw_doc.content)
        raw_chunks = self._split(text)

        documents: list[Document] = []
        doc_index = 0
        for chunk_text in raw_chunks:
            if len(chunk_text) < self.min_chunk_size:
                continue

            # Stable, content-addressed ID: identical content produces the same ID
            # across re-ingestions, enabling idempotent upserts in the vector store.
            chunk_id = hashlib.sha256(f"{raw_doc.source}:{chunk_text}".encode()).hexdigest()[:32]

            metadata: dict[str, str] = {
                **raw_doc.metadata,
                "source": raw_doc.source,
                "chunk_index": str(doc_index),
            }
            if raw_doc.page_number is not None:
                metadata["page"] = str(raw_doc.page_number)
            if raw_doc.section:
                metadata["section"] = raw_doc.section

            documents.append(Document(id=chunk_id, content=chunk_text, metadata=metadata))
            doc_index += 1

        # Fallback: if min_chunk_size filtered every candidate but the document has
        # content, emit the first raw chunk anyway. This prevents callers from silently
        # losing short-but-valid documents (e.g. single-sentence files).
        if not documents and text.strip():
            chunk_text = raw_chunks[0] if raw_chunks else text.strip()
            chunk_id = hashlib.sha256(f"{raw_doc.source}:{chunk_text}".encode()).hexdigest()[:32]
            metadata = {
                **raw_doc.metadata,
                "source": raw_doc.source,
                "chunk_index": "0",
            }
            if raw_doc.page_number is not None:
                metadata["page"] = str(raw_doc.page_number)
            if raw_doc.section:
                metadata["section"] = raw_doc.section
            documents.append(Document(id=chunk_id, content=chunk_text, metadata=metadata))

        logger.debug(
            "chunking.complete",
            source=raw_doc.source,
            raw_chars=len(text),
            chunks=len(documents),
        )
        return documents

    def _split(self, text: str) -> list[str]:
        """
        Three-tier splitting strategy with overlap stitching.
        Returns raw text chunks before Document wrapping.
        """
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        sentences: list[str] = []

        for para in paragraphs:
            if len(para) <= self.chunk_size:
                sentences.append(para)
            else:
                # Paragraph too large — break at sentence boundaries.
                sentences.extend(self._split_sentences(para))

        return self._build_chunks_with_overlap(sentences)

    def _split_sentences(self, text: str) -> list[str]:
        parts = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
        result: list[str] = []
        for part in parts:
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                # Single sentence exceeds chunk_size — word-boundary fallback.
                result.extend(self._split_by_words(part))
        return result

    def _split_by_words(self, text: str) -> list[str]:
        """Last-resort: split at word boundaries when even sentences are too long."""
        words = text.split()
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for word in words:
            word_len = len(word) + 1  # +1 for the space
            if current_len + word_len > self.chunk_size and current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            current.append(word)
            current_len += word_len

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _build_chunks_with_overlap(self, sentences: list[str]) -> list[str]:
        """
        Groups sentences into chunks and prepends overlap from the previous chunk.
        The overlap window uses character count (not token count) for determinism.
        """
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0
        overlap_tail = ""

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_len + sentence_len > self.chunk_size and current_parts:
                chunk_text = overlap_tail + " ".join(current_parts)
                chunks.append(chunk_text.strip())
                # Capture overlap: last `overlap` chars of the current chunk body.
                tail = " ".join(current_parts)
                overlap_tail = tail[-self.overlap :] + " " if self.overlap else ""
                current_parts = []
                current_len = 0

            current_parts.append(sentence)
            current_len += sentence_len + 1  # +1 for join space

        if current_parts:
            chunk_text = overlap_tail + " ".join(current_parts)
            chunks.append(chunk_text.strip())

        return chunks


def _normalize_whitespace(text: str) -> str:
    """Collapses runs of 3+ whitespace chars (excluding newlines) into two spaces."""
    return _WHITESPACE_RUN.sub("  ", text)
