"""
Document parser port (abstract interface).

A parser converts raw bytes of a specific MIME type into a list of RawDocuments.
Concrete implementations: TxtParser, MarkdownParser, PdfParser, DocxParser.

Design note: parsers are synchronous because PDF/DOCX extraction is CPU-bound
and benefits from process-level parallelism rather than async I/O.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aegis.domain.models.ingestion import RawDocument


class DocumentParserPort(ABC):
    """Hexagonal port for file-format-specific content extraction."""

    @abstractmethod
    def supported_mime_types(self) -> frozenset[str]:
        """Returns the set of MIME types this parser handles."""

    @abstractmethod
    def parse(self, content: bytes, filename: str) -> list[RawDocument]:
        """
        Extracts text content from raw bytes.

        Args:
            content:  Raw file bytes as received from the upload or file system.
            filename: Original filename; used to populate provenance metadata.

        Returns:
            One or more RawDocuments. PDF parsers typically return one per page;
            plain-text parsers return a single RawDocument.

        Raises:
            ParseError: If the content is corrupt, password-protected, or otherwise
                        unreadable. Callers should log the error and skip the file.
        """


class ParseError(Exception):
    """Raised when a parser cannot extract content from a file."""

    def __init__(self, filename: str, reason: str) -> None:
        super().__init__(f"Failed to parse '{filename}': {reason}")
        self.filename = filename
        self.reason = reason
