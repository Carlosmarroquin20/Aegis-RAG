"""
ParserRegistry — central dispatcher for document parsers.

Responsibilities:
  1. MIME type detection from magic bytes (not the Content-Type header,
     which is attacker-controlled and cannot be trusted for security decisions).
  2. Parser lookup: maps a MIME type to the correct DocumentParserPort implementation.

The registry is pre-populated with all built-in parsers. Additional parsers can
be registered at runtime via register() for extensibility.
"""

from __future__ import annotations

import structlog

from aegis.domain.models.ingestion import SupportedMimeType
from aegis.domain.ports.document_parser import DocumentParserPort
from aegis.infrastructure.parsers.docx_parser import DocxParser
from aegis.infrastructure.parsers.markdown_parser import MarkdownParser
from aegis.infrastructure.parsers.pdf_parser import PdfParser
from aegis.infrastructure.parsers.txt_parser import TxtParser

logger = structlog.get_logger(__name__)

# Extensions that should be treated as Markdown regardless of magic bytes,
# because plain Markdown files have no distinguishing byte signature.
_MARKDOWN_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown", ".mdx"})


class ParserRegistry:
    """
    Stateful registry; instantiate once and inject as a singleton.
    Thread-safe for reads (get_parser, detect_mime_type); register() is not thread-safe
    after initial setup and should only be called during application startup.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, DocumentParserPort] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for parser in [TxtParser(), MarkdownParser(), PdfParser(), DocxParser()]:
            self.register(parser)

    def register(self, parser: DocumentParserPort) -> None:
        for mime_type in parser.supported_mime_types():
            self._parsers[mime_type] = parser

    def get_parser(self, mime_type: str) -> DocumentParserPort | None:
        return self._parsers.get(mime_type)

    @staticmethod
    def detect_mime_type(content: bytes, filename: str) -> str:
        """
        Detects MIME type from magic bytes using the `filetype` library.
        Falls back to extension-based inference for text files (which have no
        magic bytes and are therefore invisible to magic-byte scanners).

        Security note: always call this before passing content to a parser.
        Trusting the client-supplied Content-Type would allow file type spoofing.
        """
        import filetype

        kind = filetype.guess(content)
        if kind is not None:
            return str(kind.mime)

        # Text files have no magic bytes — infer from extension.
        lower = filename.lower()
        if any(lower.endswith(ext) for ext in _MARKDOWN_EXTENSIONS):
            return SupportedMimeType.TEXT_MARKDOWN

        # Default to plain text for all other unrecognized content.
        # The TxtParser will raise a ParseError if decoding fails.
        return SupportedMimeType.TEXT_PLAIN

    @property
    def supported_mime_types(self) -> list[str]:
        return sorted(self._parsers.keys())
