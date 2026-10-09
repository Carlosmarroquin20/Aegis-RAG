"""
DOCX parser using python-docx.

Returns one RawDocument per logical section (heading-delimited).
Falls back to a single RawDocument containing all paragraphs if no headings
are present (flat documents without structure).

Heading detection uses python-docx's built-in style name matching ("Heading 1",
"Heading 2", etc.) rather than font size heuristics, which are unreliable.

Security note: a DOCX is a ZIP archive, so the upload size limit only bounds the
COMPRESSED bytes. Deflate reaches ratios above 1000:1 on repetitive input, which
lets a file well under the limit expand to gigabytes once python-docx inflates
its XML parts. The archive directory is therefore inspected before any member is
decompressed.
"""

from __future__ import annotations

import io
import zipfile

import structlog

from aegis.domain.models.ingestion import RawDocument, SupportedMimeType
from aegis.domain.ports.document_parser import DocumentParserPort, ParseError

logger = structlog.get_logger(__name__)

# Generous for real documents (a text-heavy DOCX of several hundred pages inflates
# to a few tens of MB; embedded media is stored nearly uncompressed and is already
# covered by the upload limit) while keeping worst-case memory bounded.
_MAX_UNCOMPRESSED_BYTES: int = 200 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS: int = 10_000


def _reject_decompression_bomb(content: bytes, filename: str) -> None:
    """
    Raises ParseError if the archive would inflate past the configured budget.

    Only the central directory is read, so the check itself costs nothing. The
    declared sizes can be trusted as an upper bound: zipfile stops reading a
    member at its declared length and fails the CRC check if the data disagrees.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise ParseError(filename, "not a valid DOCX archive") from exc

    declared_bytes = sum(member.file_size for member in members)
    if len(members) > _MAX_ARCHIVE_MEMBERS or declared_bytes > _MAX_UNCOMPRESSED_BYTES:
        logger.warning(
            "docx_parser.decompression_bomb_rejected",
            filename=filename,
            members=len(members),
            declared_bytes=declared_bytes,
            compressed_bytes=len(content),
        )
        raise ParseError(filename, "archive expands beyond the allowed uncompressed size")


class DocxParser(DocumentParserPort):
    def supported_mime_types(self) -> frozenset[str]:
        return frozenset({SupportedMimeType.APPLICATION_DOCX})

    def parse(self, content: bytes, filename: str) -> list[RawDocument]:
        try:
            from docx import Document as DocxDocument
        except ImportError as exc:
            raise ParseError(filename, "python-docx is not installed") from exc

        _reject_decompression_bomb(content, filename)

        try:
            doc = DocxDocument(io.BytesIO(content))
        except Exception as exc:  # noqa: BLE001
            raise ParseError(filename, f"could not open DOCX: {exc}") from exc

        sections = self._extract_sections(doc, filename)
        if not sections:
            raise ParseError(filename, "no readable text found in document")

        return sections

    @staticmethod
    def _extract_sections(doc: object, filename: str) -> list[RawDocument]:
        """
        Splits document into sections at Heading 1/2 boundaries.
        Each section becomes a RawDocument with the heading as metadata.
        """
        from typing import Any

        doc_typed: Any = doc
        raw_docs: list[RawDocument] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        def flush(heading: str | None, lines: list[str]) -> None:
            text = "\n".join(lines).strip()
            if text:
                raw_docs.append(
                    RawDocument(
                        content=text,
                        source=filename,
                        section=heading,
                        metadata={
                            "format": "docx",
                            **({"section": heading} if heading else {}),
                        },
                    )
                )

        for para in doc_typed.paragraphs:
            style_name: str = para.style.name if para.style else ""
            text = para.text.strip()

            if style_name.startswith("Heading"):
                flush(current_heading, current_lines)
                current_heading = text or style_name
                current_lines = []
            elif text:
                current_lines.append(text)

        flush(current_heading, current_lines)
        return raw_docs
