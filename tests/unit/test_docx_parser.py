"""
Unit tests for the DOCX parser.

Test documents are generated in memory with python-docx, so no fixture files are
needed and the real extraction path is exercised end to end.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from typing import Any

import pytest
from docx import Document as DocxDocument

from aegis.domain.models.ingestion import SupportedMimeType
from aegis.domain.ports.document_parser import ParseError
from aegis.infrastructure.parsers import docx_parser
from aegis.infrastructure.parsers.docx_parser import DocxParser


def _docx_bytes(build: Callable[[Any], None]) -> bytes:
    doc = DocxDocument()
    build(doc)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


class TestDocxParser:
    def test_supported_mime_types(self) -> None:
        assert SupportedMimeType.APPLICATION_DOCX in DocxParser().supported_mime_types()

    def test_flat_document_returns_single_section(self) -> None:
        def build(doc: Any) -> None:
            doc.add_paragraph("First paragraph.")
            doc.add_paragraph("Second paragraph.")

        docs = DocxParser().parse(_docx_bytes(build), "flat.docx")
        assert len(docs) == 1
        assert "First paragraph." in docs[0].content
        assert "Second paragraph." in docs[0].content
        assert docs[0].source == "flat.docx"

    def test_headings_split_into_sections(self) -> None:
        def build(doc: Any) -> None:
            doc.add_heading("Introduction", level=1)
            doc.add_paragraph("Intro body.")
            doc.add_heading("Details", level=1)
            doc.add_paragraph("Details body.")

        docs = DocxParser().parse(_docx_bytes(build), "structured.docx")
        assert len(docs) == 2
        assert docs[0].section == "Introduction"
        assert "Intro body." in docs[0].content
        assert docs[1].section == "Details"

    def test_corrupt_bytes_raise_parse_error(self) -> None:
        with pytest.raises(ParseError):
            DocxParser().parse(b"this is not a docx file", "bad.docx")

    def test_empty_document_raises_parse_error(self) -> None:
        def build(_doc: Any) -> None:
            return None  # no content

        with pytest.raises(ParseError):
            DocxParser().parse(_docx_bytes(build), "empty.docx")


class TestDecompressionBombGuard:
    """A DOCX is a ZIP: the upload limit bounds compressed bytes, not inflated ones."""

    @staticmethod
    def _zip_with_member(payload: bytes, *, members: int = 1) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for index in range(members):
                archive.writestr(f"word/part{index}.xml", payload)
        return buffer.getvalue()

    def test_highly_compressed_archive_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(docx_parser, "_MAX_UNCOMPRESSED_BYTES", 1024 * 1024)
        bomb = self._zip_with_member(b"\0" * (4 * 1024 * 1024))
        # The whole point: tiny on the wire, large once inflated.
        assert len(bomb) < 16 * 1024

        with pytest.raises(ParseError, match="uncompressed size"):
            DocxParser().parse(bomb, "bomb.docx")

    def test_archive_with_too_many_members_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docx_parser, "_MAX_ARCHIVE_MEMBERS", 8)
        with pytest.raises(ParseError, match="uncompressed size"):
            DocxParser().parse(self._zip_with_member(b"x", members=9), "many.docx")

    def test_guard_runs_before_the_document_is_opened(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rejecting after python-docx has inflated the parts would be too late."""
        import docx

        def fail(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("document was opened before the size guard ran")

        monkeypatch.setattr(docx, "Document", fail)
        monkeypatch.setattr(docx_parser, "_MAX_UNCOMPRESSED_BYTES", 16)
        with pytest.raises(ParseError):
            DocxParser().parse(self._zip_with_member(b"x" * 64), "bomb.docx")

    def test_ordinary_document_passes_the_guard(self) -> None:
        docs = DocxParser().parse(_docx_bytes(lambda d: d.add_paragraph("Hello.")), "ok.docx")
        assert docs[0].content == "Hello."
