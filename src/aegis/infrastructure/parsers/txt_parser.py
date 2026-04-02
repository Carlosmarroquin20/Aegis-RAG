"""Plain-text parser. UTF-8 with Latin-1 fallback."""

from __future__ import annotations

from aegis.domain.models.ingestion import RawDocument, SupportedMimeType
from aegis.domain.ports.document_parser import DocumentParserPort, ParseError


class TxtParser(DocumentParserPort):
    def supported_mime_types(self) -> frozenset[str]:
        return frozenset({SupportedMimeType.TEXT_PLAIN})

    def parse(self, content: bytes, filename: str) -> list[RawDocument]:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except UnicodeDecodeError as exc:
                raise ParseError(filename, "unable to decode as UTF-8 or Latin-1") from exc

        text = text.strip()
        if not text:
            raise ParseError(filename, "file is empty after stripping whitespace")

        return [RawDocument(content=text, source=filename)]
