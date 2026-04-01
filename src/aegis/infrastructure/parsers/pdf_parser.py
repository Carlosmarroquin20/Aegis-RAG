"""
PDF parser using pypdf.

Returns one RawDocument per page. Per-page granularity matters for citation
accuracy: knowing the exact page number is more useful than a single chunk
with a file-level source attribution.

Security notes:
  - pypdf does not execute JavaScript or render content — it extracts text only.
  - Password-protected PDFs are rejected with a clear ParseError rather than
    silently returning empty content.
  - Abnormally large page counts (> 1000) are capped to prevent resource exhaustion
    from adversarially crafted PDFs.
"""
from __future__ import annotations

import io

import structlog

from aegis.domain.models.ingestion import RawDocument, SupportedMimeType
from aegis.domain.ports.document_parser import DocumentParserPort, ParseError

logger = structlog.get_logger(__name__)

_MAX_PAGES: int = 1000


class PdfParser(DocumentParserPort):
    def supported_mime_types(self) -> frozenset[str]:
        return frozenset({SupportedMimeType.APPLICATION_PDF})

    def parse(self, content: bytes, filename: str) -> list[RawDocument]:
        try:
            import pypdf
        except ImportError as exc:
            raise ParseError(filename, "pypdf is not installed") from exc

        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
        except Exception as exc:  # noqa: BLE001
            raise ParseError(filename, f"could not open PDF: {exc}") from exc

        if reader.is_encrypted:
            raise ParseError(filename, "password-protected PDFs are not supported")

        total_pages = len(reader.pages)
        if total_pages > _MAX_PAGES:
            logger.warning(
                "pdf_parser.page_cap", filename=filename, total=total_pages, cap=_MAX_PAGES
            )

        documents: list[RawDocument] = []
        for page_num in range(min(total_pages, _MAX_PAGES)):
            try:
                page_text = reader.pages[page_num].extract_text() or ""
                page_text = page_text.strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("pdf_parser.page_extract_failed", page=page_num + 1, error=str(exc))
                continue

            if not page_text:
                continue  # Skip blank pages (e.g., separator pages, pure image pages).

            documents.append(
                RawDocument(
                    content=page_text,
                    source=filename,
                    page_number=page_num + 1,
                    metadata={"total_pages": str(total_pages), "format": "pdf"},
                )
            )

        if not documents:
            raise ParseError(filename, "no extractable text found — PDF may be image-only")

        logger.debug("pdf_parser.complete", filename=filename, pages_extracted=len(documents))
        return documents
