"""
Markdown parser — strips markup and returns plain prose.

Uses markdown-it-py to render to HTML, then strips tags, rather than
trying to regex-strip Markdown syntax directly (which breaks on edge cases
like nested code blocks, tables, and footnotes).
"""
from __future__ import annotations

from aegis.domain.models.ingestion import RawDocument, SupportedMimeType
from aegis.domain.ports.document_parser import DocumentParserPort, ParseError


class MarkdownParser(DocumentParserPort):
    def supported_mime_types(self) -> frozenset[str]:
        return frozenset({SupportedMimeType.TEXT_MARKDOWN, "text/x-markdown"})

    def parse(self, content: bytes, filename: str) -> list[RawDocument]:
        try:
            raw_text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ParseError(filename, "not valid UTF-8") from exc

        plain = self._strip_markdown(raw_text).strip()
        if not plain:
            raise ParseError(filename, "no readable content after markdown stripping")

        return [RawDocument(content=plain, source=filename, metadata={"format": "markdown"})]

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """
        Converts Markdown to plain text via markdown-it-py → HTML → tag stripping.
        Preserves paragraph structure by replacing block-level tags with newlines.
        """
        import re

        from markdown_it import MarkdownIt

        md = MarkdownIt()
        html = md.render(text)

        # Replace block-level closing tags with newlines before stripping all tags.
        html = re.sub(r"</(p|h[1-6]|li|blockquote|pre|div)>", "\n", html, flags=re.IGNORECASE)
        # Strip all remaining HTML tags.
        plain = re.sub(r"<[^>]+>", "", html)
        # Collapse excessive blank lines.
        plain = re.sub(r"\n{3,}", "\n\n", plain)
        return plain
