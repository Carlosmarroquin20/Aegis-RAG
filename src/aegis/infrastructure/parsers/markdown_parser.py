"""
Markdown parser — strips markup and returns plain prose.

Uses markdown-it-py to render to HTML, then strips tags, rather than
trying to regex-strip Markdown syntax directly (which breaks on edge cases
like nested code blocks, tables, and footnotes).

The rendered HTML is an intermediate form only. Rendering escapes every "&",
"<", ">" and quote in the source as a character reference, so the references
must be decoded again after the tags are removed; otherwise the index ends up
holding "R&amp;D" and "a &lt; b" instead of the text the author wrote, which
skews embeddings and is what the LLM would later be shown as context.
"""

from __future__ import annotations

import html
import re

from aegis.domain.models.ingestion import RawDocument, SupportedMimeType
from aegis.domain.ports.document_parser import DocumentParserPort, ParseError

# Closing tags that end a block: replaced by a newline so paragraph structure
# survives tag removal and the chunker still sees paragraph boundaries.
_BLOCK_END = re.compile(r"</(p|h[1-6]|li|blockquote|pre|div)>", re.IGNORECASE)
# Tables: one line per row, cells separated rather than concatenated ("TeamCost").
# The renderer puts every cell and row on its own line, so the whitespace that
# follows each structural tag is consumed along with it.
_CELL_END = re.compile(r"</(td|th)>\s*", re.IGNORECASE)
_ROW_OR_TABLE_END = re.compile(r"</(tr|table)>\s*", re.IGNORECASE)
_TABLE_SCAFFOLD = re.compile(r"<(table|thead|tbody|tr)>\s*|</(thead|tbody)>\s*", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_CELL_SEPARATOR = re.compile(r"[ \t]*\|[ \t]*$", re.MULTILINE)


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
        from markdown_it import MarkdownIt

        # CommonMark has no tables; without the rule a table is passed through as
        # literal pipes and dashes, which is noise to an embedding model.
        rendered = MarkdownIt("commonmark").enable("table").render(text)

        rendered = _TABLE_SCAFFOLD.sub("", rendered)
        rendered = _CELL_END.sub(" | ", rendered)
        rendered = _ROW_OR_TABLE_END.sub("\n", rendered)
        rendered = _BLOCK_END.sub("\n", rendered)
        plain = _TAG.sub("", rendered)
        # Decode only after the tags are gone: doing it first would turn text the
        # author escaped on purpose ("&lt;b&gt;") into markup and delete it.
        plain = html.unescape(plain)
        plain = _TRAILING_CELL_SEPARATOR.sub("", plain)
        return _EXCESS_BLANK_LINES.sub("\n\n", plain)
