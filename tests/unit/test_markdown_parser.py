"""
Unit tests for the Markdown parser.

The parser's output is what gets embedded and later shown to the LLM as context,
so the tests assert on the exact text produced, not just that something came out.
"""

from __future__ import annotations

import pytest

from aegis.domain.models.ingestion import SupportedMimeType
from aegis.domain.ports.document_parser import ParseError
from aegis.infrastructure.parsers.markdown_parser import MarkdownParser


def _text(markdown: str) -> str:
    return MarkdownParser().parse(markdown.encode(), "doc.md")[0].content


class TestMarkupIsRemoved:
    def test_headings_emphasis_and_links_become_prose(self) -> None:
        text = _text("# Title\n\nSome **bold** and *italic* with a [link](https://x.test).")
        assert text == "Title\n\nSome bold and italic with a link."

    def test_list_items_stay_on_separate_lines(self) -> None:
        assert _text("- one\n- two\n- three").split() == ["one", "two", "three"]

    def test_paragraph_boundaries_survive(self) -> None:
        """The chunker splits on blank lines, so they must not be flattened."""
        assert "\n\n" in _text("First paragraph.\n\nSecond paragraph.")

    def test_table_cells_are_separated_not_concatenated(self) -> None:
        table = "| Team | Cost |\n|------|------|\n| Ops  | 5    |\n"
        text = _text(table)
        assert text.splitlines() == ["Team | Cost", "Ops | 5"]
        assert "---" not in text


class TestCharactersAreNotCorrupted:
    """
    Regression: the HTML renderer escapes special characters, and the parser
    used to leave those character references in the indexed text.
    """

    @pytest.mark.parametrize(
        ("markdown", "expected"),
        [
            ("R&D budget", "R&D budget"),
            ('The "quoted" phrase', 'The "quoted" phrase'),
            ("when a < b and c > d", "when a < b and c > d"),
            ("Tom & Jerry's", "Tom & Jerry's"),
            ("Use `x && y` here", "Use x && y here"),
        ],
        ids=["ampersand", "quotes", "comparison", "apostrophe", "inline-code"],
    )
    def test_special_characters_round_trip(self, markdown: str, expected: str) -> None:
        assert _text(markdown) == expected

    def test_fenced_code_keeps_its_operators(self) -> None:
        text = _text('```python\nif a < b and name == "x":\n    pass\n```')
        assert 'if a < b and name == "x":' in text
        assert "&" not in text

    def test_entities_written_by_the_author_are_decoded_once(self) -> None:
        assert _text("Copyright &copy; 2026, 5 &lt; 6") == "Copyright © 2026, 5 < 6"

    def test_escaped_markup_is_kept_as_text_not_executed_as_tags(self) -> None:
        """Decoding must happen after tag removal, or this would vanish."""
        assert _text("Write `<b>bold</b>` to emphasise.") == "Write <b>bold</b> to emphasise."


class TestRejections:
    def test_invalid_utf8_is_a_parse_error(self) -> None:
        with pytest.raises(ParseError, match="UTF-8"):
            MarkdownParser().parse(b"\xff\xfe\x00bad", "bad.md")

    @pytest.mark.parametrize("empty", [b"", b"   \n\n  ", b"<!-- only a comment -->"])
    def test_document_without_readable_text_is_a_parse_error(self, empty: bytes) -> None:
        with pytest.raises(ParseError, match="no readable content"):
            MarkdownParser().parse(empty, "empty.md")


class TestMetadata:
    def test_supported_mime_types(self) -> None:
        assert SupportedMimeType.TEXT_MARKDOWN in MarkdownParser().supported_mime_types()

    def test_source_and_format_are_recorded(self) -> None:
        doc = MarkdownParser().parse(b"hello", "notes.md")[0]
        assert doc.source == "notes.md"
        assert doc.metadata == {"format": "markdown"}
