"""
Markup handling shared by the input gateway and the output sanitizer.

Both sides need the same definition of "a tag": the gateway removes markup from
approved queries before retrieval, and the sanitizer removes it from LLM output
before it can reach a browser. Keeping one pattern here means a fix to the
definition cannot be applied to one direction and forgotten in the other.
"""

from __future__ import annotations

import re
from typing import Final

# A tag must open the way the HTML tokenizer requires: "<" followed immediately
# by a letter, "/", "!" or "?". Anything else — notably "<" followed by a space
# or a digit — is text to a browser and is left alone. The earlier pattern,
# "<[^>]{0,500}>", treated every "<" ... ">" pair as markup and deleted ordinary
# prose such as "revenue < 5M and margin > 20%".
#
# The body is length-bounded so an unterminated "<" cannot make the engine scan
# to the end of the input from every candidate position.
HTML_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"<(?:[/!?]?[A-Za-z]|!--)[^>]{0,500}>")


def strip_html_tags(text: str) -> str:
    """Removes HTML/XML tags, comments, doctypes and processing instructions."""
    return HTML_TAG_PATTERN.sub("", text)


# ── Markdown flattening ───────────────────────────────────────────────────────
#
# The API returns plain text, and the system prompt asks the model for it, but
# hosted models in particular still decorate answers with Markdown. Left in, the
# markers show up literally in any client that does not render Markdown; rendered,
# they give retrieved content a way to shape what the user sees — most notably a
# link whose visible text says one thing and whose target is another.
#
# The rules are deliberately narrow. Each one fires only on an unambiguous
# Markdown construct, because the same characters are ordinary text elsewhere:
# "2 * 3 * 4", "snake_case", "__init__", "# of items". Underscore emphasis is not
# handled at all for that reason. Every quantifier is bounded and excludes the
# delimiter, so none of these can backtrack on adversarial output.

# **bold** — the content must touch both delimiters, as Markdown itself requires.
_BOLD: Final[re.Pattern[str]] = re.compile(r"\*\*(?=\S)([^*\n]{1,500}?)(?<=\S)\*\*")
# *italic* — not adjacent to a word character or another asterisk, which rules
# out multiplication ("a*b*c") and leftovers of the bold rule.
_ITALIC: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w*])\*(?=\S)([^*\n]{1,500}?)(?<=\S)\*(?![\w*])"
)
# "# Heading" at the start of a line. The space is required: "#hashtag" and
# "#42" are not headings.
_HEADING: Final[re.Pattern[str]] = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
# [text](target) and ![alt](target). Rewritten as "text (target)" so the real
# destination is always visible next to the label it was hiding behind.
_LINK: Final[re.Pattern[str]] = re.compile(r"!?\[([^\]\n]{1,500})\]\(([^)\s]{1,2000})\)")


def flatten_markdown(text: str) -> str:
    """
    Removes Markdown emphasis and heading markers and exposes link targets.

    Lists, code spans and tables are left alone: they read acceptably as plain
    text, and stripping them would lose structure rather than decoration.
    """
    text = _LINK.sub(r"\1 (\2)", text)
    text = _BOLD.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    return _HEADING.sub("", text)
