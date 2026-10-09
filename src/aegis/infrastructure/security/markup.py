"""
HTML tag stripping shared by the input gateway and the output sanitizer.

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
