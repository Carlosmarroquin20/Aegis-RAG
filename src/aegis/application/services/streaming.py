"""
Releases a streaming answer to the client a sentence at a time, sanitized.

The output guard (OWASP LLM05) was built for whole answers: strip markup, flatten
Markdown, and refuse an answer that echoes an injection payload (ADR 0008).
Streaming cannot take text back once it is sent, so the guard runs on the way
out instead of at the end, under three rules:

- **Only whole sentences leave.** Text is released up to a sentence boundary,
  never mid-sentence, so a phrase the reflection check looks for ("ignore all
  previous instructions") cannot be half-sent before its end arrives.
- **A margin stays behind.** A boundary is released only once ``holdback``
  characters have arrived after it, and the reflection check runs over the
  whole buffer, margin included. A payload that straddles a boundary is caught
  before the text in front of the boundary goes.
- **Open markup waits.** A boundary inside an unclosed ``<`` tag or Markdown
  link is not used, so stripping and link rewriting always see the whole
  construct.

Released text is the sanitizer's own output for the released prefix. A rule
could, in principle, rewrite earlier text once more arrives; the guard never
re-sends, it stops emitting deltas instead, and the final answer (which the
client must treat as authoritative) carries the correct text.
"""

from __future__ import annotations

import re

from aegis.domain.models.security import SanitizedOutput
from aegis.domain.ports.security import OutputGuardPort

DEFAULT_HOLDBACK_CHARS = 80
DEFAULT_MAX_CHARS = 8192

# End of a sentence: terminal punctuation, optional closing quote or bracket,
# then whitespace; or a line break.
_BOUNDARY = re.compile(r"[.!?…][\"')\]]*\s|\n")


class StreamingOutputGuard:
    def __init__(
        self,
        guard: OutputGuardPort,
        *,
        query_hash: str = "",
        holdback: int = DEFAULT_HOLDBACK_CHARS,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        if holdback < 0:
            raise ValueError("holdback must not be negative")
        self._guard = guard
        self._query_hash = query_hash
        self._holdback = holdback
        self._max_chars = max_chars
        self._raw = ""
        self._released_to = 0  # how much of the raw text has been sanitized and released
        self._emitted = ""  # the sanitized text sent so far
        self._diverged = False

    @property
    def exhausted(self) -> bool:
        """The answer has reached the length limit; the caller should stop generating."""
        return len(self._raw) >= self._max_chars

    @property
    def diverged(self) -> bool:
        """Sanitizing more text rewrote text already sent; only the final answer is exact."""
        return self._diverged

    def feed(self, delta: str) -> str:
        """
        Adds generated text; returns the sanitized text that may now be sent,
        possibly empty.

        Raises:
            OutputReflectionError: the text so far echoes an injection payload.
        """
        self._raw += delta
        boundary = self._release_point()
        if boundary <= self._released_to:
            return ""
        # The whole buffer, margin included, is checked before anything in it goes.
        self._guard.sanitize(self._raw, self._query_hash, partial=True)
        released = self._guard.sanitize(self._raw[:boundary], self._query_hash, partial=True)
        self._released_to = boundary
        return self._advance(released.text)

    def finish(self) -> tuple[SanitizedOutput, str]:
        """
        Sanitizes the finished answer in full (length limit and PII reporting
        included). Returns it with the text still to send, possibly empty.

        Raises:
            OutputReflectionError: the answer echoes an injection payload.
        """
        final = self._guard.sanitize(self._raw, self._query_hash)
        return final, self._advance(final.text)

    def _advance(self, sanitized: str) -> str:
        if self._diverged or not sanitized.startswith(self._emitted):
            self._diverged = True
            return ""
        delta = sanitized[len(self._emitted) :]
        self._emitted = sanitized
        return delta

    def _release_point(self) -> int:
        """The furthest boundary that is safe to release, or 0."""
        # Never past the length limit: the finished answer is cut there.
        limit = min(len(self._raw) - self._holdback, self._max_chars)
        candidates = [m.end() for m in _BOUNDARY.finditer(self._raw, 0, max(limit, 0))]
        for boundary in reversed(candidates):
            if boundary <= self._released_to:
                break
            if boundary <= limit and not _open_markup(self._raw[:boundary]):
                return boundary
        return self._released_to


def _open_markup(text: str) -> bool:
    """An HTML tag or a Markdown link is still open at the end of ``text``."""
    if text.rfind("<") > text.rfind(">"):
        return True
    if text.rfind("[") > text.rfind("]"):
        return True
    return text.rfind("](") > text.rfind(")")
