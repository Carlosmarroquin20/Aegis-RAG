"""
Output Sanitizer — OWASP LLM05 (Improper Output Handling) mitigation.

The LLM's response is untrusted output: even with a hardened system prompt,
a sufficiently adversarial document in the context window may have influenced
the model to produce harmful, misleading, or structurally dangerous content.

Sanitization pipeline:
  1. Length enforcement     — truncates responses that exceed a safe token budget.
  2. HTML/script stripping  — prevents XSS if the output is rendered in a browser
                              or consumed by a web application downstream.
     Markdown flattening    — keeps the answer plain text and makes every link
                              target visible instead of hidden behind its label.
  3. Reflection detection   — checks whether the output appears to echo back
                              injection payloads (adversarial prompt reflection).
  4. PII pattern detection  — flags potential PII leakage without blocking (logged
                              for review; blocking would require a dedicated classifier).
                              Covers US formats and Guatemalan DPI, NIT and phones.

The sanitizer is intentionally conservative: it modifies as little as possible
and logs what it does so security teams can tune the rules over time.
"""

from __future__ import annotations

import re

import structlog

from aegis.domain.models.security import SanitizedOutput
from aegis.domain.ports.security import OutputGuardPort, OutputReflectionError
from aegis.infrastructure.observability.metrics import output_reflections_total
from aegis.infrastructure.security.markup import flatten_markdown, strip_html_tags

logger = structlog.get_logger(__name__)

# ── Reflection Detection Signatures ──────────────────────────────────────────
# If the model's output contains these patterns, it may have been manipulated
# by instruction-carrying content in the retrieved documents.
_REFLECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(different|evil|uncensored|DAN)", re.IGNORECASE),
    re.compile(r"(system\s+prompt|initial\s+instructions?)\s*:", re.IGNORECASE),
    re.compile(r"<\s*(script|iframe|object)[^>]*>", re.IGNORECASE),
]

# ── PII Pattern Signatures (detection only, not blocking) ────────────────────
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    # Local part and domain are capped at their RFC 5321 lengths (64 / 253). The
    # caps are what keep this linear: with open-ended "+" quantifiers the engine
    # rescans to the end of the text from every position in a long dotted run.
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,253}\.[A-Za-z]{2,}\b"),
    ),
    ("phone_us", re.compile(r"\b(?:\+1[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")),
    # ── Guatemala ─────────────────────────────────────────────────────────────
    # DPI (CUI): 13 digits, usually written 4-5-4. The last four are the
    # department (01-22) and municipality codes; requiring a valid department
    # rules out most unrelated 13-digit numbers. The check digit is not
    # verified, so a mistyped DPI is still flagged, which is the safer error.
    ("dpi_gt", re.compile(r"\b\d{4}[ -]?\d{5}[ -]?(?:0[1-9]|1\d|2[0-2])(?!00)\d{2}\b")),
    # NIT: a run of digits, a hyphen and a check character (a digit or K). That
    # shape is shared by order and invoice numbers, so the label must be present.
    ("nit_gt", re.compile(r"\bNIT\b[\s:#.]{0,3}\d{4,12}-?[\dK]\b", re.IGNORECASE)),
    # Phone: eight digits starting 2-7, written 4-4, optionally with +502. A bare
    # 4-4 pair that looks like a year range ("2024-2025") is not a number.
    (
        "phone_gt",
        re.compile(
            r"\+502[\s.-]?\d{4}[\s-]?\d{4}\b"
            r"|\b(?!(?:19|20)\d{2}[\s-](?:19|20)\d{2}\b)[2-7]\d{3}[\s-]\d{4}\b"
        ),
    ),
]

_MAX_OUTPUT_CHARS: int = 8192


class OutputSanitizer(OutputGuardPort):
    """
    Stateless post-processing layer applied to every LLM response
    before it is serialized into the API response body.

    Args:
        max_chars:        Hard character limit on the output.
        strip_html:       Strip HTML/script tags from the output (default: True).
        block_reflection: Raise OutputReflectionError when injection reflection
                          is detected. Set False to log-only in permissive environments.
        flatten_markdown: Remove Markdown emphasis/heading markers and rewrite
                          links as "text (target)" (default: True). Disable only
                          for a client that renders Markdown and sanitizes it itself.
    """

    def __init__(
        self,
        max_chars: int = _MAX_OUTPUT_CHARS,
        strip_html: bool = True,
        block_reflection: bool = True,
        flatten_markdown: bool = True,
    ) -> None:
        self._max_chars = max_chars
        self._strip_html = strip_html
        self._flatten_markdown = flatten_markdown
        self._block_reflection = block_reflection

    def sanitize(self, raw_output: str, query_hash: str = "") -> SanitizedOutput:
        """
        Applies the full sanitization pipeline.
        Always returns a SanitizedOutput; callers check `.has_warnings` for alerts.
        """
        log = logger.bind(query_hash=query_hash[:16] if query_hash else "unknown")

        text = raw_output

        # ── Step 1: Length enforcement ─────────────────────────────────────────
        was_truncated = False
        if len(text) > self._max_chars:
            text = text[: self._max_chars]
            # Truncate at the last sentence boundary to avoid mid-sentence cuts.
            last_period = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if last_period > self._max_chars // 2:
                text = text[: last_period + 1]
            text += "\n\n[Response truncated for length]"
            was_truncated = True
            log.warning("output_sanitizer.truncated", original_len=len(raw_output))

        # ── Step 2: HTML stripping ─────────────────────────────────────────────
        if self._strip_html:
            text = strip_html_tags(text)

        # ── Step 2b: Markdown flattening ───────────────────────────────────────
        # Before reflection detection on purpose: emphasis markers inside a phrase
        # ("ignore **all previous** instructions") must not hide it from the scan.
        if self._flatten_markdown:
            text = flatten_markdown(text)

        # ── Step 3: Reflection detection ───────────────────────────────────────
        reflection_detected = any(p.search(text) for p in _REFLECTION_PATTERNS)
        if reflection_detected:
            output_reflections_total.inc()
            log.error(
                "output_sanitizer.reflection_detected",
                query_hash=query_hash[:16] if query_hash else "unknown",
            )
            if self._block_reflection:
                raise OutputReflectionError(
                    "LLM output contains patterns consistent with prompt injection reflection. "
                    "The response has been blocked."
                )

        # ── Step 4: PII detection (log-only, non-blocking) ────────────────────
        pii_found = [label for label, pattern in _PII_PATTERNS if pattern.search(text)]
        if pii_found:
            log.warning("output_sanitizer.pii_detected", types=pii_found)

        return SanitizedOutput(
            text=text,
            was_truncated=was_truncated,
            reflection_detected=reflection_detected,
            pii_types_detected=pii_found,
        )
