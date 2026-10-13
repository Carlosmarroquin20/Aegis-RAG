"""
Security Gateway — OWASP LLM Top 10 enforcement layer.

Every query passes through this pipeline before touching the RAG engine:
  1. Structural validation (length, null bytes)
  2. Unicode normalization (compatibility forms, invisible characters) plus a
     homoglyph-folded view used only for signature matching
  3. Heuristic threat scoring (injection signatures + entropy analysis)
  4. Deep sanitization (HTML/template stripping)

The gateway is intentionally I/O-free: no network calls, no DB access.
This keeps it fast (<1 ms for typical queries) and easily testable.

OWASP LLM coverage:
  LLM01 — Prompt Injection           → _INJECTION_SIGNATURES, _normalize_unicode,
                                       _fold_confusables
  LLM02 — Insecure Output Handling   → enforced at the LLM adapter layer (see llm_client.py)
  LLM06 — Sensitive Information      → rate limiter + API key enforcement (middleware)
  LLM07 — System Prompt Disclosure   → _INJECTION_SIGNATURES (prompt_leakage category)
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Final

import structlog

from aegis.domain.models.query import RawQuery, SanitizedQuery
from aegis.domain.models.security import GatewayResult, ThreatAssessment, ThreatLevel
from aegis.domain.ports.security import QueryGuardPort
from aegis.infrastructure.security.markup import strip_html_tags

logger = structlog.get_logger(__name__)

# ── Threat Signature Catalog ──────────────────────────────────────────────────
#
# Each entry: (compiled_pattern, threat_category, severity_score 0-100)
# Scores are NOT additive — the maximum matched score is used to avoid
# penalizing legitimate queries that happen to match low-confidence rules.
# Patterns are ordered by severity (highest first) for early-exit potential.
#
# Every pattern here runs on attacker-controlled input of up to _MAX_QUERY_CHARS,
# synchronously on the event loop, so each one must stay (near-)linear on a
# non-matching input. Avoid unbounded quantifiers that the engine has to retry
# from every start offset; TestSignatureCostIsBounded locks the worst cases down.

_INJECTION_SIGNATURES: Final[list[tuple[re.Pattern[str], str, int]]] = [
    # ── Persona / Identity Hijack (LLM01, severity: critical) ─────────────────
    (
        re.compile(
            r"you\s+are\s+now\s+(a\s+)?(different|new|evil|uncensored|unfiltered|DAN|jailbroken)",
            re.IGNORECASE,
        ),
        "persona_hijack",
        95,
    ),
    # The bare acronym is matched case-sensitively: under IGNORECASE it also hit
    # the given name, so "What did Dan say about the budget?" was hard-blocked.
    # The phrasings around it stay case-insensitive.
    (
        re.compile(r"\bDAN\b|(?i:\bdo\s+anything\s+now\b|\bdan\s+mode\b|\bas\s+dan\b)"),
        "jailbreak_dan",
        95,
    ),
    # ── Direct Instruction Override (LLM01, severity: critical) ───────────────
    (
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
            re.IGNORECASE,
        ),
        "instruction_override",
        92,
    ),
    (
        re.compile(
            r"(forget|disregard|override|bypass|neutralize)\s+(your\s+)?(system\s+)?"
            r"(prompt|instructions?|rules?|constraints?|guidelines?)",
            re.IGNORECASE,
        ),
        "instruction_override",
        92,
    ),
    # ── System Prompt Disclosure Probes (LLM07, severity: high) ───────────────
    # Covers: "show me your system prompt", "print your initial instructions",
    #         "tell me what your system context says", "reveal your context window".
    # The object must be qualified as the model's own ("your ...") or as the
    # system/initial/hidden one. With the qualifier optional, any "show me the
    # instructions for ..." or "display context about ..." question over the
    # document corpus was blocked as a disclosure probe.
    (
        re.compile(
            r"(show|print|reveal|output|repeat|tell\s+me|display|leak)\s+"
            r"(?:me\s+)?(?:what\s+)?"
            r"(?:your\s+(?:(?:system|initial|hidden|original)\s+)?"
            r"|(?:the\s+)?(?:system|initial|hidden|original)\s+)"
            r"(prompt|instructions?|message|context(?:\s+window)?)",
            re.IGNORECASE,
        ),
        "prompt_leakage",
        88,
    ),
    # ── Roleplay / Hypothetical Jailbreaks (LLM01, severity: high) ────────────
    # Covers: "pretend you are an unrestricted", "roleplay as an evil",
    #         "simulate being a rogue" — all common jailbreak phrasings.
    (
        re.compile(
            r"(pretend|act|roleplay|simulate|imagine|suppose)\s+"
            r"(?:(?:you\s+)?(?:are|you're|you\s+are)|as|being)\s+"
            r"(?:an?\s+)?(different|evil|unrestricted|unfiltered|uncensored|rogue)",
            re.IGNORECASE,
        ),
        "roleplay_jailbreak",
        85,
    ),
    # ── HTML / Script Injection (LLM02, severity: medium-high) ────────────────
    (
        re.compile(r"<\s*(script|iframe|object|embed|svg|img|link|meta)[^>]*>", re.IGNORECASE),
        "html_injection",
        78,
    ),
    # ── Template / SSTI Injection (severity: medium-high) ─────────────────────
    # Anchored at \A with an atomic group around the opening delimiter: the engine
    # commits to the FIRST "{{" and scans once for a closer. If none follows the
    # first opener, none follows a later one either, so retrying from every "{{"
    # (quadratic on "{{ {{ {{ ...") would be wasted work.
    (
        re.compile(r"\A(?>[\s\S]*?\{\{)[\s\S]*?\}\}|\A(?>[\s\S]*?\{%)[\s\S]*?%\}"),
        "template_injection",
        75,
    ),
    # ── Unicode Smuggling (severity: medium) ──────────────────────────────────
    # Three or more consecutive \uXXXX escapes indicate attempted encoding obfuscation.
    (
        re.compile(r"(?:\\u[0-9a-fA-F]{4}){3,}", re.IGNORECASE),
        "unicode_smuggling",
        65,
    ),
    # ── Repetition Flood / Token Stuffing (severity: low-medium) ──────────────
    # Repeated sequences are used to exhaust context or force model state resets.
    # The repeated unit is capped: an open-ended ``(.{5,})`` makes the engine try
    # every unit length at every offset, which is quadratic on ordinary prose
    # (about one second of CPU for a single 8 KB query). The trade-off is that a
    # unit longer than the cap is only flagged when it is itself periodic.
    (
        re.compile(r"(.{5,64})\1{9,}"),
        "repetition_flood",
        60,
    ),
]

# ── Chain-of-Thought Manipulation (LLM01, severity: high) ─────────────────────
# Adversarial prompts bury an override inside a reasoning scaffold:
#   "First, ... Then, ... Finally, ignore the rules."
# Each stage must start within its window (in characters) of the previous stage's
# end. This is evaluated procedurally in _has_cot_override_scaffold() rather than
# as one regex, because chaining windowed wildcards ("a.{0,300}b.{0,300}c...")
# backtracks combinatorially when the markers are repeated.
_COT_OPENING: Final[re.Pattern[str]] = re.compile(r"step\s+1|first,?", re.IGNORECASE)
_COT_FOLLOW_UPS: Final[list[tuple[re.Pattern[str], int]]] = [
    (re.compile(r"step\s+2|then,?", re.IGNORECASE), 300),
    (re.compile(r"step\s+3|finally,?", re.IGNORECASE), 300),
    (re.compile(r"ignore|bypass|override|disregard", re.IGNORECASE), 200),
]
_COT_SCORE: Final[int] = 85

# ── Homoglyph folding ─────────────────────────────────────────────────────────
# Cyrillic and Greek letters that render identically (or nearly so) to Latin ones.
# Unicode normalization does NOT unify these — they are distinct letters, not
# compatibility variants — so "ignоre" spelled with U+043E sails past an ASCII
# signature. The table is intentionally limited to the high-confidence lookalikes
# from the Unicode confusables data rather than the full set, which would start
# merging letters that merely resemble each other in some fonts.
_CONFUSABLES: Final[dict[int, str]] = str.maketrans(
    {
        # Cyrillic
        "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p",
        "с": "c", "т": "t", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d",
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
        "С": "C", "Т": "T", "У": "Y", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
        # Greek
        "α": "a", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t",
        "υ": "u", "χ": "x",
        "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
        "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    }
)  # fmt: skip

_MAX_QUERY_CHARS: Final[int] = 8192
_BLOCK_SCORE_THRESHOLD: Final[int] = 80  # Hard block
_SUSPICIOUS_SCORE_THRESHOLD: Final[int] = 60  # Block only in strict mode


class SecurityGateway(QueryGuardPort):
    """
    Stateless security enforcement pipeline.

    Designed as a pure service with no external I/O; inject it as a singleton.
    Rate limiting is intentionally handled upstream in the ASGI middleware layer
    to keep network-I/O concerns out of this domain service.

    Args:
        strict_mode: When True, SUSPICIOUS-level queries (score ≥ 60) are blocked
                     in addition to BLOCKED-level ones (score ≥ 80). Recommended
                     for production; disable only in sandboxed evaluation environments.
    """

    def __init__(self, strict_mode: bool = True) -> None:
        self._strict_mode = strict_mode
        # Precompute the effective threshold to avoid branch in the hot path.
        self._block_threshold = (
            _SUSPICIOUS_SCORE_THRESHOLD if strict_mode else _BLOCK_SCORE_THRESHOLD
        )

    def evaluate(self, raw_query: RawQuery) -> GatewayResult:
        """
        Full evaluation pipeline. Always returns a GatewayResult;
        callers MUST check `.blocked` before forwarding to the RAG pipeline.
        """
        full_hash = hashlib.sha256(raw_query.text.encode()).hexdigest()
        log = logger.bind(query_hash=full_hash[:16])

        # ── Stage 1: Hard structural validation ───────────────────────────────
        if len(raw_query.text) > _MAX_QUERY_CHARS:
            log.warning("security.query_rejected", reason="char_limit_exceeded")
            assessment = ThreatAssessment(
                ThreatLevel.BLOCKED, 100, ["char_limit_exceeded"], full_hash
            )
            return self._make_blocked(assessment, "Query exceeds maximum allowed length.")

        # ── Stage 2: Unicode normalization ────────────────────────────────────
        # Removes the encodings an attacker uses to split or disguise a keyword
        # without changing how the text reads: compatibility forms (fullwidth,
        # ligatures), combining sequences, and invisible format characters.
        normalized = _normalize_unicode(raw_query.text)

        # ── Stage 3: Heuristic threat scoring ─────────────────────────────────
        # Signatures are matched against a homoglyph-folded view. The fold is
        # lossy for genuine Cyrillic/Greek text, so it is used for detection only;
        # the query forwarded to retrieval keeps its original letters.
        assessment = self._score_threat(_fold_confusables(normalized), full_hash)
        log.info(
            "security.query_assessed",
            threat_level=assessment.level.name,
            score=assessment.score,
            rules=assessment.triggered_rules,
        )

        if assessment.score >= self._block_threshold:
            log.warning(
                "security.query_blocked",
                triggered_rules=assessment.triggered_rules,
                score=assessment.score,
            )
            return self._make_blocked(assessment, "Query blocked by security policy.")

        # ── Stage 4: Deep sanitization ────────────────────────────────────────
        # Runs only on queries that passed threat scoring to avoid double work.
        sanitized_text = _sanitize(normalized)
        sanitized_query = SanitizedQuery(text=sanitized_text, query_hash=full_hash)

        log.debug("security.query_approved", sanitized_length=len(sanitized_text))
        return GatewayResult(
            assessment=assessment,
            sanitized_query=sanitized_query,
            blocked=False,
        )

    @staticmethod
    def _score_threat(text: str, query_hash: str) -> ThreatAssessment:
        """
        Assigns a threat score by matching against the signature catalog
        and applying an entropy penalty for encoded-payload detection.
        Score is the maximum of all matched rules (not additive) to prevent
        low-confidence rules from unfairly penalizing legitimate queries.
        """
        max_score = 0
        triggered: list[str] = []

        for pattern, category, severity in _INJECTION_SIGNATURES:
            if pattern.search(text):
                triggered.append(category)
                if severity > max_score:
                    max_score = severity
                # Early exit: a critical-severity match cannot be superseded.
                if max_score >= 95:
                    break

        if _has_cot_override_scaffold(text):
            triggered.append("cot_manipulation")
            max_score = max(max_score, _COT_SCORE)

        entropy_score = _entropy_score(text)
        if entropy_score > 0:
            triggered.append("high_entropy_payload")
            if entropy_score > max_score:
                max_score = entropy_score

        level = ThreatLevel.CLEAN
        if max_score >= _BLOCK_SCORE_THRESHOLD:
            level = ThreatLevel.BLOCKED
        elif max_score >= _SUSPICIOUS_SCORE_THRESHOLD:
            level = ThreatLevel.SUSPICIOUS

        return ThreatAssessment(
            level=level,
            score=max_score,
            triggered_rules=triggered,
            query_hash=query_hash,
        )

    @staticmethod
    def _make_blocked(assessment: ThreatAssessment, reason: str) -> GatewayResult:
        return GatewayResult(
            assessment=assessment,
            sanitized_query=None,
            blocked=True,
            rejection_reason=reason,
        )


# ── Module-level pure helpers (no class state dependency) ─────────────────────


def _normalize_unicode(text: str) -> str:
    """
    Applies NFKC normalization and strips control and format characters.

    NFKC (rather than NFC) folds compatibility variants onto their plain form, so
    fullwidth letters or an "fi" ligature cannot hide a keyword. Category Cf covers
    the invisible characters used to split one: zero-width space and joiners, soft
    hyphen, BOM, and the bidirectional overrides behind "Trojan Source" style
    reordering. Category Cc is the C0/C1 control range.

    Standard whitespace (\\n, \\t, \\r) is preserved; it is needed for code context.
    """
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        ch for ch in normalized if unicodedata.category(ch) not in ("Cc", "Cf") or ch in "\n\t\r"
    )


def _fold_confusables(text: str) -> str:
    """Maps Cyrillic/Greek lookalikes onto Latin letters. For signature matching only."""
    return text.translate(_CONFUSABLES)


def _sanitize(text: str) -> str:
    """
    Strips residual HTML tags and template expressions from gateway-approved text.
    This is a last-resort measure; primary detection happens in _score_threat().
    """
    # Remove HTML/XML tags — indirect injection often arrives embedded in documents.
    text = strip_html_tags(text)
    # Redact template markers to prevent SSTI if the LLM output is later rendered.
    text = re.sub(r"\{\{[\s\S]{0,200}?\}\}|\{%[\s\S]{0,200}?%\}", "[REMOVED]", text)
    # Collapse pathological whitespace without destroying semantic structure.
    text = re.sub(r"[ \t]{4,}", "   ", text)
    return text.strip()


def _has_cot_override_scaffold(text: str) -> bool:
    """
    Detects a staged reasoning scaffold that ends in an override verb.

    Walks the stages left to right, keeping only the end offsets of matches that
    are reachable from the previous stage. Both lists are position-ordered, so a
    two-pointer sweep suffices and the whole check is linear in the input length.
    """
    reachable_ends = [m.end() for m in _COT_OPENING.finditer(text)]

    for pattern, window in _COT_FOLLOW_UPS:
        if not reachable_ends:
            return False
        next_ends: list[int] = []
        cursor = 0
        for match in pattern.finditer(text):
            start = match.start()
            # Advance to the closest preceding end; it yields the smallest gap.
            while cursor + 1 < len(reachable_ends) and reachable_ends[cursor + 1] <= start:
                cursor += 1
            if 0 <= start - reachable_ends[cursor] <= window:
                next_ends.append(match.end())
        reachable_ends = next_ends

    return bool(reachable_ends)


def _entropy_score(text: str) -> int:
    """
    Returns a threat score based on Shannon entropy of the input's ASCII characters.
    Payloads with entropy > 4.5 bits/char are characteristic of base64 or hex
    encoded content commonly used to bypass signature-based filters.
    Returns 0 for short inputs where entropy is not a reliable signal.

    Only ASCII is measured. Character entropy depends on the size of the alphabet:
    ordinary Japanese or Chinese prose sits well above 5.2 bits/char, so measuring
    every character blocked any long question in those languages as an "encoded
    payload". The encodings this check exists for are ASCII by definition, and
    because other characters are skipped rather than exempting the query, padding
    a payload with them does not hide it.
    """
    ascii_chars = [ch for ch in text if ch.isascii()]
    length = len(ascii_chars)
    if length < 80:
        return 0

    freq: dict[str, int] = {}
    for ch in ascii_chars:
        freq[ch] = freq.get(ch, 0) + 1

    entropy = -sum((count / length) * math.log2(count / length) for count in freq.values())

    if entropy > 5.2:
        return 70
    if entropy > 4.5:
        return 52
    return 0
