"""
Security Gateway — OWASP LLM Top 10 enforcement layer.

Every query passes through this pipeline before touching the RAG engine:
  1. Structural validation (length, null bytes)
  2. Unicode normalization (defeats homoglyph and encoding attacks)
  3. Heuristic threat scoring (injection signatures + entropy analysis)
  4. Deep sanitization (HTML/template stripping)

The gateway is intentionally I/O-free: no network calls, no DB access.
This keeps it fast (<1 ms for typical queries) and easily testable.

OWASP LLM coverage:
  LLM01 — Prompt Injection           → _INJECTION_SIGNATURES, _normalize_unicode
  LLM02 — Insecure Output Handling   → enforced at the LLM adapter layer (see llm_client.py)
  LLM06 — Sensitive Information      → rate limiter + API key enforcement (middleware)
  LLM07 — System Prompt Disclosure   → _INJECTION_SIGNATURES (prompt_leakage category)
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final

import structlog

from aegis.domain.models.query import RawQuery, SanitizedQuery

logger = structlog.get_logger(__name__)

# ── Threat Signature Catalog ──────────────────────────────────────────────────
#
# Each entry: (compiled_pattern, threat_category, severity_score 0-100)
# Scores are NOT additive — the maximum matched score is used to avoid
# penalizing legitimate queries that happen to match low-confidence rules.
# Patterns are ordered by severity (highest first) for early-exit potential.

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
    (
        re.compile(r"\bDAN\b|\bdo\s+anything\s+now\b", re.IGNORECASE),
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
    (
        re.compile(
            r"(show|print|reveal|output|repeat|tell\s+me|display|leak)\s+(your\s+)?"
            r"(system\s+)?(prompt|instructions?|initial\s+message|context\s+window)",
            re.IGNORECASE,
        ),
        "prompt_leakage",
        88,
    ),
    # ── Roleplay / Hypothetical Jailbreaks (LLM01, severity: high) ────────────
    (
        re.compile(
            r"(pretend|act|roleplay|simulate|imagine|suppose)\s+(you\s+)?(are|you're|you\s+are)\s+"
            r"(a\s+)?(different|evil|unrestricted|unfiltered|uncensored|rogue)",
            re.IGNORECASE,
        ),
        "roleplay_jailbreak",
        85,
    ),
    # ── Chain-of-Thought Manipulation (LLM01, severity: high) ─────────────────
    # Targets adversarial prompts that bury an override inside a reasoning scaffold.
    (
        re.compile(
            r"(step\s+1|first,?)[\s\S]{0,300}(step\s+2|then,?)[\s\S]{0,300}"
            r"(step\s+3|finally,?)[\s\S]{0,200}(ignore|bypass|override|disregard)",
            re.IGNORECASE | re.DOTALL,
        ),
        "cot_manipulation",
        85,
    ),
    # ── HTML / Script Injection (LLM02, severity: medium-high) ────────────────
    (
        re.compile(r"<\s*(script|iframe|object|embed|svg|img|link|meta)[^>]*>", re.IGNORECASE),
        "html_injection",
        78,
    ),
    # ── Template / SSTI Injection (severity: medium-high) ─────────────────────
    (
        re.compile(r"\{\{[\s\S]*?\}\}|\{%[\s\S]*?%\}", re.DOTALL),
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
    (
        re.compile(r"(.{5,})\1{9,}"),
        "repetition_flood",
        60,
    ),
]

_MAX_QUERY_CHARS: Final[int] = 8192
_BLOCK_SCORE_THRESHOLD: Final[int] = 80       # Hard block
_SUSPICIOUS_SCORE_THRESHOLD: Final[int] = 60  # Block only in strict mode


class ThreatLevel(IntEnum):
    CLEAN = 0
    SUSPICIOUS = 1
    BLOCKED = 2


@dataclass(frozen=True, slots=True)
class ThreatAssessment:
    level: ThreatLevel
    score: int  # 0–100; highest matched rule score
    triggered_rules: list[str] = field(default_factory=list)
    # First 16 hex chars of SHA-256 are sufficient for log correlation without
    # exposing the full query content in log aggregators.
    query_hash: str = ""

    @property
    def is_blocked(self) -> bool:
        return self.level == ThreatLevel.BLOCKED


@dataclass(frozen=True, slots=True)
class GatewayResult:
    assessment: ThreatAssessment
    sanitized_query: SanitizedQuery | None
    blocked: bool
    rejection_reason: str | None = None


class SecurityGateway:
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
        # NFC collapses homoglyphs (e.g., Cyrillic 'а' → Latin 'a') and
        # combining diacritics used to split injection keywords across code points.
        normalized = _normalize_unicode(raw_query.text)

        # ── Stage 3: Heuristic threat scoring ─────────────────────────────────
        assessment = self._score_threat(normalized, full_hash)
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
    Applies NFC normalization and strips C0/C1 control characters.
    Preserves standard whitespace (\\n, \\t, \\r) which is needed for code context.
    """
    normalized = unicodedata.normalize("NFC", text)
    return "".join(
        ch
        for ch in normalized
        if unicodedata.category(ch) != "Cc" or ch in "\n\t\r"
    )


def _sanitize(text: str) -> str:
    """
    Strips residual HTML tags and template expressions from gateway-approved text.
    This is a last-resort measure; primary detection happens in _score_threat().
    """
    # Remove HTML/XML tags — indirect injection often arrives embedded in documents.
    text = re.sub(r"<[^>]{0,500}>", "", text)
    # Redact template markers to prevent SSTI if the LLM output is later rendered.
    text = re.sub(r"\{\{[\s\S]{0,200}?\}\}|\{%[\s\S]{0,200}?%\}", "[REMOVED]", text)
    # Collapse pathological whitespace without destroying semantic structure.
    text = re.sub(r"[ \t]{4,}", "   ", text)
    return text.strip()


def _entropy_score(text: str) -> int:
    """
    Returns a threat score based on Shannon entropy of the input characters.
    Payloads with entropy > 4.5 bits/char are characteristic of base64 or hex
    encoded content commonly used to bypass signature-based filters.
    Returns 0 for short inputs where entropy is not a reliable signal.
    """
    if len(text) < 80:
        return 0

    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1

    length = len(text)
    entropy = -sum(
        (count / length) * math.log2(count / length) for count in freq.values()
    )

    if entropy > 5.2:
        return 70
    if entropy > 4.5:
        return 52
    return 0
