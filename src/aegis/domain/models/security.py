"""
Domain models for security verdicts.

These are the values that cross the boundary between the security controls and
the use case that orchestrates them: what the input guard concluded about a
query, and what the output guard did to a response. They are plain, immutable
data with no behaviour tied to a detection technique, so the application layer
can reason about a verdict without knowing how it was reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from aegis.domain.models.query import SanitizedQuery


class ThreatLevel(IntEnum):
    CLEAN = 0
    SUSPICIOUS = 1
    BLOCKED = 2


@dataclass(frozen=True, slots=True)
class ThreatAssessment:
    level: ThreatLevel
    score: int  # 0–100; highest matched rule score
    triggered_rules: list[str] = field(default_factory=list)
    # Full SHA-256 of the query. Logs and API responses carry only a prefix:
    # enough for correlation without making the hash a lookup key for content.
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


@dataclass(frozen=True, slots=True)
class SanitizedOutput:
    text: str
    was_truncated: bool
    reflection_detected: bool
    pii_types_detected: list[str]

    @property
    def has_warnings(self) -> bool:
        return self.reflection_detected or bool(self.pii_types_detected)
