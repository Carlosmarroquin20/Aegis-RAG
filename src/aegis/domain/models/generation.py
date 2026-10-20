"""
The result of one LLM generation, as the application layer sees it.

An answer alone is not enough to run the service: the tokens it consumed are
what a hosted provider bills for, and the model that produced it is what a
quality regression or a cost spike has to be traced back to. Each adapter
translates its provider's response into this shape, so the use case can report
usage without knowing which backend answered.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    # Stable identifiers for dashboards: the provider is a host or backend name
    # ("ollama", "api.groq.com"), the model the name it was requested under.
    provider: str
    model: str
    # None when the backend did not report usage. Callers count that case
    # instead of guessing, so missing data shows up rather than reading as zero.
    usage: TokenUsage | None = None
