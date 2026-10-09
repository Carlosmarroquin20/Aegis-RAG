"""
Security ports (abstract interfaces) for the query pipeline.

The use case is required to run an input guard before any I/O and an output
guard before returning anything, but it must not depend on how either one
decides. Expressing them as ports keeps the application layer free of imports
from the infrastructure layer, and lets a different control (a classifier model,
a policy engine, an external moderation service) replace the regex-based
implementations without touching the orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aegis.domain.models.query import RawQuery
from aegis.domain.models.security import GatewayResult, SanitizedOutput


class OutputReflectionError(Exception):
    """
    Raised when the LLM output exhibits prompt injection reflection.
    The API layer should return HTTP 500 (not 400) because this represents
    a pipeline failure, not a user input error.
    """


class QueryGuardPort(ABC):
    """Evaluates untrusted query text before it may reach retrieval or the LLM."""

    @abstractmethod
    def evaluate(self, raw_query: RawQuery) -> GatewayResult:
        """
        Always returns a result; callers MUST check ``.blocked`` before using
        ``.sanitized_query``. Implementations must not perform I/O: this runs on
        every request ahead of everything else.
        """


class OutputGuardPort(ABC):
    """Post-processes an LLM response before it is returned to the caller."""

    @abstractmethod
    def sanitize(self, raw_output: str, query_hash: str = "") -> SanitizedOutput:
        """
        Returns the response in a form that is safe to serialize.

        Raises:
            OutputReflectionError: the response echoes an injection payload and
                                   the guard is configured to block on it.
        """
