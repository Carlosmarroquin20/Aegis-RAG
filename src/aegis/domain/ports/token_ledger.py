"""
Token ledger port: how many LLM tokens a tenant has used today.

A shared deployment's real cost is its LLM tokens, and on a hosted provider
one team's runaway script is everyone's bill. A daily budget per tenant caps
that. The ledger is where the budget is checked and charged; it is a port so
a deployment with several replicas can keep one count for all of them.

The day is the UTC calendar day for every adapter, so a budget resets at the
same moment wherever the replicas run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DailyUsage:
    tokens: int
    seconds_to_reset: float  # until the next UTC midnight


class TokenBudgetExhaustedError(Exception):
    """The tenant has used its daily token budget; the query is refused before any cost."""

    def __init__(self, tenant_id: str, budget: int, retry_after: float) -> None:
        super().__init__(f"tenant {tenant_id!r} has used its daily budget of {budget} tokens")
        self.tenant_id = tenant_id
        self.budget = budget
        self.retry_after = retry_after


class TokenLedgerPort(ABC):
    @abstractmethod
    async def used_today(self, tenant_id: str) -> DailyUsage:
        """Tokens charged to ``tenant_id`` since the last UTC midnight."""

    @abstractmethod
    async def charge(self, tenant_id: str, tokens: int) -> None:
        """Adds ``tokens`` to today's count for ``tenant_id``."""
