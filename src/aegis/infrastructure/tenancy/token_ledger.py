"""
Daily token ledgers: in-process for one worker, Redis for several.

The Redis ledger keeps one counter per tenant per UTC day,
``aegis:tokens:<tenant>:<YYYY-MM-DD>``, incremented atomically with INCRBY.
Each counter expires two days after it is first written, so old days clean
themselves up and a clock that is slightly off between replicas still finds
yesterday's key. The client is synchronous, so every call runs in a worker
thread, for the same reason as the rate-limit store: a stalled Redis must not
stall the event loop.

Enforcement is a soft limit. The check happens before generation and the charge
after it, so concurrent requests near the limit can each pass the check: a
tenant can overshoot its budget by at most what its in-flight requests consume.
Reserving tokens up front would need an estimate of an answer's length before
it exists, and a wrong one would refuse queries that fit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from aegis.domain.ports.token_ledger import DailyUsage, TokenLedgerPort

_KEY_PREFIX = "aegis:tokens:"
_KEY_TTL_SECONDS = 2 * 24 * 3600
_REDIS_TIMEOUT_SECONDS = 0.5


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _seconds_to_midnight(now: datetime) -> float:
    midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    return (midnight - now).total_seconds()


class InMemoryTokenLedger(TokenLedgerPort):
    """Per-process counts: correct for one worker, and for tests."""

    def __init__(self, clock: Callable[[], datetime] = _utc_now) -> None:
        self._clock = clock
        self._counts: dict[tuple[str, date], int] = {}

    async def used_today(self, tenant_id: str) -> DailyUsage:
        now = self._clock()
        return DailyUsage(
            tokens=self._counts.get((tenant_id, now.date()), 0),
            seconds_to_reset=_seconds_to_midnight(now),
        )

    async def charge(self, tenant_id: str, tokens: int) -> None:
        key = (tenant_id, self._clock().date())
        self._counts[key] = self._counts.get(key, 0) + tokens


class RedisTokenLedger(TokenLedgerPort):
    """One count per tenant per day, shared by every worker and replica."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        client: Any | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if client is None:
            import redis  # optional extra, imported only when this backend is used

            client = redis.Redis.from_url(
                redis_url,
                socket_connect_timeout=_REDIS_TIMEOUT_SECONDS,
                socket_timeout=_REDIS_TIMEOUT_SECONDS,
            )
        self._client = client
        self._clock = clock

    def _key(self, tenant_id: str, day: date) -> str:
        return f"{_KEY_PREFIX}{tenant_id}:{day.isoformat()}"

    async def used_today(self, tenant_id: str) -> DailyUsage:
        now = self._clock()
        raw = await asyncio.to_thread(self._client.get, self._key(tenant_id, now.date()))
        return DailyUsage(tokens=int(raw or 0), seconds_to_reset=_seconds_to_midnight(now))

    async def charge(self, tenant_id: str, tokens: int) -> None:
        key = self._key(tenant_id, self._clock().date())
        await asyncio.to_thread(self._charge, key, tokens)

    def _charge(self, key: str, tokens: int) -> None:
        pipe = self._client.pipeline(transaction=True)
        pipe.incrby(key, tokens)
        # NX: set once, when the day's key is created, so later charges do not
        # keep pushing the expiry out.
        pipe.expire(key, _KEY_TTL_SECONDS, nx=True)
        pipe.execute()
