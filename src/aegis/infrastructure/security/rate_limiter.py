"""
Sliding-window rate limiter with per-API-key isolation.

Architecture:
  - RateLimitStore is a port; swap InMemoryRateLimitStore for RedisRateLimitStore
    in multi-worker deployments without changing any calling code.
  - InMemoryRateLimitStore is deliberately NOT thread-safe; use the Redis backend
    for uvicorn multi-worker or Gunicorn deployments.
  - The sliding window algorithm avoids the "thundering herd" problem inherent
    in fixed-window counters: bursts at window boundaries are naturally dampened.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """Immutable policy descriptor. Instantiate once and inject into RateLimiter."""

    requests_per_window: int
    window_seconds: int
    burst_allowance: int = 0  # Permits short spikes above the baseline rate.


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    allowed: bool
    remaining: int  # Requests left in the current window.
    reset_at: float  # Unix monotonic timestamp when the window resets.
    retry_after: float | None = None  # Seconds until the next allowed request.


class RateLimitStore(ABC):
    """
    Port for the rate-limit persistence backend.
    Implementations must be safe for the concurrency model they are deployed in.
    """

    @abstractmethod
    def record_and_get(self, key: str, window_seconds: int) -> list[float]:
        """
        Atomically records a new request timestamp and returns all timestamps
        within the active sliding window, including the one just recorded.
        """

    @abstractmethod
    def peek(self, key: str, window_seconds: int) -> list[float]:
        """Returns current window timestamps without recording a new request."""


class InMemoryRateLimitStore(RateLimitStore):
    """
    In-process store for single-worker deployments and testing.

    NOT suitable for multi-worker deployments because each worker maintains
    its own bucket, causing under-counting. Use RedisRateLimitStore there.
    """

    def __init__(self) -> None:
        # defaultdict avoids the key-existence check on every request.
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def record_and_get(self, key: str, window_seconds: int) -> list[float]:
        now = time.monotonic()
        bucket = self._buckets[key]
        self._evict_expired(bucket, now - window_seconds)
        bucket.append(now)
        return list(bucket)

    def peek(self, key: str, window_seconds: int) -> list[float]:
        now = time.monotonic()
        bucket = self._buckets[key]
        self._evict_expired(bucket, now - window_seconds)
        return list(bucket)

    @staticmethod
    def _evict_expired(bucket: deque[float], cutoff: float) -> None:
        while bucket and bucket[0] < cutoff:
            bucket.popleft()


class RateLimiter:
    """
    Sliding-window rate limiter.

    Usage:
        limiter = RateLimiter(policy=RateLimitPolicy(60, 60, burst_allowance=10))
        result = limiter.check_and_record(api_key)
        if not result.allowed:
            raise HTTPException(429, headers={"Retry-After": str(result.retry_after)})
    """

    def __init__(
        self,
        policy: RateLimitPolicy,
        store: RateLimitStore | None = None,
    ) -> None:
        self._policy = policy
        self._store = store or InMemoryRateLimitStore()
        self._max_requests = policy.requests_per_window + policy.burst_allowance

    def check_and_record(self, api_key: str) -> RateLimitResult:
        """
        Records the request and returns whether it is permitted.
        The request is always recorded; callers must reject if result.allowed is False.
        This "count then check" approach prevents under-counting under race conditions.
        """
        timestamps = self._store.record_and_get(api_key, self._policy.window_seconds)
        count = len(timestamps)

        reset_at = timestamps[0] + self._policy.window_seconds if timestamps else time.monotonic()

        if count > self._max_requests:
            retry_after = max(0.0, reset_at - time.monotonic())
            logger.warning(
                "rate_limit.exceeded",
                api_key_prefix=api_key[:8],
                request_count=count,
                max_allowed=self._max_requests,
            )
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=reset_at,
                retry_after=retry_after,
            )

        return RateLimitResult(
            allowed=True,
            remaining=max(0, self._max_requests - count),
            reset_at=reset_at,
        )

    def peek(self, api_key: str) -> RateLimitResult:
        """Non-mutating check — useful for pre-flight validation without side effects."""
        timestamps = self._store.peek(api_key, self._policy.window_seconds)
        count = len(timestamps)
        reset_at = timestamps[0] + self._policy.window_seconds if timestamps else time.monotonic()
        remaining = max(0, self._max_requests - count)
        allowed = count < self._max_requests
        return RateLimitResult(allowed=allowed, remaining=remaining, reset_at=reset_at)
