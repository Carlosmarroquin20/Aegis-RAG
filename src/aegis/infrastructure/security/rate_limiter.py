"""
Sliding-window rate limiter with per-client isolation.

Architecture:
  - RateLimitStore is a port; swap InMemoryRateLimitStore for RedisRateLimitStore
    in multi-worker deployments without changing any calling code.
  - InMemoryRateLimitStore is deliberately NOT thread-safe; use the Redis backend
    for uvicorn multi-worker or Gunicorn deployments.
  - The sliding window algorithm avoids the "thundering herd" problem inherent
    in fixed-window counters: bursts at window boundaries are naturally dampened.
  - Timestamps are wall-clock (time.time()), not monotonic, so a shared backend
    can compare timestamps recorded by different processes/hosts.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from math import ceil
from typing import Any, ClassVar
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)

# Upper bound for one round trip to Redis. The limiter sits on the request path
# of every call, so this is also the most a degraded Redis can add to latency.
_REDIS_TIMEOUT_SECONDS: float = 0.5


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """Immutable policy descriptor. Instantiate once and inject into RateLimiter."""

    requests_per_window: int
    window_seconds: int
    burst_allowance: int = 0  # Permits short spikes above the baseline rate.


@dataclass(frozen=True, slots=True)
class WindowState:
    """
    What the limiter needs to know about a client's sliding window.

    Only the size and the oldest entry: the decision depends on the count, and
    the window resets when its oldest entry expires. Stores report this summary
    rather than every timestamp, so the cost of a check does not grow with the
    limit. A key allowed 10,000 requests a minute would otherwise transfer and
    parse up to 10,000 entries on each of its requests.
    """

    count: int
    oldest: float | None  # wall-clock timestamp; None when the window is empty


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    allowed: bool
    remaining: int  # Requests left in the current window.
    reset_at: float  # Unix wall-clock timestamp when the window resets.
    retry_after: float | None = None  # Seconds until the next allowed request.


class RateLimitStore(ABC):
    """
    Port for the rate-limit persistence backend.
    Implementations must be safe for the concurrency model they are deployed in.
    """

    #: True when calls wait on the network. Callers on an event loop must then
    #: run them in a worker thread: a store that stalls would otherwise stall
    #: every request in the process, not only the ones it is asked about.
    #: In-process stores leave it False and are called inline, which is also
    #: what keeps a store that is not thread-safe on a single thread.
    performs_io: ClassVar[bool] = False

    @abstractmethod
    def record_and_get(
        self, key: str, window_seconds: int, max_entries: int | None = None
    ) -> WindowState:
        """
        Atomically records a new request timestamp and returns the state of the
        active sliding window, including the request just recorded.

        ``max_entries`` bounds what the store RETAINS, not what it reports: if the
        new request takes the window past it, the request is still counted in the
        returned state (so the caller can see the limit was exceeded) but is not
        kept. Without that bound a throttled client that keeps retrying would
        grow the window with every rejected attempt — unbounded memory, a
        full-window read on each request, and a window that never drains.
        """

    @abstractmethod
    def peek(self, key: str, window_seconds: int) -> WindowState:
        """Returns the current window state without recording a new request."""


class InMemoryRateLimitStore(RateLimitStore):
    """
    In-process store for single-worker deployments and testing.

    NOT suitable for multi-worker deployments because each worker maintains
    its own bucket, causing under-counting. Use RedisRateLimitStore there.
    """

    def __init__(self) -> None:
        # defaultdict avoids the key-existence check on every request.
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def record_and_get(
        self, key: str, window_seconds: int, max_entries: int | None = None
    ) -> WindowState:
        now = time.time()
        bucket = self._buckets[key]
        self._evict_expired(bucket, now - window_seconds)
        bucket.append(now)
        state = WindowState(count=len(bucket), oldest=bucket[0])
        if max_entries is not None and state.count > max_entries:
            bucket.pop()
        return state

    def peek(self, key: str, window_seconds: int) -> WindowState:
        now = time.time()
        bucket = self._buckets[key]
        self._evict_expired(bucket, now - window_seconds)
        return WindowState(count=len(bucket), oldest=bucket[0] if bucket else None)

    @staticmethod
    def _evict_expired(bucket: deque[float], cutoff: float) -> None:
        while bucket and bucket[0] < cutoff:
            bucket.popleft()


class RedisRateLimitStore(RateLimitStore):
    """
    Shared, multi-worker sliding-window store backed by a Redis sorted set.

    Each client maps to a ZSET whose members are unique request tokens scored by
    their wall-clock timestamp. This is the correct backend for any deployment
    with more than one process (uvicorn ``--workers N``, Gunicorn, several
    replicas): every worker reads and writes the same window, so counts are not
    fragmented the way they are with the in-memory store.

    Atomicity: the record path runs as a single MULTI/EXEC transaction — evict
    expired members, add the new one, refresh the key TTL, and read the window's
    size and oldest member back — so concurrent requests cannot interleave and
    under-count. ZCARD and a one-element ZRANGE are O(1) and O(log N): what a
    check costs does not depend on how many requests the window holds.

    Calls block on the network, so ``performs_io`` is True and the middleware
    runs them in a worker thread.

    The ``redis`` package is an optional dependency; install it with the extra:
    ``pip install "aegis-rag[redis]"``. Import is deferred so the module loads
    even when Redis is not installed and the in-memory store is in use.
    """

    performs_io: ClassVar[bool] = True

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        key_prefix: str = "aegis:ratelimit:",
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError(
                    "RedisRateLimitStore requires the 'redis' package. "
                    'Install it with: pip install "aegis-rag[redis]"'
                ) from exc
            # Bounded socket timeouts: redis-py waits forever by default, so a
            # Redis that accepts connections but stops answering would otherwise
            # hang every request instead of failing fast into the caller's
            # fail-open / fail-closed policy.
            self._client = redis.Redis.from_url(
                redis_url,
                socket_connect_timeout=_REDIS_TIMEOUT_SECONDS,
                socket_timeout=_REDIS_TIMEOUT_SECONDS,
            )
        self._key_prefix = key_prefix

    def record_and_get(
        self, key: str, window_seconds: int, max_entries: int | None = None
    ) -> WindowState:
        now = time.time()
        redis_key = self._key_prefix + key
        cutoff = now - window_seconds
        # Unique member so simultaneous requests with an identical score coexist.
        member = f"{now:.6f}:{uuid4().hex}"

        pipe = self._client.pipeline(transaction=True)
        pipe.zremrangebyscore(redis_key, "-inf", cutoff)
        pipe.zadd(redis_key, {member: now})
        pipe.expire(redis_key, ceil(window_seconds))
        pipe.zcard(redis_key)
        pipe.zrange(redis_key, 0, 0, withscores=True)
        *_, count, oldest = pipe.execute()
        state = _window_state(count, oldest)

        if max_entries is not None and state.count > max_entries:
            # Outside the transaction on purpose. The member is unique to this
            # request, so removing it late can only make a concurrent request see
            # one extra entry in a window that is already over the limit — it
            # would have been rejected anyway.
            self._client.zrem(redis_key, member)
        return state

    def peek(self, key: str, window_seconds: int) -> WindowState:
        now = time.time()
        redis_key = self._key_prefix + key
        cutoff = now - window_seconds
        # Read-only: do not evict or record, just measure the live window.
        pipe = self._client.pipeline(transaction=True)
        pipe.zcount(redis_key, cutoff, "+inf")
        pipe.zrangebyscore(redis_key, cutoff, "+inf", start=0, num=1, withscores=True)
        count, oldest = pipe.execute()
        return _window_state(count, oldest)


def _window_state(count: int, oldest: list[tuple[bytes, float]]) -> WindowState:
    """Builds the state from a count and a ZRANGE ... WITHSCORES reply of at most one member."""
    return WindowState(count=int(count), oldest=float(oldest[0][1]) if oldest else None)


class RateLimiter:
    """
    Sliding-window rate limiter.

    Usage:
        limiter = RateLimiter(policy=RateLimitPolicy(60, 60, burst_allowance=10))
        result = limiter.check_and_record(client_id)
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

    @property
    def limit(self) -> int:
        """Requests admitted per window, burst allowance included."""
        return self._max_requests

    @property
    def performs_io(self) -> bool:
        """Whether checks wait on the network; see ``RateLimitStore.performs_io``."""
        return self._store.performs_io

    def _reset_at(self, window: WindowState) -> float:
        """When the window frees a slot: as its oldest entry expires."""
        if window.oldest is None:
            return time.time()
        return window.oldest + self._policy.window_seconds

    def check_and_record(self, client_id: str) -> RateLimitResult:
        """
        Records the request and returns whether it is permitted.

        ``client_id`` names the bucket and is written to logs and to the store's
        keyspace, so pass an opaque identifier, never a credential.

        The request is counted before the decision is made ("count then check"),
        which prevents under-counting when requests race. A request that turns out
        to be over the limit is not retained by the store, so only admitted
        requests occupy the window: the reported reset time is then accurate, and
        a client that waits ``retry_after`` is admitted on its next attempt.
        Callers must reject if ``result.allowed`` is False.
        """
        window = self._store.record_and_get(
            client_id, self._policy.window_seconds, max_entries=self._max_requests
        )
        count = window.count

        reset_at = self._reset_at(window)

        if count > self._max_requests:
            retry_after = max(0.0, reset_at - time.time())
            logger.warning(
                "rate_limit.exceeded",
                client_id=client_id,
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

    def peek(self, client_id: str) -> RateLimitResult:
        """Non-mutating check — useful for pre-flight validation without side effects."""
        window = self._store.peek(client_id, self._policy.window_seconds)
        count = window.count
        reset_at = self._reset_at(window)
        remaining = max(0, self._max_requests - count)
        allowed = count < self._max_requests
        return RateLimitResult(allowed=allowed, remaining=remaining, reset_at=reset_at)
