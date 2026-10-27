"""
Unit tests for the sliding-window RateLimiter and InMemoryRateLimitStore.

All tests run without I/O or time mocking: a large window (3600 s) ensures
no timestamps expire mid-test, making assertions deterministic.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import fakeredis
import pytest

from aegis.infrastructure.security.rate_limiter import (
    InMemoryRateLimitStore,
    RateLimiter,
    RateLimitPolicy,
    RateLimitStore,
    RedisRateLimitStore,
)

_WINDOW = 3600  # 1 hour — timestamps never expire within a single test run


# ── RateLimitPolicy ────────────────────────────────────────────────────────────


class TestRateLimitPolicy:
    def test_fields_are_stored(self) -> None:
        policy = RateLimitPolicy(requests_per_window=10, window_seconds=60)
        assert policy.requests_per_window == 10
        assert policy.window_seconds == 60
        assert policy.burst_allowance == 0

    def test_burst_allowance_stored(self) -> None:
        policy = RateLimitPolicy(requests_per_window=10, window_seconds=60, burst_allowance=5)
        assert policy.burst_allowance == 5

    def test_policy_is_immutable(self) -> None:
        policy = RateLimitPolicy(requests_per_window=10, window_seconds=60)
        with pytest.raises(FrozenInstanceError):  # frozen dataclass forbids mutation
            policy.requests_per_window = 99  # type: ignore[misc]


# ── InMemoryRateLimitStore ─────────────────────────────────────────────────────


class TestInMemoryRateLimitStore:
    def test_record_returns_one_timestamp_on_first_call(self) -> None:
        store = InMemoryRateLimitStore()
        result = store.record_and_get("key", _WINDOW)
        assert result.count == 1

    def test_second_record_returns_two_timestamps(self) -> None:
        store = InMemoryRateLimitStore()
        store.record_and_get("key", _WINDOW)
        result = store.record_and_get("key", _WINDOW)
        assert result.count == 2

    def test_peek_does_not_add_timestamps(self) -> None:
        store = InMemoryRateLimitStore()
        store.record_and_get("key", _WINDOW)
        assert store.peek("key", _WINDOW).count == 1
        assert store.peek("key", _WINDOW).count == 1  # still 1 after two peeks

    def test_keys_are_isolated(self) -> None:
        store = InMemoryRateLimitStore()
        store.record_and_get("alice", _WINDOW)
        store.record_and_get("alice", _WINDOW)
        store.record_and_get("bob", _WINDOW)
        assert store.peek("alice", _WINDOW).count == 2
        assert store.peek("bob", _WINDOW).count == 1

    def test_unknown_key_peek_returns_empty(self) -> None:
        store = InMemoryRateLimitStore()
        assert store.peek("ghost", _WINDOW).count == 0


# ── RateLimiter ────────────────────────────────────────────────────────────────


class TestRateLimiterAllowed:
    def test_first_request_is_allowed(self) -> None:
        limiter = RateLimiter(RateLimitPolicy(5, _WINDOW))
        result = limiter.check_and_record("user")
        assert result.allowed

    def test_remaining_decrements_per_request(self) -> None:
        limiter = RateLimiter(RateLimitPolicy(5, _WINDOW))
        limiter.check_and_record("user")
        result = limiter.check_and_record("user")
        assert result.remaining == 3  # 5 - 2 = 3

    def test_requests_up_to_limit_are_all_allowed(self) -> None:
        limit = 4
        limiter = RateLimiter(RateLimitPolicy(limit, _WINDOW))
        results = [limiter.check_and_record("user") for _ in range(limit)]
        assert all(r.allowed for r in results)

    def test_keys_have_independent_quotas(self) -> None:
        limiter = RateLimiter(RateLimitPolicy(2, _WINDOW))
        limiter.check_and_record("alice")
        limiter.check_and_record("alice")
        # alice exhausted; bob still has quota
        assert not limiter.check_and_record("alice").allowed
        assert limiter.check_and_record("bob").allowed


class TestRateLimiterBlocked:
    def test_request_over_limit_is_blocked(self) -> None:
        limiter = RateLimiter(RateLimitPolicy(3, _WINDOW))
        for _ in range(3):
            limiter.check_and_record("user")
        result = limiter.check_and_record("user")
        assert not result.allowed

    def test_blocked_result_has_zero_remaining(self) -> None:
        limiter = RateLimiter(RateLimitPolicy(2, _WINDOW))
        for _ in range(2):
            limiter.check_and_record("user")
        result = limiter.check_and_record("user")
        assert result.remaining == 0

    def test_blocked_result_has_retry_after(self) -> None:
        limiter = RateLimiter(RateLimitPolicy(1, _WINDOW))
        limiter.check_and_record("user")
        result = limiter.check_and_record("user")
        assert result.retry_after is not None
        assert result.retry_after >= 0


class TestRateLimiterBurst:
    def test_burst_allowance_extends_effective_limit(self) -> None:
        # 3 base + 2 burst = 5 effective requests before blocking
        limiter = RateLimiter(RateLimitPolicy(3, _WINDOW, burst_allowance=2))
        results = [limiter.check_and_record("user") for _ in range(5)]
        assert all(r.allowed for r in results)

    def test_request_beyond_burst_is_blocked(self) -> None:
        limiter = RateLimiter(RateLimitPolicy(3, _WINDOW, burst_allowance=2))
        for _ in range(5):
            limiter.check_and_record("user")
        assert not limiter.check_and_record("user").allowed


class TestRateLimiterPeek:
    def test_peek_does_not_consume_quota(self) -> None:
        limiter = RateLimiter(RateLimitPolicy(3, _WINDOW))
        limiter.check_and_record("user")
        before = limiter.peek("user").remaining
        limiter.peek("user")
        after = limiter.peek("user").remaining
        assert before == after

    def test_peek_reports_correct_remaining(self) -> None:
        limiter = RateLimiter(RateLimitPolicy(5, _WINDOW))
        limiter.check_and_record("user")
        limiter.check_and_record("user")
        assert limiter.peek("user").remaining == 3

    def test_peek_on_fresh_key_shows_full_quota(self) -> None:
        limiter = RateLimiter(RateLimitPolicy(10, _WINDOW))
        result = limiter.peek("new_user")
        assert result.remaining == 10
        assert result.allowed


# ── RedisRateLimitStore (fakeredis) ──────────────────────────────────────────


@pytest.fixture()
def redis_store() -> RedisRateLimitStore:
    """A Redis-backed store wired to an in-memory fakeredis client."""
    return RedisRateLimitStore(client=fakeredis.FakeStrictRedis())


class TestRedisRateLimitStore:
    def test_record_returns_one_timestamp_on_first_call(
        self, redis_store: RedisRateLimitStore
    ) -> None:
        assert redis_store.record_and_get("key", _WINDOW).count == 1

    def test_second_record_returns_two_timestamps(self, redis_store: RedisRateLimitStore) -> None:
        redis_store.record_and_get("key", _WINDOW)
        assert redis_store.record_and_get("key", _WINDOW).count == 2

    def test_peek_does_not_add_timestamps(self, redis_store: RedisRateLimitStore) -> None:
        redis_store.record_and_get("key", _WINDOW)
        assert redis_store.peek("key", _WINDOW).count == 1
        assert redis_store.peek("key", _WINDOW).count == 1

    def test_keys_are_isolated(self, redis_store: RedisRateLimitStore) -> None:
        redis_store.record_and_get("alice", _WINDOW)
        redis_store.record_and_get("alice", _WINDOW)
        redis_store.record_and_get("bob", _WINDOW)
        assert redis_store.peek("alice", _WINDOW).count == 2
        assert redis_store.peek("bob", _WINDOW).count == 1

    def test_unknown_key_peek_returns_empty(self, redis_store: RedisRateLimitStore) -> None:
        assert redis_store.peek("ghost", _WINDOW).count == 0

    def test_expired_timestamps_are_evicted_on_record(
        self, redis_store: RedisRateLimitStore
    ) -> None:
        # A zero-length window means every previously recorded request has already
        # aged out, so each record starts a fresh single-entry window.
        redis_store.record_and_get("key", 0)
        assert redis_store.record_and_get("key", 0).count == 1

    def test_concurrent_members_at_same_timestamp_coexist(
        self, redis_store: RedisRateLimitStore
    ) -> None:
        # Unique members must prevent same-score requests from overwriting.
        counts = [redis_store.record_and_get("key", _WINDOW).count for _ in range(3)]
        assert counts == [1, 2, 3]


class TestRateLimiterWithRedisBackend:
    """The RateLimiter must behave identically regardless of the injected store."""

    def _limiter(self) -> RateLimiter:
        return RateLimiter(
            RateLimitPolicy(3, _WINDOW, burst_allowance=2),
            store=RedisRateLimitStore(client=fakeredis.FakeStrictRedis()),
        )

    def test_requests_within_effective_limit_are_allowed(self) -> None:
        limiter = self._limiter()
        results = [limiter.check_and_record("user") for _ in range(5)]  # 3 + 2 burst
        assert all(r.allowed for r in results)

    def test_request_beyond_effective_limit_is_blocked(self) -> None:
        limiter = self._limiter()
        for _ in range(5):
            limiter.check_and_record("user")
        blocked = limiter.check_and_record("user")
        assert not blocked.allowed
        assert blocked.remaining == 0
        assert blocked.retry_after is not None

    def test_keys_have_independent_quotas(self) -> None:
        limiter = self._limiter()
        for _ in range(5):
            limiter.check_and_record("alice")
        assert not limiter.check_and_record("alice").allowed
        assert limiter.check_and_record("bob").allowed


# ── Rejected requests must not occupy the window ──────────────────────────────


class _Clock:
    """Controllable replacement for ``time.time`` inside the rate limiter module."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    from aegis.infrastructure.security import rate_limiter as module

    fake = _Clock()
    monkeypatch.setattr(module.time, "time", fake)
    return fake


@pytest.fixture(params=["memory", "redis"])
def any_store(request: pytest.FixtureRequest) -> RateLimitStore:
    """Both backends, so the behaviour is asserted against the port, not one adapter."""
    if request.param == "redis":
        return RedisRateLimitStore(client=fakeredis.FakeStrictRedis())
    return InMemoryRateLimitStore()


class TestWindowState:
    """Stores report the window's size and oldest entry, not every timestamp."""

    def test_reports_the_count_and_the_oldest_entry(
        self, any_store: RateLimitStore, clock: _Clock
    ) -> None:
        first = clock.now
        for _ in range(3):
            any_store.record_and_get("key", _WINDOW)
            clock.now += 1.0

        recorded = any_store.record_and_get("key", _WINDOW)
        peeked = any_store.peek("key", _WINDOW)

        assert (recorded.count, recorded.oldest) == (4, pytest.approx(first))
        assert (peeked.count, peeked.oldest) == (4, pytest.approx(first))

    def test_the_oldest_entry_moves_as_entries_expire(
        self, any_store: RateLimitStore, clock: _Clock
    ) -> None:
        any_store.record_and_get("key", _WINDOW)
        clock.now += 10.0
        second = clock.now
        any_store.record_and_get("key", _WINDOW)
        clock.now += _WINDOW - 5.0  # the first entry is now out of the window

        assert any_store.peek("key", _WINDOW).oldest == pytest.approx(second)

    def test_an_empty_window_has_no_oldest_entry(self, any_store: RateLimitStore) -> None:
        state = any_store.peek("ghost", _WINDOW)
        assert (state.count, state.oldest) == (0, None)

    def test_a_large_window_is_summarised_not_transferred(self) -> None:
        """Regression: the Redis store read back the whole window on every request."""
        client = fakeredis.FakeStrictRedis()
        store = RedisRateLimitStore(client=client)
        for _ in range(500):
            store.record_and_get("key", _WINDOW)

        replies: list[object] = []
        real_execute = client.pipeline

        def recording_pipeline(*args: object, **kwargs: object) -> object:
            pipe = real_execute(*args, **kwargs)
            execute = pipe.execute

            def capture() -> list[object]:
                result = execute()
                replies.extend(result)
                return result

            pipe.execute = capture
            return pipe

        client.pipeline = recording_pipeline  # type: ignore[method-assign]
        assert store.record_and_get("key", _WINDOW).count == 501
        # No reply in the transaction carries more than one window member.
        assert all(not isinstance(r, list) or len(r) <= 1 for r in replies)


class TestRejectedRequestsAreNotRetained:
    def test_store_keeps_at_most_max_entries(self, any_store: RateLimitStore) -> None:
        for _ in range(50):
            any_store.record_and_get("key", _WINDOW, max_entries=5)
        assert any_store.peek("key", _WINDOW).count == 5

    def test_over_limit_request_is_still_visible_to_the_caller(
        self, any_store: RateLimitStore
    ) -> None:
        """The returned window includes the rejected request, or it would look admitted."""
        for _ in range(5):
            any_store.record_and_get("key", _WINDOW, max_entries=5)
        assert any_store.record_and_get("key", _WINDOW, max_entries=5).count == 6

    def test_window_does_not_grow_under_a_retry_storm(self, any_store: RateLimitStore) -> None:
        limiter = RateLimiter(RateLimitPolicy(5, _WINDOW), store=any_store)
        results = [limiter.check_and_record("user") for _ in range(2_000)]

        assert sum(r.allowed for r in results) == 5
        assert any_store.peek("user", _WINDOW).count == 5

    def test_client_that_waits_retry_after_is_admitted(
        self, any_store: RateLimitStore, clock: _Clock
    ) -> None:
        """
        Regression: rejected attempts used to be stored, so a client polling while
        throttled kept its own window full and Retry-After never came true.
        """
        limiter = RateLimiter(RateLimitPolicy(5, 60), store=any_store)
        for _ in range(5):
            assert limiter.check_and_record("user").allowed

        rejected = None
        for _ in range(50):  # keeps retrying once a second while throttled
            clock.now += 1
            rejected = limiter.check_and_record("user")
        assert rejected is not None and not rejected.allowed
        assert rejected.retry_after == pytest.approx(10.0)

        clock.now += rejected.retry_after + 0.01
        assert limiter.check_and_record("user").allowed

    def test_reset_time_is_when_the_oldest_admitted_request_expires(
        self, any_store: RateLimitStore, clock: _Clock
    ) -> None:
        limiter = RateLimiter(RateLimitPolicy(2, 60), store=any_store)
        first_at = clock.now
        limiter.check_and_record("user")
        clock.now += 20
        limiter.check_and_record("user")
        clock.now += 20

        rejected = limiter.check_and_record("user")
        assert not rejected.allowed
        assert rejected.reset_at == pytest.approx(first_at + 60)
