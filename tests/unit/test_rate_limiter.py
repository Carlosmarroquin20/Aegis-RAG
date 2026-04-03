"""
Unit tests for the sliding-window RateLimiter and InMemoryRateLimitStore.

All tests run without I/O or time mocking: a large window (3600 s) ensures
no timestamps expire mid-test, making assertions deterministic.
"""

from __future__ import annotations

import pytest

from aegis.infrastructure.security.rate_limiter import (
    InMemoryRateLimitStore,
    RateLimitPolicy,
    RateLimiter,
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
        with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
            policy.requests_per_window = 99  # type: ignore[misc]


# ── InMemoryRateLimitStore ─────────────────────────────────────────────────────


class TestInMemoryRateLimitStore:
    def test_record_returns_one_timestamp_on_first_call(self) -> None:
        store = InMemoryRateLimitStore()
        result = store.record_and_get("key", _WINDOW)
        assert len(result) == 1

    def test_second_record_returns_two_timestamps(self) -> None:
        store = InMemoryRateLimitStore()
        store.record_and_get("key", _WINDOW)
        result = store.record_and_get("key", _WINDOW)
        assert len(result) == 2

    def test_peek_does_not_add_timestamps(self) -> None:
        store = InMemoryRateLimitStore()
        store.record_and_get("key", _WINDOW)
        assert len(store.peek("key", _WINDOW)) == 1
        assert len(store.peek("key", _WINDOW)) == 1  # still 1 after two peeks

    def test_keys_are_isolated(self) -> None:
        store = InMemoryRateLimitStore()
        store.record_and_get("alice", _WINDOW)
        store.record_and_get("alice", _WINDOW)
        store.record_and_get("bob", _WINDOW)
        assert len(store.peek("alice", _WINDOW)) == 2
        assert len(store.peek("bob", _WINDOW)) == 1

    def test_unknown_key_peek_returns_empty(self) -> None:
        store = InMemoryRateLimitStore()
        assert store.peek("ghost", _WINDOW) == []


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
