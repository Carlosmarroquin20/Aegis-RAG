"""
RedisRateLimitStore against a real Redis server.

The unit suite exercises the store through fakeredis, which is fast and hermetic
but is also a reimplementation: it tells us our code is consistent with a model
of Redis, not with Redis. These tests run the same behaviour against an actual
server, so a Redis major-version upgrade is validated by something other than
the hope that the sorted-set and transaction semantics did not move.

Opt-in: they run only when ``AEGIS_TEST_REDIS_URL`` is set (the CI stack-smoke
job points it at the compose Redis) and are skipped everywhere else, so the
default test run still needs no services.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator

import pytest

from aegis.infrastructure.security.rate_limiter import (
    RateLimiter,
    RateLimitPolicy,
    RedisRateLimitStore,
)

_REDIS_URL = os.environ.get("AEGIS_TEST_REDIS_URL", "")

pytestmark = pytest.mark.skipif(
    not _REDIS_URL, reason="set AEGIS_TEST_REDIS_URL to run against a real Redis"
)


@pytest.fixture()
def prefix() -> str:
    # Unique per test so runs never see each other's keys, even on a shared server.
    return f"aegis:test:{uuid.uuid4().hex}:"


@pytest.fixture()
def store(prefix: str) -> Iterator[RedisRateLimitStore]:
    redis_store = RedisRateLimitStore(_REDIS_URL, key_prefix=prefix)
    yield redis_store
    client = redis_store._client
    for key in client.scan_iter(match=f"{prefix}*"):
        client.delete(key)


class TestAgainstRealRedis:
    def test_server_is_reachable(self, store: RedisRateLimitStore) -> None:
        assert store._client.ping() is True

    def test_record_counts_each_request(self, store: RedisRateLimitStore) -> None:
        counts = [store.record_and_get("client", 60).count for _ in range(4)]
        assert counts == [1, 2, 3, 4]

    def test_peek_does_not_record(self, store: RedisRateLimitStore) -> None:
        store.record_and_get("client", 60)
        assert store.peek("client", 60).count == 1
        assert store.peek("client", 60).count == 1

    def test_simultaneous_requests_are_all_kept(self, store: RedisRateLimitStore) -> None:
        """Members are unique, so identical timestamps must not overwrite each other."""
        for _ in range(25):
            store.record_and_get("client", 60)
        assert store.peek("client", 60).count == 25

    def test_window_never_exceeds_max_entries(self, store: RedisRateLimitStore) -> None:
        for _ in range(40):
            store.record_and_get("client", 60, max_entries=5)
        assert store.peek("client", 60).count == 5

    def test_entries_expire_with_the_window(self, store: RedisRateLimitStore) -> None:
        store.record_and_get("client", 1)
        time.sleep(1.2)
        assert store.peek("client", 1).count == 0
        assert store.record_and_get("client", 1).count == 1

    def test_key_gets_a_ttl_so_idle_clients_do_not_accumulate(
        self, store: RedisRateLimitStore, prefix: str
    ) -> None:
        store.record_and_get("client", 30)
        ttl = store._client.ttl(prefix + "client")
        assert 0 < ttl <= 30

    def test_limiter_blocks_and_recovers_through_the_real_store(
        self, store: RedisRateLimitStore
    ) -> None:
        limiter = RateLimiter(RateLimitPolicy(3, 1), store=store)

        results = [limiter.check_and_record("client") for _ in range(6)]
        assert [r.allowed for r in results] == [True, True, True, False, False, False]
        assert results[-1].retry_after is not None and results[-1].retry_after <= 1

        time.sleep(1.2)
        assert limiter.check_and_record("client").allowed

    def test_clients_do_not_share_a_window(self, store: RedisRateLimitStore) -> None:
        limiter = RateLimiter(RateLimitPolicy(1, 60), store=store)
        assert limiter.check_and_record("alice").allowed
        assert not limiter.check_and_record("alice").allowed
        assert limiter.check_and_record("bob").allowed
