"""
The rate limiter must never hold the event loop while its store waits on I/O.

Regression for a load-test finding: with Redis paused, every request blocked
the loop for the client's socket timeout, so /health, which never touches the
limiter, took as long as the outage. Network-backed stores now run in a worker
thread; in-process stores, which are fast and not thread-safe, stay inline.
"""

from __future__ import annotations

import asyncio
import threading
import time

from aegis.infrastructure.security.rate_limiter import (
    InMemoryRateLimitStore,
    RateLimiter,
    RateLimitPolicy,
    WindowState,
)
from aegis.interface.api.middleware.security_middleware import _check


class _SlowNetworkStore(InMemoryRateLimitStore):
    """Answers like the in-memory store, after a blocking wait, and says it does I/O."""

    performs_io = True

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay
        self.threads: list[int] = []

    def record_and_get(
        self, key: str, window_seconds: int, max_entries: int | None = None
    ) -> WindowState:
        self.threads.append(threading.get_ident())
        time.sleep(self.delay)  # a synchronous client waiting on the network
        return super().record_and_get(key, window_seconds, max_entries)


def _limiter(store: InMemoryRateLimitStore) -> RateLimiter:
    return RateLimiter(RateLimitPolicy(100, 60), store=store)


class TestWhereTheStoreRuns:
    async def test_a_network_store_runs_in_a_worker_thread(self) -> None:
        store = _SlowNetworkStore(delay=0)
        limiter = _limiter(store)
        await _check(limiter, limiter.check_and_record, "client")

        assert store.threads and store.threads[0] != threading.get_ident()

    async def test_an_in_process_store_runs_inline(self) -> None:
        class Recording(InMemoryRateLimitStore):
            threads: list[int] = []

            def record_and_get(
                self, key: str, window_seconds: int, max_entries: int | None = None
            ) -> WindowState:
                self.threads.append(threading.get_ident())
                return super().record_and_get(key, window_seconds, max_entries)

        store = Recording()
        limiter = _limiter(store)
        await _check(limiter, limiter.check_and_record, "client")

        assert store.threads == [threading.get_ident()]


class TestTheLoopKeepsRunning:
    async def test_a_stalled_store_does_not_stop_other_work(self) -> None:
        limiter = _limiter(_SlowNetworkStore(delay=0.3))
        started = time.perf_counter()
        ticked_at: list[float] = []

        async def other_requests() -> None:
            for _ in range(5):
                await asyncio.sleep(0.01)
                ticked_at.append(time.perf_counter() - started)

        check = asyncio.create_task(_check(limiter, limiter.check_and_record, "client"))
        await other_requests()
        result = await check

        assert result.allowed
        # Inline, the store's 0.3 s wait would hold the loop, and no tick could
        # run until it ended. Off the loop, all five finish well inside it.
        assert ticked_at[-1] < 0.2
