"""
Unit tests for failed-authentication throttling and client address resolution.

The throttle is only as good as the address it buckets by, so the two are tested
together. The properties that matter:

  - a lockout cannot be dodged by spoofing X-Forwarded-For when no proxy is trusted;
  - during a lockout a correct key is refused as well, so nothing distinguishes
    a right guess from a wrong one;
  - a client that authenticates correctly never contributes to a lockout.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from aegis.config import MIN_RECOMMENDED_API_KEY_CHARS, Settings
from aegis.infrastructure.observability.metrics import (
    auth_failures_total,
    rate_limit_backend_errors_total,
)
from aegis.infrastructure.security import rate_limiter as rate_limiter_module
from aegis.infrastructure.security.rate_limiter import (
    InMemoryRateLimitStore,
    RateLimiter,
    RateLimitPolicy,
    WindowState,
)
from aegis.interface.api.middleware.security_middleware import (
    APIKeyMiddleware,
    resolve_client_ip,
)

_KEY = "correct-key"
_LIMIT = 3
_WINDOW = 60


def _client(
    *,
    hops: int = 0,
    limiter: RateLimiter | None = None,
    throttle: bool = True,
) -> TestClient:
    if limiter is None and throttle:
        limiter = RateLimiter(RateLimitPolicy(_LIMIT, _WINDOW))
    app = FastAPI()
    app.add_middleware(
        APIKeyMiddleware,
        settings=Settings(valid_api_keys=_KEY, trusted_proxy_hops=hops),
        failure_limiter=limiter,
    )

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"pong": True}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return TestClient(app)


def _get(client: TestClient, key: str = "wrong", *, forwarded: str | None = None) -> int:
    headers = {"X-API-Key": key}
    if forwarded is not None:
        headers["X-Forwarded-For"] = forwarded
    return client.get("/ping", headers=headers).status_code


# ── Address resolution ────────────────────────────────────────────────────────


def _request(forwarded: str | None, peer: str = "10.0.0.9") -> Request:
    headers = [] if forwarded is None else [(b"x-forwarded-for", forwarded.encode())]
    return Request({"type": "http", "headers": headers, "client": (peer, 5000)})


class TestResolveClientIp:
    def test_header_is_ignored_when_no_proxy_is_trusted(self) -> None:
        assert resolve_client_ip(_request("1.1.1.1"), 0) == "10.0.0.9"

    def test_single_trusted_proxy_uses_the_rightmost_entry(self) -> None:
        assert resolve_client_ip(_request("203.0.113.7"), 1) == "203.0.113.7"

    def test_entries_left_of_the_trusted_depth_are_never_used(self) -> None:
        """The client can prepend anything; only the proxy-appended entry counts."""
        assert resolve_client_ip(_request("6.6.6.6, 9.9.9.9, 203.0.113.7"), 1) == "203.0.113.7"

    def test_two_trusted_proxies_use_the_second_entry_from_the_right(self) -> None:
        forwarded = "6.6.6.6, 203.0.113.7, 10.0.0.2"
        assert resolve_client_ip(_request(forwarded), 2) == "203.0.113.7"

    @pytest.mark.parametrize("forwarded", [None, "", " , ", "203.0.113.7"])
    def test_falls_back_to_the_peer_when_the_chain_is_too_short(
        self, forwarded: str | None
    ) -> None:
        assert resolve_client_ip(_request(forwarded), 2) == "10.0.0.9"

    def test_oversized_entry_is_truncated(self) -> None:
        assert len(resolve_client_ip(_request("a" * 500), 1)) == 64

    def test_missing_peer_is_reported_as_unknown(self) -> None:
        request = Request({"type": "http", "headers": [], "client": None})
        assert resolve_client_ip(request, 0) == "unknown"


# ── Lockout behaviour ─────────────────────────────────────────────────────────


class TestFailedAuthLockout:
    def test_failures_up_to_the_limit_are_plain_rejections(self) -> None:
        client = _client()
        assert [_get(client) for _ in range(_LIMIT)] == [403] * _LIMIT

    def test_next_attempt_after_the_limit_is_throttled(self) -> None:
        client = _client()
        for _ in range(_LIMIT):
            _get(client)

        response = client.get("/ping", headers={"X-API-Key": "wrong"})
        assert response.status_code == 429
        assert 1 <= int(response.headers["Retry-After"]) <= _WINDOW

    def test_correct_key_is_also_refused_during_the_lockout(self) -> None:
        """Otherwise 429-vs-200 would still reveal which guess was right."""
        client = _client()
        for _ in range(_LIMIT):
            _get(client)
        assert _get(client, _KEY) == 429

    def test_successful_requests_never_count_towards_a_lockout(self) -> None:
        client = _client()
        assert [_get(client, _KEY) for _ in range(_LIMIT * 5)] == [200] * (_LIMIT * 5)

    def test_public_paths_stay_reachable_during_a_lockout(self) -> None:
        """Health probes come from the same address as everything else behind a proxy."""
        client = _client()
        for _ in range(_LIMIT + 1):
            _get(client)
        assert client.get("/health").status_code == 200

    def test_lockout_lifts_when_the_oldest_failure_expires(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = [1_000.0]
        monkeypatch.setattr(rate_limiter_module.time, "time", lambda: now[0])
        client = _client()
        for _ in range(_LIMIT):
            _get(client)
        assert _get(client, _KEY) == 429

        now[0] += _WINDOW + 1
        assert _get(client, _KEY) == 200

    def test_throttled_attempts_do_not_extend_the_lockout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A client that keeps hammering must not be able to lock itself out forever."""
        now = [1_000.0]
        monkeypatch.setattr(rate_limiter_module.time, "time", lambda: now[0])
        client = _client()
        for _ in range(_LIMIT):
            _get(client)
        for _ in range(50):
            now[0] += 1
            assert _get(client) == 429

        now[0] = 1_000.0 + _WINDOW + 1
        assert _get(client, _KEY) == 200

    def test_disabled_when_no_limiter_is_configured(self) -> None:
        client = _client(throttle=False)
        assert [_get(client) for _ in range(_LIMIT * 4)] == [403] * (_LIMIT * 4)


class TestLockoutIsPerClientAddress:
    def test_spoofed_forwarded_header_does_not_evade_the_lockout_without_a_proxy(self) -> None:
        """With hops=0 the header is attacker-controlled and must not pick the bucket."""
        client = _client(hops=0)
        statuses = [_get(client, forwarded=f"198.51.100.{i}") for i in range(_LIMIT + 2)]
        assert statuses == [403] * _LIMIT + [429, 429]

    def test_clients_behind_a_trusted_proxy_are_bucketed_separately(self) -> None:
        client = _client(hops=1)
        for _ in range(_LIMIT):
            _get(client, forwarded="203.0.113.7")

        assert _get(client, _KEY, forwarded="203.0.113.7") == 429
        assert _get(client, _KEY, forwarded="203.0.113.8") == 200

    def test_prepending_addresses_does_not_evade_the_lockout_behind_a_proxy(self) -> None:
        client = _client(hops=1)
        statuses = [
            _get(client, forwarded=f"198.51.100.{i}, 203.0.113.7") for i in range(_LIMIT + 1)
        ]
        assert statuses == [403] * _LIMIT + [429]


# ── Store outage and metrics ──────────────────────────────────────────────────


class _UnreachableStore(InMemoryRateLimitStore):
    def record_and_get(
        self, key: str, window_seconds: int, max_entries: int | None = None
    ) -> WindowState:
        raise ConnectionError("redis refused")

    def peek(self, key: str, window_seconds: int) -> WindowState:
        raise ConnectionError("redis refused")


class TestThrottleStoreOutage:
    def test_authentication_keeps_working_when_the_store_is_down(
        self, reset_prometheus_registry: None
    ) -> None:
        limiter = RateLimiter(RateLimitPolicy(_LIMIT, _WINDOW), _UnreachableStore())
        client = _client(limiter=limiter)

        assert _get(client, _KEY) == 200
        assert _get(client) == 403
        # One failed peek per request, plus one failed record for the rejection.
        assert rate_limit_backend_errors_total._value.get() == 3


class TestAuthMetrics:
    def test_rejections_and_throttled_attempts_are_counted_separately(
        self, reset_prometheus_registry: None
    ) -> None:
        client = _client()
        for _ in range(_LIMIT + 2):
            _get(client)

        assert auth_failures_total.labels(outcome="rejected")._value.get() == _LIMIT
        assert auth_failures_total.labels(outcome="throttled")._value.get() == 2


# ── Configuration ─────────────────────────────────────────────────────────────


class TestThrottleSettings:
    def test_defaults_do_not_trust_any_proxy(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.trusted_proxy_hops == 0
        assert settings.auth_failure_limit > 0

    def test_negative_hops_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="trusted_proxy_hops"):
            Settings(trusted_proxy_hops=-1)

    def test_weak_keys_are_counted(self) -> None:
        strong = "s" * MIN_RECOMMENDED_API_KEY_CHARS
        settings = Settings(valid_api_keys=f"short,{strong},also-short")
        assert settings.weak_api_key_count == 2

    def test_limiter_factory_returns_none_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aegis.interface.api import dependencies

        monkeypatch.setattr(dependencies, "get_settings", lambda: Settings(auth_failure_limit=0))
        dependencies.get_auth_failure_limiter.cache_clear()
        try:
            assert dependencies.get_auth_failure_limiter() is None
        finally:
            dependencies.get_auth_failure_limiter.cache_clear()
