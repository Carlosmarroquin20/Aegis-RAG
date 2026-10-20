"""
Tests for the LLM fallback router and its circuit breakers.

Backends are scripted fakes and time is a manual clock, so every transition of
the breaker can be driven deterministically: closed, open after the threshold,
half-open after the cool-down, and back.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from aegis.domain.models.document import Document
from aegis.domain.models.generation import Generation, TokenUsage
from aegis.domain.ports.llm_client import (
    LLMClientPort,
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from aegis.infrastructure.llm.fallback import (
    Backend,
    CircuitBreaker,
    CircuitState,
    FallbackLLMClient,
)
from aegis.infrastructure.observability.metrics import registry


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _ScriptedLLM(LLMClientPort):
    """Answers, or raises the next scripted error. Counts every call."""

    def __init__(self, name: str, *errors: Exception, healthy: bool = True) -> None:
        self.name = name
        self.errors: list[Exception] = list(errors)
        self.calls = 0
        self.closed = False
        self._healthy = healthy

    async def generate(
        self,
        query: str,
        context: list[Document],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Generation:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return Generation(
            text=f"from {self.name}", provider=self.name, model="m", usage=TokenUsage(1, 1)
        )

    async def health_check(self) -> bool:
        return self._healthy

    async def aclose(self) -> None:
        self.closed = True


def _router(
    primary: _ScriptedLLM, fallback: _ScriptedLLM, clock: _Clock, *, threshold: int = 3
) -> FallbackLLMClient:
    return FallbackLLMClient(
        [Backend(primary.name, primary), Backend(fallback.name, fallback)],
        failure_threshold=threshold,
        reset_seconds=30.0,
        clock=clock,
    )


def _down(n: int) -> list[Exception]:
    return [LLMUnavailableError("down") for _ in range(n)]


def _fallbacks(reason: str) -> float | None:
    labels = {"from_provider": "groq", "to_provider": "ollama", "reason": reason}
    return registry.get_sample_value("aegis_llm_fallbacks_total", labels)


@pytest.fixture()
def clock() -> _Clock:
    return _Clock()


@pytest.fixture(autouse=True)
def _clean_metrics(reset_prometheus_registry: None) -> Iterator[None]:
    yield


class TestRouting:
    async def test_the_primary_answers_when_it_can(self, clock: _Clock) -> None:
        primary, fallback = _ScriptedLLM("groq"), _ScriptedLLM("ollama")
        generation = await _router(primary, fallback, clock).generate("q", [])

        assert generation.provider == "groq"
        assert fallback.calls == 0
        assert _fallbacks("unavailable") is None

    @pytest.mark.parametrize(
        ("error", "reason"),
        [
            (LLMUnavailableError("down"), "unavailable"),
            (LLMTimeoutError("slow"), "timeout"),
            (LLMRateLimitedError("busy", retry_after=5), "rate_limited"),
        ],
        ids=["unavailable", "timeout", "rate-limited"],
    )
    async def test_a_failed_primary_falls_back_and_says_why(
        self, clock: _Clock, error: Exception, reason: str
    ) -> None:
        primary, fallback = _ScriptedLLM("groq", error), _ScriptedLLM("ollama")
        generation = await _router(primary, fallback, clock).generate("q", [])

        # The answer reports who produced it, so usage and cost are attributed right.
        assert generation.provider == "ollama"
        assert _fallbacks(reason) == 1

    async def test_when_every_backend_fails_the_last_error_keeps_its_type(
        self, clock: _Clock
    ) -> None:
        """A timeout must still become a 504, not a generic 503."""
        primary = _ScriptedLLM("groq", LLMUnavailableError("down"))
        fallback = _ScriptedLLM("ollama", LLMTimeoutError("slow"))
        with pytest.raises(LLMTimeoutError):
            await _router(primary, fallback, clock).generate("q", [])

    async def test_a_defect_is_raised_not_routed_around(self, clock: _Clock) -> None:
        primary = _ScriptedLLM("groq", RuntimeError("bug"))
        fallback = _ScriptedLLM("ollama")
        router = _router(primary, fallback, clock, threshold=1)
        with pytest.raises(RuntimeError):
            await router.generate("q", [])
        assert fallback.calls == 0
        # Counted against the backend, so a buggy backend still trips its circuit.
        assert router.breaker("groq").state is CircuitState.OPEN

    def test_a_single_backend_is_not_a_fallback(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            FallbackLLMClient([Backend("groq", _ScriptedLLM("groq"))])


class TestCircuitBreaker:
    async def test_the_circuit_opens_after_the_threshold(self, clock: _Clock) -> None:
        primary, fallback = _ScriptedLLM("groq", *_down(3)), _ScriptedLLM("ollama")
        router = _router(primary, fallback, clock)
        for _ in range(4):
            await router.generate("q", [])

        # Three real failures, then the fourth request skips the primary entirely.
        assert primary.calls == 3
        assert _fallbacks("unavailable") == 3
        assert _fallbacks("circuit_open") == 1
        assert registry.get_sample_value("aegis_llm_circuit_open", {"provider": "groq"}) == 1

    async def test_a_success_resets_the_failure_count(self, clock: _Clock) -> None:
        primary = _ScriptedLLM("groq", *_down(2))
        router = _router(primary, _ScriptedLLM("ollama"), clock)
        await router.generate("q", [])
        await router.generate("q", [])
        await router.generate("q", [])  # primary answers
        primary.errors = _down(2)
        await router.generate("q", [])
        await router.generate("q", [])

        assert router.breaker("groq").state is CircuitState.CLOSED

    async def test_one_trial_after_the_cool_down_closes_it_again(self, clock: _Clock) -> None:
        primary = _ScriptedLLM("groq", *_down(3))
        router = _router(primary, _ScriptedLLM("ollama"), clock)
        for _ in range(3):
            await router.generate("q", [])

        clock.now += 31
        generation = await router.generate("q", [])

        assert generation.provider == "groq"
        assert router.breaker("groq").state is CircuitState.CLOSED
        assert registry.get_sample_value("aegis_llm_circuit_open", {"provider": "groq"}) == 0

    async def test_a_failed_trial_opens_it_for_another_period(self, clock: _Clock) -> None:
        primary = _ScriptedLLM("groq", *_down(4))
        router = _router(primary, _ScriptedLLM("ollama"), clock)
        for _ in range(3):
            await router.generate("q", [])

        clock.now += 31
        await router.generate("q", [])  # the trial fails
        await router.generate("q", [])  # skipped: open again

        assert primary.calls == 4
        assert router.breaker("groq").state is CircuitState.OPEN

    async def test_a_rate_limit_opens_it_at_once_for_the_requested_wait(
        self, clock: _Clock
    ) -> None:
        """Calling a provider sooner than it asked only earns another refusal."""
        primary = _ScriptedLLM("groq", LLMRateLimitedError("busy", retry_after=120))
        router = _router(primary, _ScriptedLLM("ollama"), clock)
        await router.generate("q", [])
        assert router.breaker("groq").state is CircuitState.OPEN

        clock.now += 100  # past the 30 s reset period, inside the provider's 120 s
        await router.generate("q", [])
        assert primary.calls == 1

        clock.now += 21
        await router.generate("q", [])
        assert primary.calls == 2

    async def test_every_circuit_open_is_an_unavailable_backend(self, clock: _Clock) -> None:
        primary = _ScriptedLLM("groq", *_down(1))
        fallback = _ScriptedLLM("ollama", *_down(1))
        router = _router(primary, fallback, clock, threshold=1)
        with pytest.raises(LLMUnavailableError):
            await router.generate("q", [])
        with pytest.raises(LLMUnavailableError, match="open circuit"):
            await router.generate("q", [])
        assert (primary.calls, fallback.calls) == (1, 1)

    def test_the_half_open_state_admits_a_single_trial(self, clock: _Clock) -> None:
        breaker = CircuitBreaker("groq", failure_threshold=1, reset_seconds=30, clock=clock)
        breaker.record_failure()
        clock.now += 31

        assert breaker.allows_call()
        # A second request arriving while the trial is in flight goes elsewhere.
        assert not breaker.allows_call()


class TestLifecycle:
    async def test_ready_while_any_backend_is_healthy(self, clock: _Clock) -> None:
        primary = _ScriptedLLM("groq", healthy=False)
        assert await _router(primary, _ScriptedLLM("ollama"), clock).health_check()

    async def test_not_ready_when_no_backend_is_healthy(self, clock: _Clock) -> None:
        router = _router(
            _ScriptedLLM("groq", healthy=False), _ScriptedLLM("ollama", healthy=False), clock
        )
        assert not await router.health_check()

    async def test_shutdown_closes_every_backend(self, clock: _Clock) -> None:
        primary, fallback = _ScriptedLLM("groq"), _ScriptedLLM("ollama")
        await _router(primary, fallback, clock).aclose()
        assert primary.closed and fallback.closed


# ── Settings and wiring ───────────────────────────────────────────────────────

_HOSTED = {
    "llm_base_url": "https://api.groq.com/openai/v1",
    "llm_model": "openai/gpt-oss-120b",
    "llm_api_key": "sk-test",
}


class TestFallbackSettings:
    def test_no_fallback_by_default(self) -> None:
        from aegis.config import Settings

        assert Settings(_env_file=None).llm_fallback_provider == ""  # type: ignore[call-arg]

    def test_unknown_fallback_is_rejected(self) -> None:
        from aegis.config import Settings

        with pytest.raises(ValueError, match="llm_fallback_provider"):
            Settings(_env_file=None, llm_fallback_provider="bedrock")  # type: ignore[call-arg]

    def test_a_backend_cannot_be_its_own_fallback(self) -> None:
        from aegis.config import Settings

        with pytest.raises(ValueError, match="must differ"):
            Settings(_env_file=None, llm_fallback_provider="ollama")  # type: ignore[call-arg]

    def test_a_hosted_fallback_needs_its_settings_too(self) -> None:
        from aegis.config import Settings

        with pytest.raises(ValueError, match="LLM_BASE_URL"):
            Settings(_env_file=None, llm_fallback_provider="openai_compatible")  # type: ignore[call-arg]


class TestFallbackWiring:
    async def test_configuring_a_fallback_routes_through_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aegis.config import Settings
        from aegis.interface.api import dependencies

        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            llm_provider="openai_compatible",
            llm_fallback_provider="ollama",
            **_HOSTED,
        )
        monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
        dependencies.get_llm_client.cache_clear()
        try:
            client = dependencies.get_llm_client()
            assert isinstance(client, FallbackLLMClient)
            # Named as their answers will name themselves, so the circuit gauge,
            # the fallback counter and the cost metrics all agree.
            assert client.breaker("api.groq.com").state is CircuitState.CLOSED
            assert client.breaker("ollama").state is CircuitState.CLOSED
            await client.aclose()
        finally:
            dependencies.get_llm_client.cache_clear()
