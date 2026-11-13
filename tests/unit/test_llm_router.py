"""
The model router (ADR 0026): simple questions to the small model, the rest to
the large one, and the large one whenever the small one cannot answer.

Backends are scripted fakes and time is a manual clock, as in the fallback
tests, so the small model's circuit can be driven deterministically.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator

import pytest

from aegis.domain.models.document import Document
from aegis.domain.models.generation import Generation, GenerationEvent, TextDelta, TokenUsage
from aegis.domain.ports.llm_client import (
    LLMClientPort,
    LLMRateLimitedError,
    LLMUnavailableError,
)
from aegis.infrastructure.llm.fallback import Backend, CircuitState
from aegis.infrastructure.llm.router import RoutingLLMClient
from aegis.infrastructure.observability.metrics import (
    LLMPricing,
    PrometheusQueryMetrics,
    registry,
)

_SIMPLE = "Do you deliver on Sundays?"
_COMPLEX = "What is the difference between the standard and the extended warranty?"
_CONTEXT = [Document(id="d1", content="We deliver Monday to Saturday.", metadata={})]
_SPANISH_ANSWER = "No, las entregas se hacen de lunes a sábado y no hay servicio los domingos."
_LONG_ENGLISH = (
    "No, we do not deliver on Sundays. Deliveries are made from Monday to Saturday, and an "
    "order placed on a Saturday afternoon is scheduled for the following Monday at the earliest."
)
_LONG_SPANISH = (
    "No, las entregas se hacen de lunes a sábado. Un pedido que se hace el sábado por la "
    "tarde se programa para el lunes siguiente, y no hay servicio de entrega los domingos."
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _ScriptedLLM(LLMClientPort):
    """
    Answers ``pieces`` joined (by default "from <name>"), or raises the next
    scripted error; streams the pieces one by one, raising after the first when
    ``fail_after_text``.
    """

    def __init__(
        self,
        name: str,
        *errors: Exception,
        fail_after_text: bool = False,
        pieces: list[str] | None = None,
    ) -> None:
        self.name = name
        self.errors: list[Exception] = list(errors)
        self.fail_after_text = fail_after_text
        self.pieces = pieces if pieces is not None else ["from ", name]
        self.calls = 0
        self.closed = False
        self.healthy = True

    def _generation(self) -> Generation:
        return Generation(
            text="".join(self.pieces), provider="host", model=self.name, usage=TokenUsage(10, 5)
        )

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
        return self._generation()

    async def stream(
        self,
        query: str,
        context: list[Document],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncGenerator[GenerationEvent, None]:
        self.calls += 1
        if self.errors and not self.fail_after_text:
            raise self.errors.pop(0)
        first, *rest = self.pieces
        yield TextDelta(first)
        if self.errors:
            raise self.errors.pop(0)
        for piece in rest:
            yield TextDelta(piece)
        yield self._generation()

    async def health_check(self) -> bool:
        return self.healthy

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _clean_metrics(reset_prometheus_registry: None) -> Iterator[None]:
    yield


@pytest.fixture()
def clock() -> _Clock:
    return _Clock()


def _router(small: _ScriptedLLM, large: _ScriptedLLM, clock: _Clock) -> RoutingLLMClient:
    return RoutingLLMClient(
        small=Backend("host/small", small),
        large=large,
        failure_threshold=2,
        reset_seconds=30.0,
        clock=clock,
    )


def _sample(name: str, labels: dict[str, str]) -> float:
    return registry.get_sample_value(name, labels) or 0.0


async def _streamed(client: LLMClientPort, query: str, context: list[Document]) -> str:
    text = ""
    async for event in client.stream(query, context):
        if isinstance(event, TextDelta):
            text += event.text
    return text


class TestRouting:
    async def test_a_simple_question_is_answered_by_the_small_model(self, clock: _Clock) -> None:
        small, large = _ScriptedLLM("small"), _ScriptedLLM("large")

        generation = await _router(small, large, clock).generate(_SIMPLE, _CONTEXT)

        assert generation.model == "small"
        assert (small.calls, large.calls) == (1, 0)
        labels = {"tier": "simple", "reason": "plain_lookup"}
        assert _sample("aegis_llm_routes_total", labels) == 1

    async def test_a_complex_question_is_answered_by_the_large_model(self, clock: _Clock) -> None:
        small, large = _ScriptedLLM("small"), _ScriptedLLM("large")

        generation = await _router(small, large, clock).generate(_COMPLEX, _CONTEXT)

        assert generation.model == "large"
        assert small.calls == 0
        labels = {"tier": "complex", "reason": "comparison"}
        assert _sample("aegis_llm_routes_total", labels) == 1

    async def test_without_context_even_a_complex_question_goes_small(self, clock: _Clock) -> None:
        small, large = _ScriptedLLM("small"), _ScriptedLLM("large")

        generation = await _router(small, large, clock).generate(_COMPLEX, [])

        assert generation.model == "small"


class TestEscalation:
    async def test_a_failing_small_model_hands_the_question_to_the_large_one(
        self, clock: _Clock
    ) -> None:
        small = _ScriptedLLM("small", LLMUnavailableError("down"))
        large = _ScriptedLLM("large")

        generation = await _router(small, large, clock).generate(_SIMPLE, _CONTEXT)

        assert generation.model == "large"
        assert _sample("aegis_llm_route_escalations_total", {"reason": "small_unavailable"}) == 1

    async def test_after_the_threshold_the_small_model_is_skipped(self, clock: _Clock) -> None:
        small = _ScriptedLLM("small", LLMUnavailableError("a"), LLMUnavailableError("b"))
        large = _ScriptedLLM("large")
        router = _router(small, large, clock)

        for _ in range(3):
            await router.generate(_SIMPLE, _CONTEXT)

        assert small.calls == 2
        assert router.breaker.state is CircuitState.OPEN
        assert _sample("aegis_llm_route_escalations_total", {"reason": "circuit_open"}) == 1

    async def test_after_the_cool_down_the_small_model_is_tried_again(self, clock: _Clock) -> None:
        small = _ScriptedLLM("small", LLMUnavailableError("a"), LLMUnavailableError("b"))
        router = _router(small, _ScriptedLLM("large"), clock)
        for _ in range(2):
            await router.generate(_SIMPLE, _CONTEXT)

        clock.now += 31
        generation = await router.generate(_SIMPLE, _CONTEXT)

        assert generation.model == "small"
        assert router.breaker.state is CircuitState.CLOSED

    async def test_a_rate_limit_opens_the_circuit_at_once(self, clock: _Clock) -> None:
        small = _ScriptedLLM("small", LLMRateLimitedError("429", retry_after=120))
        router = _router(small, _ScriptedLLM("large"), clock)

        await router.generate(_SIMPLE, _CONTEXT)
        clock.now += 60

        assert not router.breaker.allows_call()

    async def test_a_defect_in_the_small_model_is_raised_not_routed_around(
        self, clock: _Clock
    ) -> None:
        small = _ScriptedLLM("small", KeyError("bug"))
        large = _ScriptedLLM("large")

        with pytest.raises(KeyError):
            await _router(small, large, clock).generate(_SIMPLE, _CONTEXT)

        assert large.calls == 0

    async def test_the_large_models_failure_is_the_callers(self, clock: _Clock) -> None:
        large = _ScriptedLLM("large", LLMUnavailableError("down"))

        with pytest.raises(LLMUnavailableError):
            await _router(_ScriptedLLM("small"), large, clock).generate(_COMPLEX, _CONTEXT)


class TestStreaming:
    async def test_a_simple_question_streams_from_the_small_model(self, clock: _Clock) -> None:
        small, large = _ScriptedLLM("small"), _ScriptedLLM("large")

        assert await _streamed(_router(small, large, clock), _SIMPLE, _CONTEXT) == "from small"
        assert large.calls == 0

    async def test_a_complex_question_streams_from_the_large_model(self, clock: _Clock) -> None:
        small, large = _ScriptedLLM("small"), _ScriptedLLM("large")

        assert await _streamed(_router(small, large, clock), _COMPLEX, _CONTEXT) == "from large"

    async def test_a_failure_before_any_text_escalates(self, clock: _Clock) -> None:
        small = _ScriptedLLM("small", LLMUnavailableError("down"))
        large = _ScriptedLLM("large")

        assert await _streamed(_router(small, large, clock), _SIMPLE, _CONTEXT) == "from large"

    async def test_a_failure_after_text_is_raised_not_restarted_elsewhere(
        self, clock: _Clock
    ) -> None:
        small = _ScriptedLLM(
            "small",
            LLMUnavailableError("cut"),
            fail_after_text=True,
            pieces=[_LONG_ENGLISH, " And more."],
        )
        large = _ScriptedLLM("large")
        router = _router(small, large, clock)

        with pytest.raises(LLMUnavailableError):
            await _streamed(router, _SIMPLE, _CONTEXT)

        assert large.calls == 0
        assert router.breaker.consecutive_failures == 1

    async def test_a_failure_while_the_start_is_still_held_escalates(self, clock: _Clock) -> None:
        # Nothing has reached the caller yet, so the large model can still answer.
        small = _ScriptedLLM("small", LLMUnavailableError("cut"), fail_after_text=True)
        large = _ScriptedLLM("large")

        assert await _streamed(_router(small, large, clock), _SIMPLE, _CONTEXT) == "from large"

    async def test_an_open_circuit_streams_from_the_large_model(self, clock: _Clock) -> None:
        small = _ScriptedLLM("small", LLMUnavailableError("a"), LLMUnavailableError("b"))
        large = _ScriptedLLM("large")
        router = _router(small, large, clock)
        for _ in range(2):
            await router.generate(_SIMPLE, _CONTEXT)

        assert await _streamed(router, _SIMPLE, _CONTEXT) == "from large"
        assert small.calls == 2

    async def test_a_caller_that_stops_after_text_counts_as_a_success(self, clock: _Clock) -> None:
        small = _ScriptedLLM("small")
        router = _router(small, _ScriptedLLM("large"), clock)
        router.breaker.record_failure()

        stream = router.stream(_SIMPLE, _CONTEXT)
        await anext(stream)
        await stream.aclose()

        assert router.breaker.consecutive_failures == 0

    async def test_a_caller_that_stops_before_text_counts_as_a_failure(self, clock: _Clock) -> None:
        class _Silent(_ScriptedLLM):
            async def stream(
                self,
                query: str,
                context: list[Document],
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> AsyncGenerator[GenerationEvent, None]:
                yield TextDelta("")
                yield TextDelta("late")

        router = _router(_Silent("small"), _ScriptedLLM("large"), clock)

        stream = router.stream(_SIMPLE, _CONTEXT)
        await anext(stream)
        await stream.aclose()

        assert router.breaker.consecutive_failures == 1


def _discarded(direction: str) -> float:
    labels = {"provider": "host", "model": "small", "direction": direction}
    return _sample("aegis_llm_discarded_tokens_total", labels)


class TestWrongLanguage:
    async def test_a_small_answer_in_another_language_is_regenerated_by_the_large_model(
        self, clock: _Clock
    ) -> None:
        small = _ScriptedLLM("small", pieces=[_SPANISH_ANSWER])
        large = _ScriptedLLM("large", pieces=[_LONG_ENGLISH])
        router = _router(small, large, clock)

        generation = await router.generate(_SIMPLE, _CONTEXT)

        assert generation.model == "large"
        assert _sample("aegis_llm_route_escalations_total", {"reason": "wrong_language"}) == 1
        # The discarded answer was billed; the use case only sees the large one.
        assert (_discarded("input"), _discarded("output")) == (10, 5)
        # The small model worked; its circuit is not penalised.
        assert router.breaker.consecutive_failures == 0

    async def test_an_answer_in_the_questions_language_is_kept(self, clock: _Clock) -> None:
        small = _ScriptedLLM("small", pieces=[_LONG_ENGLISH])
        large = _ScriptedLLM("large")

        generation = await _router(small, large, clock).generate(_SIMPLE, _CONTEXT)

        assert generation.model == "small"
        assert large.calls == 0

    async def test_an_answer_without_a_detectable_language_is_kept(self, clock: _Clock) -> None:
        small = _ScriptedLLM("small", pieces=["Q55."])
        large = _ScriptedLLM("large")

        generation = await _router(small, large, clock).generate(_SIMPLE, _CONTEXT)

        assert generation.model == "small"

    async def test_a_question_without_a_detectable_language_keeps_the_answer(
        self, clock: _Clock
    ) -> None:
        small = _ScriptedLLM("small", pieces=[_SPANISH_ANSWER])

        generation = await _router(small, _ScriptedLLM("large"), clock).generate(
            "¿Aceptan American Express?", _CONTEXT
        )

        assert generation.model == "small"

    async def test_a_streamed_answer_in_another_language_never_reaches_the_caller(
        self, clock: _Clock
    ) -> None:
        small = _ScriptedLLM("small", pieces=[_LONG_SPANISH[:80], _LONG_SPANISH[80:]])
        large = _ScriptedLLM("large", pieces=["Deliveries ", "are Monday to Saturday."])

        text = await _streamed(_router(small, large, clock), _SIMPLE, _CONTEXT)

        assert text == "Deliveries are Monday to Saturday."
        assert _sample("aegis_llm_route_escalations_total", {"reason": "wrong_language"}) == 1

    async def test_a_short_streamed_answer_is_checked_when_it_ends(self, clock: _Clock) -> None:
        small = _ScriptedLLM("small", pieces=["No, ", "las entregas son de lunes a sábado."])
        large = _ScriptedLLM("large", pieces=[_LONG_ENGLISH])

        text = await _streamed(_router(small, large, clock), _SIMPLE, _CONTEXT)

        assert text == _LONG_ENGLISH
        assert _discarded("output") == 5

    async def test_a_held_answer_in_the_right_language_is_passed_on_whole_and_in_order(
        self, clock: _Clock
    ) -> None:
        pieces = [_LONG_ENGLISH[:60], _LONG_ENGLISH[60:120], _LONG_ENGLISH[120:]]
        small = _ScriptedLLM("small", pieces=pieces)
        large = _ScriptedLLM("large")
        events = [e async for e in _router(small, large, clock).stream(_SIMPLE, _CONTEXT)]

        assert [e.text for e in events if isinstance(e, TextDelta)] == pieces
        assert isinstance(events[-1], Generation) and events[-1].model == "small"
        assert large.calls == 0


class TestLifecycle:
    async def test_readiness_follows_the_large_model(self, clock: _Clock) -> None:
        small, large = _ScriptedLLM("small"), _ScriptedLLM("large")
        router = _router(small, large, clock)
        small.healthy = False

        assert await router.health_check()
        large.healthy = False
        assert not await router.health_check()

    async def test_shutdown_closes_both_models(self, clock: _Clock) -> None:
        small, large = _ScriptedLLM("small"), _ScriptedLLM("large")

        await _router(small, large, clock).aclose()

        assert small.closed and large.closed


class TestCostPerModel:
    def test_a_model_price_wins_over_its_providers(self) -> None:
        metrics = PrometheusQueryMetrics(
            {"host": LLMPricing(1.0, 1.0), ("host", "small"): LLMPricing(0.1, 0.1)}
        )
        usage = TokenUsage(1_000_000, 0)

        metrics.llm_generation("c", Generation("a", "host", "small", usage))
        metrics.llm_generation("c", Generation("a", "host", "large", usage))

        cost = "aegis_llm_cost_usd_total"
        assert _sample(cost, {"provider": "host", "model": "small", "client": "c"}) == 0.1
        assert _sample(cost, {"provider": "host", "model": "large", "client": "c"}) == 1.0


_HOSTED = {
    "llm_base_url": "https://api.groq.com/openai/v1",
    "llm_model": "openai/gpt-oss-120b",
    "llm_api_key": "sk-test",
}


class TestRoutingSettings:
    def test_routing_is_off_by_default(self) -> None:
        from aegis.config import Settings

        assert not Settings(_env_file=None).routing_enabled  # type: ignore[call-arg]

    def test_unknown_routing_mode_is_rejected(self) -> None:
        from aegis.config import Settings

        with pytest.raises(ValueError, match="llm_routing"):
            Settings(_env_file=None, llm_routing="ml")  # type: ignore[call-arg]

    def test_routing_needs_a_small_model(self) -> None:
        from aegis.config import Settings

        with pytest.raises(ValueError, match="LLM_SMALL_MODEL"):
            Settings(_env_file=None, llm_routing="heuristic")  # type: ignore[call-arg]

    def test_the_small_model_cannot_be_the_large_one(self) -> None:
        from aegis.config import Settings

        with pytest.raises(ValueError, match="must differ"):
            Settings(  # type: ignore[call-arg]
                _env_file=None, llm_routing="heuristic", llm_small_model="llama3.2"
            )

    def test_a_hosted_small_model_needs_the_hosted_settings(self) -> None:
        from aegis.config import Settings

        with pytest.raises(ValueError, match="LLM_BASE_URL"):
            Settings(  # type: ignore[call-arg]
                _env_file=None,
                llm_routing="heuristic",
                llm_small_provider="openai_compatible",
                llm_small_model="openai/gpt-oss-20b",
            )

    def test_a_hosted_small_model_is_reported_as_sending_data_off_the_host(self) -> None:
        from aegis.config import Settings

        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            llm_routing="heuristic",
            llm_small_provider="openai_compatible",
            llm_small_model="openai/gpt-oss-20b",
            **_HOSTED,
        )

        assert settings.hosted_llm_roles == ["small"]


class TestRoutingWiring:
    async def test_one_provider_serves_both_models_with_separate_circuits_and_prices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aegis.config import Settings
        from aegis.interface.api import dependencies

        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            llm_provider="openai_compatible",
            llm_routing="heuristic",
            llm_small_model="openai/gpt-oss-20b",
            llm_input_price_per_mtok=0.15,
            llm_small_input_price_per_mtok=0.05,
            **_HOSTED,
        )
        monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
        dependencies.get_llm_client.cache_clear()
        try:
            client = dependencies.get_llm_client()
            assert isinstance(client, RoutingLLMClient)
            assert client.breaker.name == "api.groq.com/openai/gpt-oss-20b"
            prices = dependencies._hosted_llm_prices()  # noqa: SLF001
            assert prices["api.groq.com"].input_per_million == 0.15
            assert prices[("api.groq.com", "openai/gpt-oss-20b")].input_per_million == 0.05
            await client.aclose()
        finally:
            dependencies.get_llm_client.cache_clear()

    def test_routing_off_leaves_the_client_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aegis.config import Settings
        from aegis.interface.api import dependencies

        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
        dependencies.get_llm_client.cache_clear()
        try:
            assert not isinstance(dependencies.get_llm_client(), RoutingLLMClient)
        finally:
            dependencies.get_llm_client.cache_clear()
