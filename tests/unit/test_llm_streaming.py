"""
Streaming generation through the LLM port: the default for backends that cannot
stream, both adapters' wire formats, and the fallback's rule that a backend can
only be skipped before it has streamed any text.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import httpx
import pytest
import respx

from aegis.domain.models.document import Document
from aegis.domain.models.generation import Generation, GenerationEvent, TextDelta, TokenUsage
from aegis.domain.ports.llm_client import (
    LLMClientPort,
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from aegis.infrastructure.llm.fallback import Backend, CircuitState, FallbackLLMClient
from aegis.infrastructure.llm.ollama_adapter import OllamaAdapter
from aegis.infrastructure.llm.openai_compatible_adapter import OpenAICompatibleAdapter
from tests.conftest import FakeLLMClient

_DOCS = [Document(id="1", content="X is a thing.", metadata={"source": "x.md"})]
_OLLAMA = "http://ollama-test:11434"
_HOSTED = "https://llm.example.test/v1"


async def _collect(events: AsyncGenerator[GenerationEvent, None]) -> list[GenerationEvent]:
    return [event async for event in events]


def _text(events: list[GenerationEvent]) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


def _final(events: list[GenerationEvent]) -> Generation:
    assert isinstance(events[-1], Generation)
    return events[-1]


class TestDefaultStream:
    async def test_a_backend_without_streaming_yields_one_delta_then_the_generation(self) -> None:
        events = await _collect(FakeLLMClient(fixed_response="Whole answer.").stream("q", _DOCS))

        assert [type(e) for e in events] == [TextDelta, Generation]
        assert _text(events) == "Whole answer."
        assert _final(events).usage == TokenUsage(120, 30)


def _ndjson(*objects: object) -> bytes:
    return b"".join(json.dumps(o).encode() + b"\n" for o in objects)


class TestOllamaStream:
    @respx.mock
    async def test_streams_pieces_then_the_usage(self) -> None:
        route = respx.post(f"{_OLLAMA}/api/chat").mock(
            return_value=httpx.Response(
                200,
                content=_ndjson(
                    {"message": {"content": "Hola, "}, "done": False},
                    {"message": {"content": "mundo."}, "done": False},
                    {
                        "message": {"content": ""},
                        "done": True,
                        "prompt_eval_count": 9,
                        "eval_count": 4,
                    },
                ),
            )
        )
        adapter = OllamaAdapter(base_url=_OLLAMA, model="llama3.2")

        events = await _collect(adapter.stream("q", _DOCS))
        await adapter.aclose()

        assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hola, ", "mundo."]
        final = _final(events)
        assert (final.text, final.provider, final.usage) == (
            "Hola, mundo.",
            "ollama",
            TokenUsage(9, 4),
        )
        assert json.loads(route.calls.last.request.content)["stream"] is True

    @respx.mock
    async def test_an_error_status_is_unavailable(self) -> None:
        respx.post(f"{_OLLAMA}/api/chat").mock(return_value=httpx.Response(404, text="no model"))
        adapter = OllamaAdapter(base_url=_OLLAMA, model="llama3.2")

        with pytest.raises(LLMUnavailableError, match="HTTP 404"):
            await _collect(adapter.stream("q", _DOCS))

    @respx.mock
    async def test_an_error_line_mid_stream_is_unavailable(self) -> None:
        respx.post(f"{_OLLAMA}/api/chat").mock(
            return_value=httpx.Response(
                200, content=_ndjson({"message": {"content": "Hola"}}, {"error": "out of memory"})
            )
        )
        adapter = OllamaAdapter(base_url=_OLLAMA, model="llama3.2")

        with pytest.raises(LLMUnavailableError, match="mid-stream"):
            await _collect(adapter.stream("q", _DOCS))

    @respx.mock
    async def test_an_unreadable_line_is_unavailable(self) -> None:
        respx.post(f"{_OLLAMA}/api/chat").mock(return_value=httpx.Response(200, content=b"{oops\n"))
        adapter = OllamaAdapter(base_url=_OLLAMA, model="llama3.2")

        with pytest.raises(LLMUnavailableError, match="unreadable"):
            await _collect(adapter.stream("q", _DOCS))

    @respx.mock
    async def test_a_timeout_is_a_timeout(self) -> None:
        respx.post(f"{_OLLAMA}/api/chat").mock(side_effect=httpx.ReadTimeout("slow"))
        adapter = OllamaAdapter(base_url=_OLLAMA, model="llama3.2")

        with pytest.raises(LLMTimeoutError):
            await _collect(adapter.stream("q", _DOCS))


def _sse(*objects: object, done: bool = True) -> bytes:
    body = b"".join(b"data: " + json.dumps(o).encode() + b"\n\n" for o in objects)
    return body + (b"data: [DONE]\n\n" if done else b"")


def _hosted() -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter(base_url=_HOSTED, api_key="sk-test", model="m")


class TestOpenAICompatibleStream:
    @respx.mock
    async def test_streams_deltas_then_the_usage_chunk(self) -> None:
        route = respx.post(f"{_HOSTED}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=b": keep-alive\n\n"
                + _sse(
                    {"choices": [{"delta": {"role": "assistant"}}]},
                    {"choices": [{"delta": {"content": "Hola, "}}]},
                    {"choices": [{"delta": {"content": "mundo."}}]},
                    {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 3}},
                ),
            )
        )
        adapter = _hosted()

        events = await _collect(adapter.stream("q", _DOCS))
        await adapter.aclose()

        assert _text(events) == "Hola, mundo."
        final = _final(events)
        assert (final.provider, final.usage) == ("llm.example.test", TokenUsage(11, 3))
        payload = json.loads(route.calls.last.request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}

    @respx.mock
    async def test_a_missing_usage_chunk_is_no_usage(self) -> None:
        respx.post(f"{_HOSTED}/chat/completions").mock(
            return_value=httpx.Response(
                200, content=_sse({"choices": [{"delta": {"content": "ok"}}]})
            )
        )

        events = await _collect(_hosted().stream("q", _DOCS))

        assert _final(events).usage is None

    @respx.mock
    async def test_a_429_keeps_its_retry_after(self) -> None:
        respx.post(f"{_HOSTED}/chat/completions").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "12"}, json={"error": {}})
        )

        with pytest.raises(LLMRateLimitedError) as excinfo:
            await _collect(_hosted().stream("q", _DOCS))

        assert excinfo.value.retry_after == 12

    @respx.mock
    async def test_a_rejected_key_is_unavailable_and_the_key_stays_out(self) -> None:
        respx.post(f"{_HOSTED}/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": {"code": "invalid_api_key"}})
        )

        with pytest.raises(LLMUnavailableError) as excinfo:
            await _collect(_hosted().stream("q", _DOCS))

        assert "rejected the API key" in str(excinfo.value)
        assert "sk-test" not in str(excinfo.value)

    @respx.mock
    async def test_an_empty_answer_is_unavailable(self) -> None:
        respx.post(f"{_HOSTED}/chat/completions").mock(
            return_value=httpx.Response(200, content=_sse({"choices": [{"delta": {}}]}))
        )

        with pytest.raises(LLMUnavailableError, match="no text"):
            await _collect(_hosted().stream("q", _DOCS))

    @respx.mock
    async def test_an_error_event_mid_stream_is_unavailable(self) -> None:
        respx.post(f"{_HOSTED}/chat/completions").mock(
            return_value=httpx.Response(
                200, content=_sse({"choices": [{"delta": {"content": "a"}}]}, {"error": {"x": 1}})
            )
        )

        with pytest.raises(LLMUnavailableError, match="mid-stream"):
            await _collect(_hosted().stream("q", _DOCS))

    @respx.mock
    async def test_a_timeout_is_a_timeout(self) -> None:
        respx.post(f"{_HOSTED}/chat/completions").mock(side_effect=httpx.ConnectTimeout("slow"))

        with pytest.raises(LLMTimeoutError):
            await _collect(_hosted().stream("q", _DOCS))


class _StreamingBackend(LLMClientPort):
    """Streams scripted pieces; can fail before or after its first piece."""

    def __init__(self, name: str, *, fail_before: bool = False, fail_after: bool = False) -> None:
        self.name = name
        self.fail_before = fail_before
        self.fail_after = fail_after
        self.calls = 0
        self.closed_early = False

    async def generate(self, query: str, context: list[Document], **_: object) -> Generation:
        raise NotImplementedError

    async def stream(
        self, query: str, context: list[Document], **_: object
    ) -> AsyncGenerator[GenerationEvent, None]:
        self.calls += 1
        if self.fail_before:
            raise LLMUnavailableError(f"{self.name} down")
        try:
            yield TextDelta(f"from {self.name}. ")
            if self.fail_after:
                raise LLMUnavailableError(f"{self.name} broke")
            yield TextDelta("more. ")
            yield Generation(text=f"from {self.name}. more. ", provider=self.name, model="m")
        except GeneratorExit:
            self.closed_early = True
            raise

    async def health_check(self) -> bool:
        return True


def _fallback(primary: LLMClientPort, secondary: LLMClientPort) -> FallbackLLMClient:
    return FallbackLLMClient(
        [Backend("primary", primary), Backend("secondary", secondary)], failure_threshold=1
    )


class TestFallbackStream:
    async def test_a_failure_before_any_text_falls_back(self) -> None:
        client = _fallback(
            _StreamingBackend("primary", fail_before=True), _StreamingBackend("secondary")
        )

        events = await _collect(client.stream("q", _DOCS))

        assert _final(events).provider == "secondary"
        assert client.breaker("primary").state is CircuitState.OPEN

    async def test_a_failure_after_text_is_raised_not_restarted(self) -> None:
        secondary = _StreamingBackend("secondary")
        client = _fallback(_StreamingBackend("primary", fail_after=True), secondary)

        with pytest.raises(LLMUnavailableError, match="primary broke"):
            await _collect(client.stream("q", _DOCS))

        assert secondary.calls == 0

    async def test_a_caller_that_stops_early_closes_the_backend_stream(self) -> None:
        primary = _StreamingBackend("primary")
        client = _fallback(primary, _StreamingBackend("secondary"))

        events = client.stream("q", _DOCS)
        assert isinstance(await anext(events), TextDelta)
        await events.aclose()

        assert primary.closed_early
        # Text arrived, so the backend is known to work.
        assert client.breaker("primary").state is CircuitState.CLOSED

    async def test_every_backend_failing_before_text_raises_the_last_error(self) -> None:
        client = _fallback(
            _StreamingBackend("primary", fail_before=True),
            _StreamingBackend("secondary", fail_before=True),
        )

        with pytest.raises(LLMUnavailableError, match="secondary down"):
            await _collect(client.stream("q", _DOCS))
