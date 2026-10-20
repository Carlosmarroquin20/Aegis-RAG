"""
Unit tests for the Ollama LLM adapter.

HTTPX calls are mocked with respx, so these run without a live Ollama instance
while still exercising the real request/response handling.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from aegis.domain.models.document import Document
from aegis.domain.models.generation import TokenUsage
from aegis.domain.ports.llm_client import LLMTimeoutError, LLMUnavailableError
from aegis.infrastructure.llm.ollama_adapter import OllamaAdapter

_BASE = "http://ollama-test:11434"


def _doc(i: str, content: str, source: str = "src.md") -> Document:
    return Document(id=i, content=content, metadata={"source": source}, relevance_score=0.9)


# ── generate ─────────────────────────────────────────────────────────────────


class TestOllamaGenerate:
    @respx.mock
    async def test_returns_message_content(self) -> None:
        route = respx.post(f"{_BASE}/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": "Grounded answer."}})
        )
        adapter = OllamaAdapter(base_url=_BASE, model="llama3.2")
        generation = await adapter.generate("What is X?", [_doc("1", "X is a thing.")])
        await adapter.aclose()

        assert generation.text == "Grounded answer."
        assert (generation.provider, generation.model) == ("ollama", "llama3.2")
        assert route.called

    @respx.mock
    @pytest.mark.parametrize(
        ("counts", "usage"),
        [
            ({"prompt_eval_count": 210, "eval_count": 42}, TokenUsage(210, 42)),
            # Ollama omits the prompt count when the prompt came from its cache.
            ({"eval_count": 42}, TokenUsage(0, 42)),
            ({}, None),
            ({"prompt_eval_count": "210", "eval_count": 42}, None),
        ],
        ids=["both", "cached-prompt", "absent", "malformed"],
    )
    async def test_reports_token_usage(
        self, counts: dict[str, object], usage: TokenUsage | None
    ) -> None:
        body = {"message": {"content": "ok"}, **counts}
        respx.post(f"{_BASE}/api/chat").mock(return_value=httpx.Response(200, json=body))
        adapter = OllamaAdapter(base_url=_BASE, model="llama3.2")
        generation = await adapter.generate("q", [])
        await adapter.aclose()

        assert generation.usage == usage

    @respx.mock
    async def test_sends_system_prompt_and_context(self) -> None:
        route = respx.post(f"{_BASE}/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": "ok"}})
        )
        adapter = OllamaAdapter(base_url=_BASE, model="llama3.2")
        await adapter.generate("question?", [_doc("1", "alpha"), _doc("2", "beta")])
        await adapter.aclose()

        payload = route.calls.last.request.read().decode()
        assert '"role":"system"' in payload
        assert "alpha" in payload and "beta" in payload
        assert "question?" in payload


class TestOllamaFailuresAreTranslated:
    """No httpx exception may cross the port; callers only know the domain errors."""

    @staticmethod
    async def _generate() -> object:
        adapter = OllamaAdapter(base_url=_BASE, model="llama3.2")
        try:
            return await adapter.generate("q", [])
        finally:
            await adapter.aclose()

    @respx.mock
    @pytest.mark.parametrize("status_code", [404, 500, 503])
    async def test_error_status_is_unavailable(self, status_code: int) -> None:
        respx.post(f"{_BASE}/api/chat").mock(return_value=httpx.Response(status_code))
        with pytest.raises(LLMUnavailableError) as excinfo:
            await self._generate()
        assert not isinstance(excinfo.value, LLMTimeoutError)

    @respx.mock
    async def test_connection_failure_is_unavailable(self) -> None:
        respx.post(f"{_BASE}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(LLMUnavailableError):
            await self._generate()

    @respx.mock
    async def test_timeout_is_reported_as_timeout(self) -> None:
        respx.post(f"{_BASE}/api/chat").mock(side_effect=httpx.ReadTimeout("slow model"))
        with pytest.raises(LLMTimeoutError):
            await self._generate()

    @respx.mock
    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, text="<html>proxy error</html>"),
            httpx.Response(200, json={"error": "model not loaded"}),
            httpx.Response(200, json={"message": {"content": None}}),
            httpx.Response(200, json=["unexpected"]),
        ],
        ids=["not-json", "missing-message", "null-content", "wrong-shape"],
    )
    async def test_malformed_success_body_is_unavailable(self, response: httpx.Response) -> None:
        respx.post(f"{_BASE}/api/chat").mock(return_value=response)
        with pytest.raises(LLMUnavailableError):
            await self._generate()


# ── health_check ─────────────────────────────────────────────────────────────


class TestOllamaHealthCheck:
    @respx.mock
    async def test_healthy_when_tags_ok(self) -> None:
        respx.get(f"{_BASE}/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))
        adapter = OllamaAdapter(base_url=_BASE, model="llama3.2")
        assert await adapter.health_check() is True
        await adapter.aclose()

    @respx.mock
    async def test_unhealthy_on_connection_error(self) -> None:
        respx.get(f"{_BASE}/api/tags").mock(side_effect=httpx.ConnectError("down"))
        adapter = OllamaAdapter(base_url=_BASE, model="llama3.2")
        assert await adapter.health_check() is False
        await adapter.aclose()


# ── _format_context ──────────────────────────────────────────────────────────


class TestFormatContext:
    def test_no_documents_returns_placeholder(self) -> None:
        assert "No context" in OllamaAdapter._format_context([])

    def test_numbers_and_attributes_documents(self) -> None:
        block = OllamaAdapter._format_context(
            [_doc("1", "first", "a.md"), _doc("2", "second", "b.md")]
        )
        assert "[1]" in block and "[2]" in block
        assert "a.md" in block and "b.md" in block
        assert "first" in block and "second" in block
