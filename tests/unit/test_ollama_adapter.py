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
        answer = await adapter.generate("What is X?", [_doc("1", "X is a thing.")])
        await adapter.aclose()

        assert answer == "Grounded answer."
        assert route.called

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

    @respx.mock
    async def test_raises_on_http_error(self) -> None:
        respx.post(f"{_BASE}/api/chat").mock(return_value=httpx.Response(500))
        adapter = OllamaAdapter(base_url=_BASE, model="llama3.2")
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.generate("q", [])
        await adapter.aclose()


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
