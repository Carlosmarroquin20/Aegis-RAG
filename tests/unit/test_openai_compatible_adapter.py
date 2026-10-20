"""
Unit tests for the OpenAI-compatible LLM adapter.

HTTP is mocked with respx, so no provider is contacted and no real key exists in
the suite. Beyond request/response mapping, the tests pin down the two things
that make a hosted backend risky: the credential must never leak through the
error path, and no client-library exception may cross the port.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from aegis.config import Settings
from aegis.domain.models.document import Document
from aegis.domain.models.generation import Generation, TokenUsage
from aegis.domain.ports.llm_client import LLMTimeoutError, LLMUnavailableError
from aegis.infrastructure.llm import openai_compatible_adapter as adapter_module
from aegis.infrastructure.llm.ollama_adapter import OllamaAdapter
from aegis.infrastructure.llm.openai_compatible_adapter import OpenAICompatibleAdapter
from aegis.infrastructure.llm.prompting import SYSTEM_PROMPT

_BASE = "https://llm.example.test/v1"
_KEY = "sk-unit-test-secret-value"
_MODEL = "vendor/some-model"
_CHAT = f"{_BASE}/chat/completions"
_DOCS = [Document(id="1", content="Remote work: two days.", metadata={"source": "policy.md"})]


def _ok(content: object = "The answer.", **extra: object) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}], **extra})


async def _generate(query: str = "q", docs: list[Document] | None = None) -> Generation:
    adapter = OpenAICompatibleAdapter(_BASE, _KEY, _MODEL)
    try:
        return await adapter.generate(query, docs if docs is not None else _DOCS)
    finally:
        await adapter.aclose()


class TestRequest:
    @respx.mock
    async def test_returns_the_message_content(self) -> None:
        respx.post(_CHAT).mock(return_value=_ok("Two days per week."))
        generation = await _generate()
        assert generation.text == "Two days per week."
        assert (generation.provider, generation.model) == ("llm.example.test", _MODEL)

    @respx.mock
    @pytest.mark.parametrize(
        ("usage", "expected"),
        [
            (
                {"prompt_tokens": 512, "completion_tokens": 64, "total_tokens": 576},
                TokenUsage(512, 64),
            ),
            (None, None),
            ({"prompt_tokens": 512}, None),
        ],
        ids=["reported", "absent", "partial"],
    )
    async def test_reports_token_usage(
        self, usage: dict[str, int] | None, expected: TokenUsage | None
    ) -> None:
        extra = {"usage": usage} if usage is not None else {}
        respx.post(_CHAT).mock(return_value=_ok("ok", **extra))
        assert (await _generate()).usage == expected

    @respx.mock
    async def test_sends_bearer_key_model_and_generation_parameters(self) -> None:
        route = respx.post(_CHAT).mock(return_value=_ok())
        adapter = OpenAICompatibleAdapter(_BASE, _KEY, _MODEL)
        await adapter.generate("q", _DOCS, max_tokens=256, temperature=0.3)
        await adapter.aclose()

        request = route.calls.last.request
        assert request.headers["authorization"] == f"Bearer {_KEY}"
        body = json.loads(request.content)
        assert body["model"] == _MODEL
        assert body["max_tokens"] == 256
        assert body["temperature"] == 0.3
        assert body["stream"] is False

    @respx.mock
    async def test_uses_the_shared_hardened_prompt_and_context_framing(self) -> None:
        """A hosted backend must get exactly the same injection defence as the local one."""
        route = respx.post(_CHAT).mock(return_value=_ok())
        await _generate("How many days?")

        system, user = json.loads(route.calls.last.request.content)["messages"]
        assert system == {"role": "system", "content": SYSTEM_PROMPT}
        assert user["role"] == "user"
        assert "[1] Source: policy.md\nRemote work: two days." in user["content"]
        assert user["content"].endswith("QUESTION: How many days?")

    @respx.mock
    async def test_prompt_is_identical_to_the_one_ollama_receives(self) -> None:
        hosted = respx.post(_CHAT).mock(return_value=_ok())
        local = respx.post("http://ollama.test/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": "x"}})
        )
        await _generate("same question")
        ollama = OllamaAdapter("http://ollama.test", "llama3.2")
        await ollama.generate("same question", _DOCS)
        await ollama.aclose()

        assert (
            json.loads(hosted.calls.last.request.content)["messages"]
            == json.loads(local.calls.last.request.content)["messages"]
        )

    @respx.mock
    async def test_trailing_slash_in_base_url_is_tolerated(self) -> None:
        route = respx.post(_CHAT).mock(return_value=_ok())
        adapter = OpenAICompatibleAdapter(_BASE + "/", _KEY, _MODEL)
        await adapter.generate("q", [])
        await adapter.aclose()
        assert route.called


class TestFailuresAreTranslated:
    @respx.mock
    @pytest.mark.parametrize(
        ("status", "fragment"),
        [
            (401, "rejected the API key"),
            (403, "rejected the API key"),
            (404, "does not serve model"),
            (429, "rate limiting"),
            (500, "returned an error"),
            (503, "returned an error"),
        ],
    )
    async def test_error_status_becomes_unavailable(self, status: int, fragment: str) -> None:
        respx.post(_CHAT).mock(return_value=httpx.Response(status, json={"error": {}}))
        with pytest.raises(LLMUnavailableError, match=fragment) as excinfo:
            await _generate()
        assert not isinstance(excinfo.value, LLMTimeoutError)
        assert f"HTTP {status}" in str(excinfo.value)

    @respx.mock
    async def test_timeout_is_reported_as_timeout(self) -> None:
        respx.post(_CHAT).mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(LLMTimeoutError):
            await _generate()

    @respx.mock
    async def test_connection_failure_is_unavailable(self) -> None:
        respx.post(_CHAT).mock(side_effect=httpx.ConnectError("dns"))
        with pytest.raises(LLMUnavailableError):
            await _generate()

    @respx.mock
    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, text="<html>gateway</html>"),
            httpx.Response(200, json={"choices": []}),
            httpx.Response(200, json={"choices": [{"message": {}}]}),
            httpx.Response(200, json={"unexpected": True}),
            _ok(None),
            _ok(""),
            _ok("   \n"),
            _ok(["not", "text"]),
        ],
        ids=[
            "not-json",
            "no-choices",
            "no-content",
            "wrong-shape",
            "null-content",
            "empty-content",
            "blank-content",
            "non-text-content",
        ],
    )
    async def test_unusable_success_body_is_unavailable(self, response: httpx.Response) -> None:
        respx.post(_CHAT).mock(return_value=response)
        with pytest.raises(LLMUnavailableError):
            await _generate()


class TestCredentialDoesNotLeak:
    """The error text reaches server logs; nothing in it may identify the key."""

    @respx.mock
    @pytest.mark.parametrize("status", [401, 429, 500])
    async def test_key_is_absent_from_error_messages(self, status: int) -> None:
        # Providers echo request details; simulate one echoing the key back.
        echoed = {"error": {"message": f"Invalid key {_KEY} supplied", "code": "invalid_api_key"}}
        respx.post(_CHAT).mock(return_value=httpx.Response(status, json=echoed))

        with pytest.raises(LLMUnavailableError) as excinfo:
            await _generate()

        assert _KEY not in str(excinfo.value)
        assert _KEY not in repr(excinfo.value)
        assert "invalid_api_key" in str(excinfo.value)  # the code is safe and useful

    @respx.mock
    async def test_key_is_absent_from_transport_error_messages(self) -> None:
        respx.post(_CHAT).mock(side_effect=httpx.ConnectError(f"failed with header Bearer {_KEY}"))
        with pytest.raises(LLMUnavailableError) as excinfo:
            await _generate()
        assert _KEY not in str(excinfo.value)

    def test_key_is_not_exposed_by_settings_repr(self) -> None:
        settings = Settings(llm_api_key=_KEY)
        assert _KEY not in repr(settings)
        assert _KEY not in str(settings.llm_api_key)
        assert settings.llm_api_key.get_secret_value() == _KEY


class TestHealthCheck:
    @respx.mock
    async def test_healthy_when_models_endpoint_answers(self) -> None:
        respx.get(f"{_BASE}/models").mock(return_value=httpx.Response(200, json={"data": []}))
        adapter = OpenAICompatibleAdapter(_BASE, _KEY, _MODEL)
        assert await adapter.health_check() is True
        await adapter.aclose()

    @respx.mock
    @pytest.mark.parametrize(
        "outcome", [httpx.Response(401), httpx.Response(503), httpx.ConnectError("down")]
    )
    async def test_unhealthy_on_rejected_key_error_or_no_connection(
        self, outcome: httpx.Response | Exception
    ) -> None:
        route = respx.get(f"{_BASE}/models")
        if isinstance(outcome, Exception):
            route.mock(side_effect=outcome)
        else:
            route.mock(return_value=outcome)
        adapter = OpenAICompatibleAdapter(_BASE, _KEY, _MODEL)
        assert await adapter.health_check() is False
        await adapter.aclose()

    @respx.mock
    async def test_result_is_cached_so_probes_do_not_spend_provider_quota(self) -> None:
        """/ready is unauthenticated; each probe must not become an outbound call."""
        route = respx.get(f"{_BASE}/models").mock(return_value=httpx.Response(200, json={}))
        adapter = OpenAICompatibleAdapter(_BASE, _KEY, _MODEL)
        for _ in range(25):
            assert await adapter.health_check() is True
        await adapter.aclose()
        assert route.call_count == 1

    @respx.mock
    async def test_cache_expires(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = [100.0]
        monkeypatch.setattr(adapter_module.time, "monotonic", lambda: now[0])
        route = respx.get(f"{_BASE}/models").mock(return_value=httpx.Response(200, json={}))
        adapter = OpenAICompatibleAdapter(_BASE, _KEY, _MODEL)

        await adapter.health_check()
        now[0] += adapter_module._HEALTH_CACHE_SECONDS + 1
        await adapter.health_check()
        await adapter.aclose()
        assert route.call_count == 2


class TestProviderSettings:
    def test_default_provider_is_the_local_one(self) -> None:
        assert Settings(_env_file=None).llm_provider == "ollama"  # type: ignore[call-arg]

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="llm_provider"):
            Settings(llm_provider="bedrock")

    @pytest.mark.parametrize(
        ("overrides", "missing"),
        [
            ({"llm_model": "m", "llm_api_key": "k"}, "LLM_BASE_URL"),
            ({"llm_base_url": "https://x.test/v1", "llm_api_key": "k"}, "LLM_MODEL"),
            ({"llm_base_url": "https://x.test/v1", "llm_model": "m"}, "LLM_API_KEY"),
        ],
    )
    def test_hosted_provider_requires_its_settings_at_startup(
        self, overrides: dict[str, str], missing: str
    ) -> None:
        with pytest.raises(ValueError, match=missing):
            Settings(_env_file=None, llm_provider="openai_compatible", **overrides)  # type: ignore[arg-type]

    def test_plain_http_base_url_is_refused_outside_debug(self) -> None:
        """The key is sent on every request; it must not travel in clear text."""
        values = {"llm_base_url": "http://x.test/v1", "llm_model": "m", "llm_api_key": "k"}
        with pytest.raises(ValueError, match="https"):
            Settings(_env_file=None, llm_provider="openai_compatible", **values)  # type: ignore[arg-type]
        # Allowed in debug, e.g. for a local gateway during development.
        Settings(_env_file=None, llm_provider="openai_compatible", debug=True, **values)  # type: ignore[arg-type]

    def test_hosted_settings_are_not_required_for_ollama(self) -> None:
        assert Settings(_env_file=None, llm_provider="ollama").llm_model == ""  # type: ignore[call-arg]


class TestWiring:
    @pytest.mark.parametrize(
        ("settings", "expected"),
        [
            (Settings(_env_file=None), OllamaAdapter),  # type: ignore[call-arg]
            (
                Settings(
                    _env_file=None,  # type: ignore[call-arg]
                    llm_provider="openai_compatible",
                    llm_base_url=_BASE,
                    llm_model=_MODEL,
                    llm_api_key=_KEY,
                ),
                OpenAICompatibleAdapter,
            ),
        ],
        ids=["ollama", "openai_compatible"],
    )
    async def test_provider_setting_selects_the_adapter(
        self, monkeypatch: pytest.MonkeyPatch, settings: Settings, expected: type
    ) -> None:
        from aegis.interface.api import dependencies

        monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
        dependencies.get_llm_client.cache_clear()
        try:
            client = dependencies.get_llm_client()
            assert type(client) is expected
            await client.aclose()
        finally:
            dependencies.get_llm_client.cache_clear()
