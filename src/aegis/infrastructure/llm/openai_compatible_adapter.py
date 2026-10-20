"""
Adapter for hosted LLMs that expose the OpenAI Chat Completions API.

One adapter covers every provider that speaks this wire format — Groq,
OpenRouter, Mistral, Together, a self-hosted vLLM or LiteLLM gateway, OpenAI
itself — so switching provider is a matter of three settings (base URL, model,
API key) rather than new code.

Trade-off, stated plainly: selecting this adapter ends the air-gap. The query and
the retrieved document excerpts are sent to a third party. It is therefore opt-in
(``LLM_PROVIDER=openai_compatible``); the default remains the local Ollama
backend. Everything else about the pipeline is unchanged — the input gateway
still runs before retrieval, the same hardened system prompt is sent, and the
output sanitizer still post-processes whatever comes back.

Handling of the credential:
  - It is held only as the client's Authorization header. It is never logged and
    never included in an exception message, so it cannot surface through the
    error path or a traceback.
  - A rejected key is reported as LLMUnavailableError like any other upstream
    failure: callers get a 503, and the reason goes to the server log only.
"""

from __future__ import annotations

import time

import httpx
import structlog

from aegis.domain.models.document import Document
from aegis.domain.models.generation import Generation, TokenUsage
from aegis.domain.ports.llm_client import LLMClientPort, LLMTimeoutError, LLMUnavailableError
from aegis.infrastructure.llm.prompting import SYSTEM_PROMPT, build_user_message

logger = structlog.get_logger(__name__)

# /ready is unauthenticated and polled by orchestrators. Without a cache every
# probe would become an outbound, rate-limited, billable call to the provider,
# and anyone able to reach /ready could drain the quota. One real check per
# interval is plenty to notice an outage.
_HEALTH_CACHE_SECONDS = 30.0


class OpenAICompatibleAdapter(LLMClientPort):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 60,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        # Host only, for logs: enough to tell providers apart without the path.
        self._provider_host = httpx.URL(base_url).host
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )
        self._health: tuple[float, bool] | None = None

    async def generate(
        self,
        query: str,
        context: list[Document],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Generation:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(query, context)},
            ],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.debug(
            "llm.request",
            provider=self._provider_host,
            model=self._model,
            context_docs=len(context),
        )
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"{self._provider_host} did not answer in time ({type(exc).__name__})"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"request to {self._provider_host} failed ({type(exc).__name__})"
            ) from exc

        if response.status_code != httpx.codes.OK:
            raise LLMUnavailableError(self._describe_failure(response))

        try:
            body = response.json()
            answer = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError(
                f"{self._provider_host} returned an unexpected response body"
            ) from exc
        if not isinstance(answer, str) or not answer.strip():
            # Reasoning models can spend the whole token budget before producing
            # visible text; an empty answer is a failed generation, not a result.
            raise LLMUnavailableError(f"{self._provider_host} returned no text content")

        usage = _openai_usage(body)
        logger.info(
            "llm.response_received",
            provider=self._provider_host,
            model=self._model,
            words=len(answer.split()),
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
        )
        return Generation(text=answer, provider=self._provider_host, model=self._model, usage=usage)

    def _describe_failure(self, response: httpx.Response) -> str:
        """
        A log-safe summary of a non-200 reply.

        Only the status and the provider's own error ``code``/``type`` are used.
        The free-text ``message`` is left out on purpose: providers echo request
        details in it, and this string ends up in logs.
        """
        status = response.status_code
        try:
            error = response.json().get("error", {})
            reason = str(error.get("code") or error.get("type") or "")[:80]
        except (ValueError, AttributeError):
            reason = ""

        if status in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            kind = "rejected the API key"
        elif status == httpx.codes.TOO_MANY_REQUESTS:
            kind = "is rate limiting this key"
        elif status == httpx.codes.NOT_FOUND:
            kind = f"does not serve model {self._model!r}"
        else:
            kind = "returned an error"
        suffix = f": {reason}" if reason else ""
        return f"{self._provider_host} {kind} (HTTP {status}){suffix}"

    async def health_check(self) -> bool:
        now = time.monotonic()
        if self._health is not None and now - self._health[0] < _HEALTH_CACHE_SECONDS:
            return self._health[1]

        try:
            # Listing models exercises DNS, TLS and the credential without
            # spending generation quota.
            response = await self._client.get("/models", timeout=5.0)
            healthy = response.status_code == httpx.codes.OK
        except httpx.HTTPError:
            healthy = False

        self._health = (now, healthy)
        return healthy

    async def aclose(self) -> None:
        await self._client.aclose()


def _openai_usage(body: object) -> TokenUsage | None:
    """
    Token counts from the ``usage`` object of a /chat/completions reply.

    Every provider this adapter targets sends it, but it is optional in the
    wire format, so its absence is reported as no usage rather than an error:
    the answer is still good.
    """
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
