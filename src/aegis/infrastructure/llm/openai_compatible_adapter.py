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

import json
import time
from collections.abc import AsyncGenerator
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import structlog

from aegis.domain.models.document import Document
from aegis.domain.models.generation import Generation, GenerationEvent, TextDelta, TokenUsage
from aegis.domain.ports.llm_client import (
    LLMClientPort,
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
)
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

    def _payload(
        self, query: str, context: list[Document], max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(query, context)},
            ],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    async def generate(
        self,
        query: str,
        context: list[Document],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Generation:
        payload = self._payload(query, context, max_tokens, temperature)

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

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise LLMRateLimitedError(
                self._describe_failure(response), retry_after=_retry_after(response)
            )
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

    async def stream(
        self,
        query: str,
        context: list[Document],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncGenerator[GenerationEvent, None]:
        """
        Streams /chat/completions as server-sent events. ``include_usage`` asks
        for a last chunk carrying the token counts, which every provider this
        adapter targets supports; without it a streamed answer could not be billed.
        Failures before the first byte keep their non-streaming meaning (a 429 is
        still LLMRateLimitedError with its Retry-After).
        """
        payload = self._payload(query, context, max_tokens, temperature)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        parts: list[str] = []
        usage: TokenUsage | None = None
        logger.debug(
            "llm.request",
            provider=self._provider_host,
            model=self._model,
            context_docs=len(context),
            stream=True,
        )
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code != httpx.codes.OK:
                    await response.aread()
                    if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
                        raise LLMRateLimitedError(
                            self._describe_failure(response), retry_after=_retry_after(response)
                        )
                    raise LLMUnavailableError(self._describe_failure(response))
                async for line in response.aiter_lines():
                    data = line.removeprefix("data:").strip() if line.startswith("data:") else ""
                    if not data:
                        continue  # blank separators, comments, other SSE fields
                    if data == "[DONE]":
                        break
                    chunk = self._parse_chunk(data)
                    usage = _openai_usage(chunk) or usage
                    content = _delta_content(chunk)
                    if content:
                        parts.append(content)
                        yield TextDelta(content)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"{self._provider_host} did not answer in time ({type(exc).__name__})"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"request to {self._provider_host} failed ({type(exc).__name__})"
            ) from exc

        answer = "".join(parts)
        if not answer.strip():
            raise LLMUnavailableError(f"{self._provider_host} returned no text content")
        logger.info(
            "llm.response_received",
            provider=self._provider_host,
            model=self._model,
            words=len(answer.split()),
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            stream=True,
        )
        yield Generation(text=answer, provider=self._provider_host, model=self._model, usage=usage)

    def _parse_chunk(self, data: str) -> dict[str, Any]:
        try:
            chunk = json.loads(data)
        except ValueError as exc:
            raise LLMUnavailableError(
                f"{self._provider_host} streamed an unreadable event"
            ) from exc
        if not isinstance(chunk, dict):
            raise LLMUnavailableError(f"{self._provider_host} streamed an unexpected event")
        if "error" in chunk:
            raise LLMUnavailableError(f"{self._provider_host} reported an error mid-stream")
        return chunk

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


def _delta_content(chunk: dict[str, Any]) -> str:
    """The text in a streamed chunk; the usage chunk has no choices at all."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    delta = choices[0].get("delta")
    content = delta.get("content") if isinstance(delta, dict) else None
    return content if isinstance(content, str) else ""


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


def _retry_after(response: httpx.Response) -> float | None:
    """
    Seconds to wait, from a Retry-After header in either form RFC 9110 allows:
    a number of seconds or an HTTP date. None when absent or unreadable.
    """
    value = response.headers.get("retry-after", "").strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())
