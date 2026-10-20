"""
Ollama LLM adapter.

Sends grounded-generation requests to a local Ollama instance. This is the
default backend and the one that keeps the system air-gapped: no query or
document content leaves the host.

The hardened system prompt and the context framing live in ``prompting`` and are
shared with every other adapter.
"""

from __future__ import annotations

import httpx
import structlog

from aegis.domain.models.document import Document
from aegis.domain.models.generation import Generation, TokenUsage
from aegis.domain.ports.llm_client import LLMClientPort, LLMTimeoutError, LLMUnavailableError
from aegis.infrastructure.llm.prompting import SYSTEM_PROMPT, build_user_message, format_context

logger = structlog.get_logger(__name__)


class OllamaAdapter(LLMClientPort):
    """
    Adapter for Ollama's REST API (/api/chat endpoint).
    Uses httpx for async HTTP with explicit timeouts to prevent connection starvation.
    """

    def __init__(self, base_url: str, model: str, timeout_seconds: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds),
        )

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
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        logger.debug("llm.request", model=self._model, context_docs=len(context))
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama did not answer in time: {exc!r}") from exc
        except httpx.HTTPError as exc:
            # Connection failures and non-2xx statuses (e.g. 404 when the model
            # has not been pulled yet).
            raise LLMUnavailableError(f"Ollama request failed: {exc!r}") from exc

        try:
            body = response.json()
            answer = body["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMUnavailableError("Ollama returned an unexpected response body") from exc
        if not isinstance(answer, str):
            raise LLMUnavailableError("Ollama returned a non-text message content")

        usage = _ollama_usage(body)
        logger.info(
            "llm.response_received",
            model=self._model,
            words=len(answer.split()),
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
        )
        return Generation(text=answer, provider="ollama", model=self._model, usage=usage)

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    # Kept as a method for callers and tests that format context through the adapter.
    _format_context = staticmethod(format_context)

    async def aclose(self) -> None:
        await self._client.aclose()


def _ollama_usage(body: object) -> TokenUsage | None:
    """
    Token counts from an Ollama /api/chat reply.

    ``prompt_eval_count`` is omitted when the prompt was served from Ollama's
    cache, so a missing input count is read as zero rather than as no usage;
    a missing output count means the reply carried no usage at all.
    """
    if not isinstance(body, dict):
        return None
    output_tokens = body.get("eval_count")
    input_tokens = body.get("prompt_eval_count", 0)
    if not isinstance(output_tokens, int) or not isinstance(input_tokens, int):
        return None
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
