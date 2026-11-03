"""
LLM client port (abstract interface).

Decouples the application layer from any specific LLM backend.
Concrete adapters: OllamaAdapter (local, the default) and
OpenAICompatibleAdapter (any hosted provider speaking the OpenAI wire format).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from aegis.domain.models.document import Document
from aegis.domain.models.generation import Generation, GenerationEvent, TextDelta


class LLMUnavailableError(Exception):
    """The LLM backend could not produce an answer (unreachable, erroring, malformed reply)."""


class LLMTimeoutError(LLMUnavailableError):
    """The LLM backend was reached but did not answer within the configured timeout."""


class LLMRateLimitedError(LLMUnavailableError):
    """
    The backend refused the request because the caller is over its quota.

    ``retry_after`` is how long the provider asked callers to wait, in seconds,
    when it said; None when it did not. Callers that route around a busy backend
    use it to decide how long to leave it alone.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMClientPort(ABC):
    """Hexagonal port for LLM text generation."""

    @abstractmethod
    async def generate(
        self,
        query: str,
        context: list[Document],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Generation:
        """
        Generates a grounded answer for query using context as retrieved evidence.
        Implementations are responsible for building a safe system prompt that
        instructs the model to answer only from the provided context (LLM09, misinformation).

        Returns the answer together with the provider, the model and the token
        usage the backend reported, or ``usage=None`` if it reported none.

        Raises:
            LLMTimeoutError:     the backend did not answer in time.
            LLMUnavailableError: any other backend failure. Adapters must translate
                                 their client library's exceptions into these so no
                                 vendor type crosses the port.
        """

    async def stream(
        self,
        query: str,
        context: list[Document],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncGenerator[GenerationEvent, None]:
        """
        Generates the same answer as ``generate``, delivered as it is produced:
        ``TextDelta`` items in order, then exactly one ``Generation`` carrying the
        whole text and the usage.

        Errors are those of ``generate``. A failure can arrive after some deltas
        have been yielded; callers decide what that means for text already sent.

        The default generates in one piece and yields it as a single delta, so a
        backend without streaming still works with streaming callers. Adapters
        that can stream override it. Closing the iterator early (a client that
        disconnects) must release the upstream request.
        """
        generation = await self.generate(
            query, context, max_tokens=max_tokens, temperature=temperature
        )
        yield TextDelta(generation.text)
        yield generation

    @abstractmethod
    async def health_check(self) -> bool:
        """Returns True if the LLM backend is reachable and the model is loaded."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook, deliberately not abstract
        """
        Releases any resources held by the client (connection pools, sessions).
        Called once at shutdown. The default does nothing, so implementations
        with nothing to release need not override it.
        """
