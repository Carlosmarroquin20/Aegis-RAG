"""
LLM client port (abstract interface).

Decouples the application layer from any specific LLM backend.
Concrete adapters: OllamaAdapter (local, the default) and
OpenAICompatibleAdapter (any hosted provider speaking the OpenAI wire format).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aegis.domain.models.document import Document


class LLMUnavailableError(Exception):
    """The LLM backend could not produce an answer (unreachable, erroring, malformed reply)."""


class LLMTimeoutError(LLMUnavailableError):
    """The LLM backend was reached but did not answer within the configured timeout."""


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
    ) -> str:
        """
        Generates a grounded answer for query using context as retrieved evidence.
        Implementations are responsible for building a safe system prompt that
        instructs the model to answer only from the provided context (LLM02 mitigation).

        Raises:
            LLMTimeoutError:     the backend did not answer in time.
            LLMUnavailableError: any other backend failure. Adapters must translate
                                 their client library's exceptions into these so no
                                 vendor type crosses the port.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Returns True if the LLM backend is reachable and the model is loaded."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook, deliberately not abstract
        """
        Releases any resources held by the client (connection pools, sessions).
        Called once at shutdown. The default does nothing, so implementations
        with nothing to release need not override it.
        """
