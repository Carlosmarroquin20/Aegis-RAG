"""
Cross-encoder reranker, run locally with SentenceTransformers.

A bi-encoder (the embedding model) turns the query and each chunk into vectors
independently and compares them; it never sees the two together. A cross-encoder
reads the query and one chunk as a single input and scores how well that chunk
answers it, which is markedly more accurate and far too slow to run over a whole
collection. Run over the twenty-odd candidates the first stage returns, it costs
a fraction of the LLM call that follows.

The default model, ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``, is a
multilingual MiniLM fine-tuned on mMARCO (MS MARCO machine-translated into
fourteen languages, Spanish included), the same size class as the embedding
model. It loads without ``trust_remote_code``: a reranker that needs to execute
code shipped with its weights is refused here rather than configured.

Like the vector store adapter, every call into the model runs on a worker thread
so CPU-bound inference never stalls the event loop.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any

import structlog

from aegis.domain.models.document import Document
from aegis.domain.ports.retrieval import RerankerPort

logger = structlog.get_logger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CrossEncoderReranker(RerankerPort):
    """
    Wraps ``sentence_transformers.CrossEncoder``. The model loads on first use, or
    up front with ``load()`` so the first query does not pay for it.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    async def load(self) -> None:
        """Loads the model if it is not loaded yet. Safe to call more than once."""
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._build)
                logger.info("reranker.loaded", model=self._model_name, device=self._device)

    def _build(self) -> Any:
        from sentence_transformers import CrossEncoder  # Deferred: imports torch.

        return CrossEncoder(
            self._model_name,
            device=self._device,
            max_length=self._max_length,
            trust_remote_code=False,
        )

    async def rerank(self, query: str, documents: list[Document], k: int) -> list[Document]:
        if not documents or k < 1:
            return []
        await self.load()
        pairs = [(query, document.content) for document in documents]
        raw = await asyncio.to_thread(self._predict, pairs)
        scores = _as_probabilities([float(score) for score in raw])
        # Stable sort: equal scores keep the first stage's order.
        ranked = sorted(range(len(documents)), key=lambda i: -scores[i])[:k]
        return [
            documents[i].model_copy(update={"relevance_score": round(scores[i], 4)}) for i in ranked
        ]

    def _predict(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        assert self._model is not None
        result: Sequence[float] = self._model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )
        return result


def _as_probabilities(scores: list[float]) -> list[float]:
    """
    Puts scores in [0, 1] for ``relevance_score``. Most single-label
    cross-encoders already apply a sigmoid; some ship configured to return raw
    logits. Either way the order is the same, so only out-of-range values are
    squashed.
    """
    if all(0.0 <= score <= 1.0 for score in scores):
        return scores
    return [_sigmoid(score) for score in scores]


def _sigmoid(x: float) -> float:
    """Logistic function, written so a large negative logit cannot overflow."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
