"""
Natural-language-inference model run locally with Hugging Face Transformers.

The model is a sequence classifier over (premise, hypothesis) pairs with the
usual three labels: entailment, neutral, contradiction. Only the entailment
probability is used: a claim is supported when the context entails it, and
"neutral" (the context does not say) is exactly the hallucination the judge
exists to catch.

Label order differs between models, so the entailment index is read from the
model's own configuration rather than assumed. Weights load without
``trust_remote_code``, as with the reranker.

Inference is CPU-bound and runs on a worker thread so it never blocks the event
loop. Premises are truncated to the model's maximum length from the end; the
hypothesis (the claim) is never truncated.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from aegis.domain.ports.faithfulness import EntailmentModelPort

logger = structlog.get_logger(__name__)

DEFAULT_NLI_MODEL = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"


class EntailmentLabelError(RuntimeError):
    """The model's configuration names no entailment label."""


class TransformersNLIModel(EntailmentModelPort):
    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        *,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._loaded: tuple[Any, Any, int] | None = None  # tokenizer, model, entailment index
        self._load_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    async def load(self) -> None:
        """Loads the model if it is not loaded yet. Safe to call more than once."""
        async with self._load_lock:
            if self._loaded is None:
                self._loaded = await asyncio.to_thread(self._build)
                logger.info("nli_model.loaded", model=self._model_name, device=self._device)

    def _build(self) -> tuple[Any, Any, int]:
        # Deferred: importing transformers' model classes imports torch.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name, trust_remote_code=False
        )
        model.to(self._device)
        model.eval()
        return tokenizer, model, entailment_index(model.config.id2label)

    async def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        await self.load()
        return await asyncio.to_thread(self._predict, pairs)

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch

        assert self._loaded is not None
        tokenizer, model, entail = self._loaded
        probabilities: list[float] = []
        with torch.no_grad():
            for start in range(0, len(pairs), self._batch_size):
                batch = pairs[start : start + self._batch_size]
                encoded = tokenizer(
                    [premise for premise, _ in batch],
                    [hypothesis for _, hypothesis in batch],
                    truncation="only_first",
                    max_length=self._max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)
                logits = model(**encoded).logits
                probabilities.extend(torch.softmax(logits, dim=-1)[:, entail].tolist())
        return probabilities


def entailment_index(id2label: dict[int, str]) -> int:
    """The class index whose label is entailment, whatever the model calls it."""
    for index, label in id2label.items():
        if str(label).lower().startswith("entail"):
            return int(index)
    raise EntailmentLabelError(f"no entailment label in {id2label!r}")
