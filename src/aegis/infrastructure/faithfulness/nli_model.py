"""
Natural-language-inference model run locally with Hugging Face Transformers.

The model is a sequence classifier over (premise, hypothesis) pairs with the
usual three labels: entailment, neutral, contradiction. The entailment and
contradiction probabilities are returned; "neutral" (the context does not say)
is exactly the hallucination the judge exists to catch, and needs no score.

Label order differs between models, so both indices are read from the model's
own configuration rather than assumed. Weights load without
``trust_remote_code``, as with the reranker.

Inference is CPU-bound and runs on a worker thread so it never blocks the event
loop. Pairs are batched by length, so short sentences are not padded to the
length of the longest premise in the request. Premises are truncated to the
model's maximum length; the hypothesis (the claim) is never truncated.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from aegis.domain.models.faithfulness import Inference
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
        # tokenizer, model, entailment index, contradiction index (None if absent)
        self._loaded: tuple[Any, Any, int, int | None] | None = None
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

    def _build(self) -> tuple[Any, Any, int, int | None]:
        # Deferred: importing transformers' model classes imports torch.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name, trust_remote_code=False
        )
        model.to(self._device)
        model.eval()
        labels = model.config.id2label
        return tokenizer, model, entailment_index(labels), contradiction_index(labels)

    async def infer(self, pairs: list[tuple[str, str]]) -> list[Inference]:
        if not pairs:
            return []
        await self.load()
        return await asyncio.to_thread(self._predict, pairs)

    def _predict(self, pairs: list[tuple[str, str]]) -> list[Inference]:
        import torch

        assert self._loaded is not None
        tokenizer, model, entail, contradict = self._loaded
        results: list[Inference | None] = [None] * len(pairs)
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        with torch.no_grad():
            for start in range(0, len(order), self._batch_size):
                indices = order[start : start + self._batch_size]
                batch = [pairs[i] for i in indices]
                encoded = tokenizer(
                    [premise for premise, _ in batch],
                    [hypothesis for _, hypothesis in batch],
                    truncation="only_first",
                    max_length=self._max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)
                probabilities = torch.softmax(model(**encoded).logits, dim=-1).tolist()
                for index, row in zip(indices, probabilities, strict=True):
                    results[index] = Inference(
                        entailment=row[entail],
                        contradiction=row[contradict] if contradict is not None else 0.0,
                    )
        return [r for r in results if r is not None]


def entailment_index(id2label: dict[int, str]) -> int:
    """The class index whose label is entailment, whatever the model calls it."""
    for index, label in id2label.items():
        if str(label).lower().startswith("entail"):
            return int(index)
    raise EntailmentLabelError(f"no entailment label in {id2label!r}")


def contradiction_index(id2label: dict[int, str]) -> int | None:
    """The class index whose label is contradiction; None for a two-way model."""
    for index, label in id2label.items():
        if str(label).lower().startswith("contradict"):
            return int(index)
    return None
