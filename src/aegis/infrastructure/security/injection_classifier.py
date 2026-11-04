"""
A fine-tuned prompt-injection classifier, run locally with Transformers.

The SecurityGateway is a signature engine: it stops the phrasings and
obfuscations it was written for and is blind to a reworded request (ADR 0002,
docs/GATEWAY_EVALUATION.md). A classifier trained on injection datasets is the
usual second opinion. This adapter scores texts with one, so the two can be
measured against the same labelled corpus (scripts/compare_injection_classifier.py)
before either is trusted with more.

The default, ``protectai/deberta-v3-base-prompt-injection-v2``, is Apache-2.0
and openly downloadable; Meta's Prompt Guard models are gated behind a licence
acceptance and Llama Guard is an 8B-parameter LLM, neither of which a CI runner
can fetch or run. Weights load without ``trust_remote_code``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from typing import Any

from aegis.domain.ports.security import InjectionScorerPort

DEFAULT_INJECTION_CLASSIFIER = "protectai/deberta-v3-base-prompt-injection-v2"


class InjectionLabelError(RuntimeError):
    """The model's configuration names no injection label."""


class TransformersInjectionClassifier(InjectionScorerPort):
    """
    Probability, per text, that it is a prompt injection. ``scores`` is
    synchronous and CPU-bound (the offline comparison uses it directly);
    ``score`` runs it on a worker thread for use inside the API.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_INJECTION_CLASSIFIER,
        *,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._loaded: tuple[Any, Any, int] | None = None
        # load() can be reached from a worker thread (score) and from startup.
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self) -> None:
        with self._load_lock:
            if self._loaded is not None:
                return
            # Deferred: importing transformers' model classes imports torch.
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=False)
            model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name, trust_remote_code=False
            )
            model.to(self._device)
            model.eval()
            self._loaded = (tokenizer, model, injection_index(model.config.id2label))

    async def score(self, text: str) -> float:
        """One text's injection probability, computed off the event loop."""
        (result,) = await asyncio.to_thread(self.scores, [text])
        return result

    def scores(self, texts: Sequence[str]) -> list[float]:
        """The injection probability of each text, in order."""
        if not texts:
            return []
        self.load()
        import torch

        assert self._loaded is not None
        tokenizer, model, injection = self._loaded
        results: list[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), self._batch_size):
                batch = list(texts[start : start + self._batch_size])
                encoded = tokenizer(
                    batch,
                    truncation=True,
                    max_length=self._max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)
                probabilities = torch.softmax(model(**encoded).logits, dim=-1)
                results.extend(probabilities[:, injection].tolist())
        return results


def injection_index(id2label: dict[int, str]) -> int:
    """
    The class index meaning "injection". Models name it INJECTION, JAILBREAK,
    MALICIOUS or plain LABEL_1; a two-class model with generic labels is read
    the usual way, class 1 positive.
    """
    for index, label in id2label.items():
        if str(label).lower() in {"injection", "jailbreak", "malicious", "unsafe"}:
            return int(index)
    if len(id2label) == 2 and {str(v).upper() for v in id2label.values()} == {
        "LABEL_0",
        "LABEL_1",
    }:
        return 1
    raise InjectionLabelError(f"no injection label in {id2label!r}")
