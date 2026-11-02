"""
Tests for the Transformers NLI adapter.

Loading is checked against stand-in classes, so no model is downloaded. The
prediction path needs torch; it runs where torch is installed (CI, the image)
and is skipped elsewhere. How well the real model judges is measured by
scripts/evaluate_faithfulness.py.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

from aegis.infrastructure.faithfulness.nli_model import (
    DEFAULT_NLI_MODEL,
    EntailmentLabelError,
    TransformersNLIModel,
    contradiction_index,
    entailment_index,
)


class TestEntailmentIndex:
    @pytest.mark.parametrize(
        ("id2label", "expected"),
        [
            ({0: "entailment", 1: "neutral", 2: "contradiction"}, 0),
            ({0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}, 2),
            ({0: "contradiction", 1: "entails", 2: "neutral"}, 1),
        ],
    )
    def test_finds_entailment_whatever_the_order(
        self, id2label: dict[int, str], expected: int
    ) -> None:
        assert entailment_index(id2label) == expected

    def test_a_model_without_entailment_is_refused(self) -> None:
        with pytest.raises(EntailmentLabelError):
            entailment_index({0: "LABEL_0", 1: "LABEL_1"})

    def test_finds_contradiction_or_reports_a_two_way_model(self) -> None:
        assert contradiction_index({0: "entailment", 1: "neutral", 2: "contradiction"}) == 2
        assert contradiction_index({0: "entailment", 1: "not_entailment"}) is None


class TestLoading:
    def test_loads_without_remote_code_and_reads_the_label_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[str, str, dict[str, Any]]] = []

        class _Config:
            id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}

        class _Model:
            config = _Config()

            def to(self, device: str) -> None:
                seen.append(("to", device, {}))

            def eval(self) -> None:
                seen.append(("eval", "", {}))

        class _AutoModel:
            @staticmethod
            def from_pretrained(name: str, **kwargs: Any) -> _Model:
                seen.append(("model", name, kwargs))
                return _Model()

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(name: str, **kwargs: Any) -> object:
                seen.append(("tokenizer", name, kwargs))
                return object()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(
                AutoModelForSequenceClassification=_AutoModel, AutoTokenizer=_AutoTokenizer
            ),
        )

        _, _, entail, contradict = TransformersNLIModel("org/nli", device="cpu")._build()

        assert (entail, contradict) == (2, 0)
        assert ("tokenizer", "org/nli", {"trust_remote_code": False}) in seen
        assert ("model", "org/nli", {"trust_remote_code": False}) in seen
        assert ("eval", "", {}) in seen

    async def test_loads_once_however_many_callers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        builds: list[str] = []

        def build(self: TransformersNLIModel) -> tuple[Any, Any, int, int | None]:
            builds.append(self.model_name)
            return object(), object(), 0, None

        monkeypatch.setattr(TransformersNLIModel, "_build", build)
        model = TransformersNLIModel("org/nli")

        await asyncio.gather(model.load(), model.load(), model.load())

        assert builds == ["org/nli"]

    async def test_no_pairs_does_not_load_the_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(TransformersNLIModel, "_build", lambda self: pytest.fail("loaded"))

        assert await TransformersNLIModel().infer([]) == []

    def test_the_default_is_a_multilingual_model(self) -> None:
        assert "multilingual" in DEFAULT_NLI_MODEL
        assert TransformersNLIModel().model_name == DEFAULT_NLI_MODEL


class TestPredict:
    async def test_returns_probabilities_per_pair_in_the_callers_order(self) -> None:
        torch = pytest.importorskip("torch")

        class _Encoded(dict[str, Any]):
            def to(self, device: str) -> _Encoded:
                return self

        class _Tokenizer:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def __call__(self, premises: list[str], hypotheses: list[str], **kwargs: Any) -> Any:
                self.calls.append({"premises": premises, "hypotheses": hypotheses, **kwargs})
                return _Encoded(hypotheses=hypotheses)

        class _Model:
            def __call__(self, hypotheses: list[str]) -> Any:
                # Logits order: contradiction, neutral, entailment. "yes" entails strongly.
                rows = [[0.0, 0.0, 5.0] if "yes" in h else [5.0, 0.0, 0.0] for h in hypotheses]
                return types.SimpleNamespace(logits=torch.tensor(rows))

        tokenizer = _Tokenizer()
        model = TransformersNLIModel(batch_size=2, max_length=128)
        model._loaded = (tokenizer, _Model(), 2, 0)

        # Lengths differ, so batches are formed by length, not by position.
        results = await model.infer(
            [("a long premise here", "yes one"), ("p", "no two"), ("p", "yes three")]
        )

        assert results[0].entailment > 0.98 and results[2].entailment > 0.98
        assert results[1].entailment < 0.02 and results[1].contradiction > 0.98
        assert results[0].contradiction < 0.02
        assert len(tokenizer.calls) == 2  # batches of two
        assert tokenizer.calls[0]["premises"] == ["p", "p"]  # the two shortest first
        assert tokenizer.calls[0]["truncation"] == "only_first"
        assert tokenizer.calls[0]["max_length"] == 128
