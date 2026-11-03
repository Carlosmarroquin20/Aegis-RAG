"""
Tests for the prompt-injection classifier adapter and the comparison script.

The model is replaced by stand-ins: these pin label handling, the comparison's
arithmetic and its wiring to the real corpus and gateway. What the real model
scores is measured in CI by scripts/compare_injection_classifier.py.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from aegis.infrastructure.security.injection_classifier import (
    DEFAULT_INJECTION_CLASSIFIER,
    InjectionLabelError,
    TransformersInjectionClassifier,
    injection_index,
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_injection_classifier.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_injection_classifier", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cmp = _load_script()


class TestInjectionIndex:
    @pytest.mark.parametrize(
        ("id2label", "expected"),
        [
            ({0: "SAFE", 1: "INJECTION"}, 1),
            ({0: "INJECTION", 1: "SAFE"}, 0),
            ({0: "BENIGN", 1: "JAILBREAK"}, 1),
            ({0: "LABEL_0", 1: "LABEL_1"}, 1),
        ],
    )
    def test_finds_the_positive_class(self, id2label: dict[int, str], expected: int) -> None:
        assert injection_index(id2label) == expected

    def test_an_unknown_labelling_is_refused(self) -> None:
        with pytest.raises(InjectionLabelError):
            injection_index({0: "POSITIVE", 1: "NEGATIVE"})


class TestAdapter:
    def test_loads_without_remote_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[tuple[str, dict[str, Any]]] = []

        class _Model:
            config = types.SimpleNamespace(id2label={0: "SAFE", 1: "INJECTION"})

            def to(self, device: str) -> None:
                pass

            def eval(self) -> None:
                pass

        def _loader(kind: str, result: Any) -> Any:
            def from_pretrained(name: str, **kwargs: Any) -> Any:
                seen.append((kind, kwargs))
                return result

            return types.SimpleNamespace(from_pretrained=from_pretrained)

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(
                AutoTokenizer=_loader("tokenizer", object()),
                AutoModelForSequenceClassification=_loader("model", _Model()),
            ),
        )

        classifier = TransformersInjectionClassifier("org/clf")
        classifier.load()
        classifier.load()  # loads once

        assert seen == [
            ("tokenizer", {"trust_remote_code": False}),
            ("model", {"trust_remote_code": False}),
        ]
        assert classifier._loaded is not None and classifier._loaded[2] == 1

    def test_no_texts_does_not_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(TransformersInjectionClassifier, "load", lambda self: pytest.fail())

        assert TransformersInjectionClassifier().scores([]) == []

    def test_the_default_is_openly_licensed_and_ungated(self) -> None:
        assert DEFAULT_INJECTION_CLASSIFIER == "protectai/deberta-v3-base-prompt-injection-v2"


def _sample(label: str, category: str, text: str = "t") -> Any:
    return cmp.Sample(id=f"{label}-{category}-{text}", label=label, category=category, text=text)


class TestSummary:
    def test_figures_follow_the_gateway_report_definitions(self) -> None:
        samples = [
            _sample("attack", "instruction_override", "a"),
            _sample("attack", "instruction_override", "b"),
            _sample("attack", "paraphrase", "c"),
            _sample("benign", "knowledge_base", "d"),
            _sample("benign", "hard_negative", "e"),
        ]
        blocked = [True, False, True, True, False]

        figures = cmp.summary(samples, blocked)

        assert (figures["targeted"].hits, figures["targeted"].total) == (1, 2)
        assert (figures["gaps"].hits, figures["gaps"].total) == (1, 1)
        assert (figures["attacks"].hits, figures["attacks"].total) == (2, 3)
        assert (figures["false_positives"].hits, figures["false_positives"].total) == (1, 2)
        assert (figures["precision"].hits, figures["precision"].total) == (2, 3)


class TestEndToEnd:
    def test_compares_on_the_real_corpus(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(TransformersInjectionClassifier, "load", lambda self: None)
        # A stand-in that flags anything mentioning "instructions".
        monkeypatch.setattr(
            TransformersInjectionClassifier,
            "scores",
            lambda self, texts: [0.9 if "instruction" in t.lower() else 0.1 for t in texts],
        )

        assert cmp.main(["--show-misses"]) == 0

        out = capsys.readouterr().out
        assert "326 queries" in out
        assert "| Gateway (strict) | Classifier | Either |" in out
        assert "attack/paraphrase †" in out
        assert "Classifier errors:" in out
