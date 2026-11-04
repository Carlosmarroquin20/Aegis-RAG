"""
Tests for scripts/evaluate_faithfulness.py and its labelled answers.

The gate runs in CI with the real NLI model; these tests pin the arithmetic,
the dataset's consistency and the end-to-end wiring with a stand-in model, so a
mistake in any of them cannot make the gate pass or fail silently.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from aegis.domain.models.faithfulness import FaithfulnessAssessment, Inference, SentenceSupport
from aegis.domain.services.claims import is_abstention
from aegis.infrastructure.faithfulness.nli_model import TransformersNLIModel

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_faithfulness.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_faithfulness", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_script()
_KINDS = {
    "supported": "faithful",
    "supported_with_filler": "faithful",
    "abstention": "faithful",
    "wrong_detail": "unfaithful",
    "contradiction": "unfaithful",
    "unsupported_addition": "unfaithful",
    "outside_knowledge": "unfaithful",
}
_RECORDS = [
    json.loads(line)
    for line in ev.DEFAULT_ANSWERS.read_text(encoding="utf-8").splitlines()
    if line.strip()
]


def _outcome(label: str, flagged: bool) -> object:
    item = ev.LabelledAnswer("id", "q", "es", "s.md", "kind", label, "a")
    sentence = SentenceSupport("a", 0.1 if flagged else 0.9, not flagged)
    return ev.Outcome(item, FaithfulnessAssessment(sentences=(sentence,)))


class TestSummary:
    def test_counts(self) -> None:
        outcomes = [
            _outcome("unfaithful", True),
            _outcome("unfaithful", False),
            _outcome("faithful", False),
            _outcome("faithful", False),
            _outcome("faithful", True),
        ]

        summary = ev.summarise(outcomes)

        assert (summary.caught, summary.unfaithful) == (1, 2)
        assert (summary.false_flags, summary.faithful) == (1, 3)
        assert summary.recall == 0.5
        assert summary.false_flag_rate == pytest.approx(1 / 3)
        assert summary.accuracy == pytest.approx(3 / 5)

    def test_meeting_the_thresholds_exactly_passes(self) -> None:
        summary = ev.Summary(caught=4, unfaithful=5, false_flags=1, faithful=10)

        assert ev.shortfalls(summary, min_recall=0.8, max_false_flags=0.1) == []

    def test_each_missed_threshold_is_reported(self) -> None:
        summary = ev.Summary(caught=3, unfaithful=5, false_flags=2, faithful=10)

        problems = ev.shortfalls(summary, min_recall=0.8, max_false_flags=0.1)

        assert problems == [
            "recall 0.600 is below the required 0.800",
            "false-flag rate 0.200 is above the allowed 0.100",
        ]

    def test_no_thresholds_never_fail(self) -> None:
        summary = ev.Summary(caught=0, unfaithful=5, false_flags=5, faithful=5)

        assert ev.shortfalls(summary, min_recall=None, max_false_flags=None) == []


class TestDataset:
    def test_ids_are_unique(self) -> None:
        ids = [r["id"] for r in _RECORDS]
        assert len(ids) == len(set(ids))

    def test_every_source_exists_in_the_corpus(self) -> None:
        corpus = {p.name for p in ev.DEFAULT_CORPUS.iterdir()}
        assert {r["source"] for r in _RECORDS} <= corpus

    def test_each_kind_has_one_label(self) -> None:
        labels: dict[str, set[str]] = {}
        for record in _RECORDS:
            labels.setdefault(record["kind"], set()).add(record["label"])
        assert all(len(found) == 1 for found in labels.values()), labels

    def test_both_labels_are_well_represented(self) -> None:
        unfaithful = sum(r["label"] == "unfaithful" for r in _RECORDS)
        assert 0.3 < unfaithful / len(_RECORDS) < 0.7

    def test_every_kind_is_known_and_labelled_as_its_name_says(self) -> None:
        assert {r["kind"]: r["label"] for r in _RECORDS} == _KINDS

    def test_every_kind_is_measured_in_both_languages(self) -> None:
        """A kind seen in one language only cannot say whether the judge works in the other."""
        for kind in _KINDS:
            assert {r["lang"] for r in _RECORDS if r["kind"] == kind} == {"es", "en"}, kind

    def test_each_language_has_enough_answers_to_read_on_its_own(self) -> None:
        for lang in ("es", "en"):
            subset = [r for r in _RECORDS if r["lang"] == lang]
            assert sum(r["label"] == "unfaithful" for r in subset) >= 40, lang
            assert sum(r["label"] == "faithful" for r in subset) >= 40, lang

    def test_every_abstention_is_recognised_as_one(self) -> None:
        """Otherwise the judge would score a refusal's wording instead of skipping it."""
        for record in _RECORDS:
            assert is_abstention(record["answer"]) == (record["kind"] == "abstention"), record["id"]

    def test_the_same_answer_is_not_labelled_twice(self) -> None:
        pairs = [(r["question"], r["answer"]) for r in _RECORDS]
        assert len(pairs) == len(set(pairs))

    def test_an_invalid_label_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "answers.jsonl"
        record = dict(_RECORDS[0], label="maybe")
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="label must be one of"):
            ev.load_answers(path)


_WORDS = re.compile(r"\w{3,}")


async def _overlap_entailment(
    self: TransformersNLIModel, pairs: list[tuple[str, str]]
) -> list[Inference]:
    """Stand-in NLI: entailed by the share of the claim's words found in the premise."""
    scores = []
    for premise, claim in pairs:
        words = set(_WORDS.findall(claim.lower()))
        shared = words & set(_WORDS.findall(premise.lower()))
        scores.append(Inference(len(shared) / len(words) if words else 0.0, 0.0))
    return scores


class TestEndToEnd:
    def test_runs_over_the_real_dataset_and_corpus(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(TransformersNLIModel, "infer", _overlap_entailment)

        assert ev.main(["--show-errors"]) == 0
        assert ev.main(["--granularity", "chunk", "--no-contradiction"]) == 0

        out = capsys.readouterr().out
        assert f"{len(_RECORDS)} labelled answers" in out
        assert "(gated)" in out
        assert "wrong_detail" in out and "abstention" in out

    def test_the_gate_applies_to_the_gated_model_only(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(TransformersNLIModel, "infer", _overlap_entailment)

        assert ev.main(["--compare-model", "org/other", "--min-recall", "1.01"]) == 1

        out = capsys.readouterr().out
        assert out.count("FAIL: recall") == 1
        assert "org/other" in out


class _BagOfWordsEncoder:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        vectors = np.zeros((len(texts), 256))
        for row, text in enumerate(texts):
            for word in text.lower().split():
                vectors[row, sum(map(ord, word)) % 256] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


class TestLive:
    def test_judges_the_pipelines_own_answers(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aegis.interface.api import dependencies
        from tests.conftest import FakeLLMClient

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            types.SimpleNamespace(SentenceTransformer=_BagOfWordsEncoder),
        )
        monkeypatch.setattr(TransformersNLIModel, "infer", _overlap_entailment)
        llm = FakeLLMClient(fixed_response="Los domingos no hay entregas a domicilio.")
        monkeypatch.setattr(dependencies, "get_llm_client", lambda: llm)

        assert ev.main(["--live", "--min-score", "0.0"]) == 0

        out = capsys.readouterr().out
        assert "mean faithfulness" in out
        assert "q-001" in out and "q-038" in out
