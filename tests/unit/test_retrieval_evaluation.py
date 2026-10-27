"""
Tests for the retrieval quality gate in scripts/evaluate_retrieval.py.

The gate itself runs in CI with the real embedding model; these tests pin the
threshold arithmetic without one, so a mistake there (an off-by-one cutoff, a
comparison the wrong way round) cannot make the gate pass or fail silently.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_retrieval.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_retrieval", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: dataclasses resolve their own module through
    # sys.modules while the class body is being processed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_script()


class TestShortfalls:
    def test_no_thresholds_never_fail(self) -> None:
        assert ev.shortfalls([50, 50], min_hit5=None, min_mrr=None) == []

    def test_rank_five_counts_as_a_hit(self) -> None:
        assert ev.shortfalls([5, 5], min_hit5=1.0, min_mrr=None) == []

    def test_rank_six_is_a_miss(self) -> None:
        problems = ev.shortfalls([1, 6], min_hit5=0.94, min_mrr=None)
        assert problems == ["hit@5 0.500 is below the required 0.940"]

    def test_exactly_meeting_a_threshold_passes(self) -> None:
        # MRR of [1, 2] is 0.75.
        assert ev.shortfalls([1, 2], min_hit5=None, min_mrr=0.75) == []

    def test_each_missed_threshold_is_reported(self) -> None:
        problems = ev.shortfalls([10, 10], min_hit5=0.5, min_mrr=0.5)
        assert len(problems) == 2
        assert problems[1].startswith("MRR 0.100")


class TestQuestionSet:
    def test_every_expected_document_exists_in_the_corpus(self) -> None:
        """A renamed document would otherwise surface as a crash, not as a gate result."""
        corpus = {p.name for p in ev.DEFAULT_CORPUS.iterdir()}
        for line in ev.DEFAULT_QUESTIONS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                assert json.loads(line)["expected_source"] in corpus
