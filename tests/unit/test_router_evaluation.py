"""
Tests for the router evaluation (scripts/evaluate_router.py).

The published report quotes routing rates, and a rate that was true three rules
ago is worse than none, so the report must be what the code produces today.
Like the gateway evaluation, the rates are not asserted against a floor: a
floor invites a rule per missed question.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_router.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_router", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_script()
SAMPLES = ev.load_dataset(ev.DEFAULT_DATASET)


class TestPublishedReport:
    def test_report_matches_the_current_classifier_and_dataset(self) -> None:
        published = ev.REPORT_PATH.read_text(encoding="utf-8")
        assert published == ev.render_report(SAMPLES), (
            "docs/ROUTER_EVALUATION.md is stale; "
            "run `uv run python scripts/evaluate_router.py --write`"
        )

    def test_check_mode_agrees(self) -> None:
        assert ev.main(["--check"]) == 0


class TestDataset:
    def test_it_includes_every_retrieval_question(self) -> None:
        retrieval = _ROOT / "evals" / "retrieval" / "atencion-cliente.jsonl"
        questions = {
            json.loads(line)["question"]
            for line in retrieval.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

        assert questions <= {s.question for s in SAMPLES}

    def test_both_labels_and_languages_are_represented(self) -> None:
        assert {s.label.value for s in SAMPLES} == {"simple", "complex"}
        assert {s.lang for s in SAMPLES} == {"es", "en"}

    def test_a_malformed_line_names_its_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "x"}\n', encoding="utf-8")

        with pytest.raises(ev.DatasetError, match="bad.jsonl:1"):
            ev.load_dataset(path)

    def test_duplicate_ids_are_rejected(self, tmp_path: Path) -> None:
        record = {
            "id": "x",
            "lang": "en",
            "origin": "authored",
            "kind": "lookup",
            "label": "simple",
            "question": "q",
        }
        path = tmp_path / "dup.jsonl"
        path.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")

        with pytest.raises(ev.DatasetError, match="duplicate"):
            ev.load_dataset(path)


def test_the_readme_quotes_the_current_figures() -> None:
    verdicts = ev.evaluate(SAMPLES)
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")

    for rate in (ev.complex_kept_large(verdicts), ev.simple_moved_small(verdicts)):
        assert f"{rate.value:.1%}" in readme, (
            f"README.md should quote {rate.value:.1%}; update it with the routing figures"
        )
