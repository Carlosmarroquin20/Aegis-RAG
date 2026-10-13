"""
Tests for the gateway evaluation (scripts/evaluate_gateway.py).

Two things are worth guarding here. The first is that the published report is
what the code produces today: it quotes detection rates, and a rate that was
true three signatures ago is worse than no rate. The second is the one result
in it that is a product requirement rather than a measurement — ordinary
questions must not be blocked.

Detection rates are deliberately not asserted against a floor. A floor invites
adding a signature per missed sample, which improves the figure and nothing else.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_gateway.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_gateway", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: dataclasses resolve their own module through
    # sys.modules while the class body is being processed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_script()
SAMPLES = ev.load_dataset(ev.DEFAULT_DATASET)


def _write_dataset(path: Path, *records: dict[str, str]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def _record(**overrides: str) -> dict[str, str]:
    return {"id": "x-1", "label": "benign", "category": "c", "text": "hello"} | overrides


class TestPublishedReport:
    def test_report_matches_the_current_gateway_and_corpus(self) -> None:
        published = ev.REPORT_PATH.read_text(encoding="utf-8")
        assert published == ev.render_report(SAMPLES), (
            "docs/GATEWAY_EVALUATION.md is stale; "
            "run `uv run python scripts/evaluate_gateway.py --write`"
        )

    def test_check_mode_agrees(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert ev.main(["--check"]) == 0
        assert "up to date" in capsys.readouterr().out

    def test_check_mode_fails_on_a_different_corpus(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        other = _write_dataset(tmp_path / "other.jsonl", _record())
        assert ev.main(["--dataset", str(other), "--check"]) == 1
        assert "stale" in capsys.readouterr().err

    def test_readme_quotes_the_measured_rates(self) -> None:
        readme = (_ROOT / "README.md").read_text(encoding="utf-8")
        verdicts = ev.evaluate(SAMPLES, strict_mode=True)
        attacks = [v for v in verdicts if v.sample.label == ev.ATTACK]
        targeted = [v for v in attacks if v.sample.category not in ev.DOCUMENTED_GAPS]
        benign = [v for v in verdicts if v.sample.label == ev.BENIGN]

        assert f"corpus of {len(SAMPLES)} queries" in readme
        for rate in (targeted, attacks, benign):
            assert f"**{ev.blocked_rate(rate).value:.0%}**" in readme


class TestEverydayTraffic:
    @pytest.mark.parametrize("strict_mode", [True, False])
    def test_ordinary_questions_are_never_blocked(self, strict_mode: bool) -> None:
        ordinary = [s for s in SAMPLES if s.category in ("knowledge_base", "long_form")]
        assert len(ordinary) >= 40

        blocked = [v for v in ev.evaluate(ordinary, strict_mode=strict_mode) if v.blocked]

        assert not blocked, [(v.sample.id, v.rules) for v in blocked]


class TestCorpus:
    def test_both_classes_are_represented_in_similar_numbers(self) -> None:
        attacks = sum(s.label == ev.ATTACK for s in SAMPLES)
        benign = sum(s.label == ev.BENIGN for s in SAMPLES)
        assert attacks >= 100 and benign >= 100
        # A lopsided corpus makes precision a function of the mix, not the detector.
        assert 0.8 <= attacks / benign <= 1.25

    def test_documented_gaps_are_actually_sampled(self) -> None:
        attack_categories = {s.category for s in SAMPLES if s.label == ev.ATTACK}
        assert attack_categories >= ev.DOCUMENTED_GAPS
        assert attack_categories - ev.DOCUMENTED_GAPS

    def test_strict_mode_blocks_everything_permissive_mode_blocks(self) -> None:
        strict = {v.sample.id for v in ev.evaluate(SAMPLES, strict_mode=True) if v.blocked}
        permissive = {v.sample.id for v in ev.evaluate(SAMPLES, strict_mode=False) if v.blocked}
        assert permissive <= strict


class TestDatasetLoader:
    def test_rejects_an_unknown_label(self, tmp_path: Path) -> None:
        path = _write_dataset(tmp_path / "d.jsonl", _record(label="malicious"))
        with pytest.raises(ev.DatasetError, match="d.jsonl:1: unknown label"):
            ev.load_dataset(path)

    def test_rejects_a_duplicate_id(self, tmp_path: Path) -> None:
        path = _write_dataset(tmp_path / "d.jsonl", _record(), _record(text="other"))
        with pytest.raises(ev.DatasetError, match="d.jsonl:2: duplicate id"):
            ev.load_dataset(path)

    def test_rejects_a_duplicate_text(self, tmp_path: Path) -> None:
        path = _write_dataset(tmp_path / "d.jsonl", _record(), _record(id="x-2"))
        with pytest.raises(ev.DatasetError, match="duplicate text"):
            ev.load_dataset(path)

    def test_rejects_a_record_without_text(self, tmp_path: Path) -> None:
        path = _write_dataset(tmp_path / "d.jsonl", {"id": "x-1", "label": "benign"})
        with pytest.raises(ev.DatasetError, match="d.jsonl:1"):
            ev.load_dataset(path)

    def test_rejects_an_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text("\n", encoding="utf-8")
        with pytest.raises(ev.DatasetError, match="no samples"):
            ev.load_dataset(path)

    def test_a_malformed_corpus_is_a_usage_error_not_a_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text("{not json\n", encoding="utf-8")
        assert ev.main(["--dataset", str(path)]) == 2
        assert "d.jsonl:1" in capsys.readouterr().err


class TestRate:
    def test_interval_does_not_collapse_at_zero(self) -> None:
        low, high = ev.Rate(0, 10).interval()
        assert low == 0.0
        # Zero hits in ten tries does not show the true rate is zero.
        assert high == pytest.approx(0.2775, abs=1e-3)

    def test_interval_does_not_collapse_at_one(self) -> None:
        low, high = ev.Rate(10, 10).interval()
        assert low == pytest.approx(0.7225, abs=1e-3)
        assert high == pytest.approx(1.0)

    def test_interval_brackets_the_point_estimate(self) -> None:
        rate = ev.Rate(58, 86)
        low, high = rate.interval()
        assert low < rate.value < high

    def test_empty_rate_is_zero_rather_than_an_error(self) -> None:
        assert ev.Rate(0, 0).value == 0.0
        assert ev.Rate(0, 0).interval() == (0.0, 0.0)


class TestReportCells:
    """The corpus is hostile text; the report must display it, not render it."""

    def test_html_is_escaped(self) -> None:
        assert ev._cell("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"

    def test_table_and_emphasis_syntax_is_escaped(self) -> None:
        assert ev._cell("a|b *c* `d` [e](f)") == r"a\|b \*c\* \`d\` \[e\](f)"

    def test_invisible_characters_are_made_visible(self) -> None:
        assert ev._cell("ig​nore") == r"ig\\u200bnore"

    def test_newlines_cannot_break_the_row(self) -> None:
        assert ev._cell("one\ntwo\r\nthree") == "one two three"

    def test_long_text_is_truncated(self) -> None:
        cell = ev._cell("word " * 200)
        assert cell.endswith("…")
        assert len(cell) <= 91
