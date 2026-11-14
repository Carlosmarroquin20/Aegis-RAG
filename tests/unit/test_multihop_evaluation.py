"""
The multi-hop retrieval evaluation (scripts/evaluate_multihop.py, ADR 0028).

The report itself needs the embedding model, so CI checks that it is current in
the Retrieval Quality Gate job. These tests cover what runs without models: the
dataset and the recorded plans are well formed and agree with the corpus, a
planner reply is accepted only when it is exactly what was asked for, and
results are interleaved without duplicates up to the budget.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from aegis.domain.models.document import Document

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_multihop.py"
_CORPUS = _ROOT / "examples" / "atencion-cliente"


def _load_script() -> ModuleType:
    sys.path.insert(0, str(_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("evaluate_multihop", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_script()
SAMPLES = ev.load_dataset(ev.DEFAULT_DATASET)
PLANS = ev.load_plans(ev.DEFAULT_PLANS)


class TestDataset:
    def test_every_required_document_exists_in_the_corpus(self) -> None:
        corpus = {p.name for p in _CORPUS.glob("*.md")}

        assert set().union(*(s.required for s in SAMPLES)) <= corpus

    def test_it_has_multi_document_questions_and_controls_in_both_languages(self) -> None:
        assert {s.kind for s in SAMPLES} == {"multi", "single"}
        assert {s.lang for s in SAMPLES} == {"es", "en"}

    def test_every_question_has_a_recorded_plan(self) -> None:
        assert {s.id for s in SAMPLES} == set(PLANS)

    @pytest.mark.parametrize(
        ("record", "message"),
        [
            ({"id": "x", "lang": "en", "kind": "multi", "question": "q"}, "required_sources"),
            (
                {
                    "id": "x",
                    "lang": "en",
                    "kind": "multi",
                    "question": "q",
                    "required_sources": ["a.md"],
                },
                "kind does not match",
            ),
        ],
    )
    def test_a_malformed_line_is_rejected_with_its_number(
        self, tmp_path: Path, record: dict[str, object], message: str
    ) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with pytest.raises(ev.DatasetError, match=f"bad.jsonl:1.*{message}"):
            ev.load_dataset(path)


class TestParsePlan:
    def test_a_well_formed_reply_gives_its_queries(self) -> None:
        reply = json.dumps({"queries": [" shipping costs ", "Gold benefits"]})

        assert ev.parse_plan(reply, "q") == (("shipping costs", "Gold benefits"), True)

    @pytest.mark.parametrize(
        "reply",
        [
            "not json",
            json.dumps(["a list"]),
            json.dumps({"queries": []}),
            json.dumps({"queries": ["a", "b", "c", "d"]}),
            json.dumps({"queries": ["ok", ""]}),
            json.dumps({"queries": ["ok", 3]}),
            json.dumps({"queries": ["x" * 201]}),
            json.dumps({"answer": "Yes, it is free."}),
        ],
    )
    def test_anything_else_falls_back_to_the_question(self, reply: str) -> None:
        assert ev.parse_plan(reply, "the question") == (("the question",), False)


def _doc(doc_id: str) -> Document:
    return Document(id=doc_id, content="text", metadata={"source": f"{doc_id}.md"})


class TestInterleave:
    def test_it_takes_the_first_of_each_query_before_the_second(self) -> None:
        first = [_doc("a1"), _doc("a2")]
        second = [_doc("b1"), _doc("b2")]

        assert [d.id for d in ev.interleave([first, second], 3)] == ["a1", "b1", "a2"]

    def test_a_chunk_found_by_two_queries_counts_once(self) -> None:
        merged = ev.interleave([[_doc("a"), _doc("b")], [_doc("a"), _doc("c")]], 5)

        assert [d.id for d in merged] == ["a", "b", "c"]

    def test_it_never_exceeds_the_budget(self) -> None:
        results = [[_doc(f"{q}{i}") for i in range(5)] for q in "xyz"]

        assert len(ev.interleave(results, 5)) == 5


def test_the_report_names_where_the_strategies_differ() -> None:
    sample = ev.Sample("m-1", "en", "multi", "q?", frozenset({"a.md", "b.md"}))
    outcomes = [
        ev.Outcome(sample, "single", 5, frozenset({"a.md"}), 5, 100),
        ev.Outcome(sample, "decompose", 5, frozenset({"a.md", "b.md"}), 5, 100),
        ev.Outcome(sample, "single", 10, frozenset({"a.md", "b.md"}), 10, 200),
        ev.Outcome(sample, "decompose", 10, frozenset({"a.md", "b.md"}), 10, 200),
    ]
    plans = {"m-1": ev.Plan("m-1", ("a", "b"), True, 10, 5)}

    report = ev.render_report([sample], plans, outcomes, "model-x")

    assert "| m-1 | decompose | b.md | q? |" in report
    assert "| single, 5 passages | 0.0% (0/1) |" in report
