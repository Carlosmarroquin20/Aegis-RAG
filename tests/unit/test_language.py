"""
The answer-language check behind the router's wrong-language escalation
(ADR 0027). It must never call a language wrongly: an undecided None is safe
(the answer is kept), a wrong call discards a good answer or keeps a bad one.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from aegis.domain.services.language import ENGLISH, SPANISH, detect_language, languages_differ

_ROOT = Path(__file__).resolve().parents[2]


class TestDetection:
    @pytest.mark.parametrize(
        ("text", "language"),
        [
            ("Do you deliver on Sundays?", ENGLISH),
            ("How long is the warranty on power tools?", ENGLISH),
            ("¿Cuántos días tengo para devolver un colchón?", SPANISH),
            ("Sí, la afiliación es gratuita.", SPANISH),
            ("No, plants are listed as non-returnable items.", ENGLISH),
            ("No, no se aceptan devoluciones de plantas.", SPANISH),
        ],
    )
    def test_it_tells_the_two_languages_apart(self, text: str, language: str) -> None:
        assert detect_language(text) == language

    @pytest.mark.parametrize("text", ["Q55.", "15:00", "", "¿Aceptan American Express?"])
    def test_without_function_words_it_does_not_guess(self, text: str) -> None:
        assert detect_language(text) is None

    def test_a_mismatch_needs_both_languages_known(self) -> None:
        assert languages_differ("Do you deliver?", "Sí, entregamos los sábados.")
        assert not languages_differ("Do you deliver?", "Q55.")
        assert not languages_differ("¿Aceptan American Express?", "Yes, we do.")
        assert not languages_differ("¿Entregan los domingos?", "No, los domingos no.")


class TestOnTheLabelledQuestions:
    def test_no_routing_question_is_given_the_wrong_language(self) -> None:
        dataset = _ROOT / "evals" / "routing" / "dataset.jsonl"
        rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]

        wrong = [r["id"] for r in rows if detect_language(r["question"]) not in (None, r["lang"])]

        assert wrong == []

    def test_on_the_recorded_answers_it_flags_only_the_small_models_wrong_language(self) -> None:
        # Answers recorded in the live comparison (docs/ROUTER_LIVE_EVALUATION.md):
        # the large model never answered in the wrong language.
        recorded = _ROOT / "evals" / "routing" / "live" / "gpt-oss-20b.jsonl"
        rows = [json.loads(line) for line in recorded.read_text(encoding="utf-8").splitlines()]

        def flagged(side: str) -> int:
            return sum(
                "answer" in r[side] and languages_differ(r["question"], r[side]["answer"])
                for r in rows
            )

        assert (flagged("small"), flagged("large")) == (11, 0)


def test_an_8kb_text_is_classified_quickly() -> None:
    start = time.perf_counter()
    detect_language("palabra " * 1024)

    assert time.perf_counter() - start < 0.05
