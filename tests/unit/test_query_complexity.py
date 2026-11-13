"""
The complexity classifier the model router acts on (ADR 0026).

Each rule is pinned by a question it exists for, in English and Spanish; the
asymmetry (any one signal means complex) and the empty-context shortcut are
pinned separately. How well the rules do on a labelled set is measured, not
asserted, in scripts/evaluate_router.py.
"""

from __future__ import annotations

import time

import pytest

from aegis.domain.services.query_complexity import (
    MAX_SIMPLE_WORDS,
    Tier,
    assess_complexity,
    fold,
)


def _assess(question: str) -> tuple[Tier, tuple[str, ...]]:
    assessment = assess_complexity(question, context_size=5)
    return assessment.tier, assessment.signals


class TestSimpleQuestions:
    @pytest.mark.parametrize(
        "question",
        [
            "¿Cuántos días tengo para devolver un colchón?",
            "Do you deliver on Sundays?",
            "¿Aceptan transferencias bancarias?",
            "How long is the warranty on power tools?",
            "Si devuelvo algo que estaba rebajado, ¿me regresan el precio normal o lo que pagué?",
        ],
    )
    def test_a_single_lookup_goes_to_the_small_model(self, question: str) -> None:
        assert _assess(question) == (Tier.SIMPLE, ())

    def test_a_simple_question_reports_why(self) -> None:
        assert assess_complexity("Do you deliver?", context_size=3).reason == "plain_lookup"


class TestComplexSignals:
    @pytest.mark.parametrize(
        ("question", "signal"),
        [
            ("What is the difference between the two warranties?", "comparison"),
            ("¿Qué diferencia hay entre la garantía normal y la extendida?", "comparison"),
            ("How do store hours differ between the capital and Xela?", "comparison"),
            ("¿Qué es mejor, pagar en cuotas o al contado?", "comparison"),
            ("Compará el plazo de muebles con el de colchones.", "comparison"),
            ("Why was I charged for shipping?", "explanation"),
            ("¿Por qué me cobraron el envío?", "explanation"),
            ("Explicame como funciona el programa.", "explanation"),
            ("Calculate the monthly payment for a Q9,000 purchase.", "calculation"),
            ("¿Cuánto me costaría la lavadora con instalación?", "calculation"),
            ("Summarize your return policy.", "synthesis"),
            ("Explícame paso a paso cómo hago un reclamo.", "synthesis"),
            ("Is the extended warranty worth it?", "synthesis"),
            ("¿Me devuelven el dinero y cuánto tarda?", "multi_part"),
            ("Can I return it, and who pays the pickup?", "multi_part"),
            ("Can I return it? Who pays the pickup?", "multiple_questions"),
            ("I spent Q16,000 and Q4,000 more in December.", "several_figures"),
        ],
    )
    def test_each_rule_sends_its_questions_to_the_large_model(
        self, question: str, signal: str
    ) -> None:
        tier, signals = _assess(question)

        assert tier is Tier.COMPLEX
        assert signal in signals

    def test_a_long_question_is_complex_whatever_it_says(self) -> None:
        question = " ".join(["word"] * (MAX_SIMPLE_WORDS + 1))

        assert _assess(question) == (Tier.COMPLEX, ("long_question",))

    def test_accents_and_case_do_not_matter(self) -> None:
        assert _assess("EXPLÍCAME el proceso")[0] is Tier.COMPLEX
        assert _assess("explicame el proceso")[0] is Tier.COMPLEX
        assert fold("¿Por QUÉ?") == "¿por que?"

    def test_one_figure_is_not_a_calculation(self) -> None:
        assert _assess("¿Puedo pagar en cuotas una compra de Q1,500?")[0] is Tier.SIMPLE

    def test_every_signal_that_fired_is_reported_in_order(self) -> None:
        _, signals = _assess("Compare both plans and explain why one is cheaper?")

        assert signals[:2] == ("comparison", "explanation")


class TestContext:
    def test_no_retrieved_context_goes_to_the_small_model_whatever_the_question(self) -> None:
        assessment = assess_complexity("Compare and explain everything", context_size=0)

        assert assessment.tier is Tier.SIMPLE
        assert assessment.reason == "no_context"


class TestCostIsBounded:
    @pytest.mark.parametrize(
        "text",
        [
            "a" * 8192,
            "y " * 4096,
            "compa" * 1638,
            "1," * 4096,
            "and " * 2048,
            "?" * 8192,
        ],
    )
    def test_an_8kb_adversarial_question_is_classified_quickly(self, text: str) -> None:
        start = time.perf_counter()
        assess_complexity(text, context_size=5)

        assert time.perf_counter() - start < 0.05
