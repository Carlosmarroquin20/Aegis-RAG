"""
Tests for the NLI faithfulness judge and the assessment it returns.

The entailment model is a stand-in that "entails" a claim when every content
word of the claim appears in the premise. That is enough to pin what the judge
does with the model's answers: which pairs it asks about, how it aggregates
them, where the threshold falls, and when it does not ask at all.
"""

from __future__ import annotations

import re

import pytest

from aegis.application.services.faithfulness import NLIFaithfulnessJudge
from aegis.domain.models.document import Document
from aegis.domain.models.faithfulness import FaithfulnessAssessment, SentenceSupport
from aegis.domain.ports.faithfulness import EntailmentModelPort

_WORDS = re.compile(r"\w{3,}")


class _WordOverlapModel(EntailmentModelPort):
    """Entailment 0.9 when the premise contains every 3+ letter word of the hypothesis."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    async def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        return [
            0.9 if set(_WORDS.findall(h.lower())) <= set(_WORDS.findall(p.lower())) else 0.1
            for p, h in pairs
        ]


def _chunk(text: str) -> Document:
    return Document(id=text[:12], content=text)


_CONTEXT = [
    _chunk("Para muebles y colchones el plazo es de 15 días."),
    _chunk("Sin comprobante solo se ofrece un cambio por otro producto."),
]


class TestAssess:
    async def test_supported_claims_score_one(self) -> None:
        judge = NLIFaithfulnessJudge(_WordOverlapModel())

        result = await judge.assess("Para colchones el plazo es de 15 días.", _CONTEXT)

        assert result.score == 1.0
        assert result.fully_supported
        assert result.unsupported == ()

    async def test_an_invented_claim_is_named_and_lowers_the_score(self) -> None:
        judge = NLIFaithfulnessJudge(_WordOverlapModel())

        result = await judge.assess(
            "Para colchones el plazo es de 15 días. La tienda paga el flete de regreso.",
            _CONTEXT,
        )

        assert result.score == 0.5
        assert not result.fully_supported
        assert result.unsupported == ("La tienda paga el flete de regreso.",)

    async def test_a_claim_needing_two_chunks_is_supported_by_the_joined_context(self) -> None:
        model = _WordOverlapModel()
        judge = NLIFaithfulnessJudge(model)

        result = await judge.assess(
            "Colchones: plazo de 15 días; sin comprobante solo cambio.", _CONTEXT
        )

        assert result.fully_supported
        # One premise per chunk plus the joined context.
        assert len(model.calls[0]) == 3

    async def test_support_is_the_best_premise(self) -> None:
        judge = NLIFaithfulnessJudge(_WordOverlapModel())

        result = await judge.assess("Sin comprobante solo se ofrece un cambio.", _CONTEXT)

        assert result.sentences[0].support == 0.9

    async def test_the_model_is_asked_once_for_every_pair(self) -> None:
        model = _WordOverlapModel()
        judge = NLIFaithfulnessJudge(model)

        await judge.assess("Primera afirmación del texto. Segunda afirmación del texto.", _CONTEXT)

        assert len(model.calls) == 1
        assert len(model.calls[0]) == 2 * 3

    async def test_an_abstention_is_not_scored_and_not_sent_to_the_model(self) -> None:
        model = _WordOverlapModel()
        judge = NLIFaithfulnessJudge(model)

        result = await judge.assess(
            "El contexto proporcionado no contiene información suficiente para responder.",
            _CONTEXT,
        )

        assert result.abstained
        assert result.score == 1.0
        assert model.calls == []

    async def test_no_context_supports_nothing(self) -> None:
        model = _WordOverlapModel()
        judge = NLIFaithfulnessJudge(model)

        result = await judge.assess("Para colchones el plazo es de 15 días.", [])

        assert result.score == 0.0
        assert model.calls == []

    async def test_an_empty_answer_has_nothing_to_judge(self) -> None:
        result = await NLIFaithfulnessJudge(_WordOverlapModel()).assess("  ", _CONTEXT)

        assert result.sentences == ()
        assert result.score == 1.0

    async def test_the_threshold_decides_support(self) -> None:
        strict = NLIFaithfulnessJudge(_WordOverlapModel(), threshold=0.95)

        result = await strict.assess("Para colchones el plazo es de 15 días.", _CONTEXT)

        assert not result.fully_supported

    @pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
    def test_the_threshold_must_be_a_probability(self, threshold: float) -> None:
        with pytest.raises(ValueError):
            NLIFaithfulnessJudge(_WordOverlapModel(), threshold=threshold)


class TestAssessment:
    def test_score_is_the_share_of_supported_claims(self) -> None:
        assessment = FaithfulnessAssessment(
            sentences=(
                SentenceSupport("a", 0.9, True),
                SentenceSupport("b", 0.8, True),
                SentenceSupport("c", 0.2, False),
            )
        )

        assert assessment.score == pytest.approx(2 / 3)
        assert assessment.unsupported == ("c",)

    def test_an_abstention_is_fully_supported(self) -> None:
        assessment = FaithfulnessAssessment(sentences=(), abstained=True)

        assert assessment.fully_supported
        assert assessment.score == 1.0
