"""
Tests for the NLI faithfulness judge and the assessment it returns.

The NLI model is a stand-in: it "entails" a claim when every content word of
the claim appears in the premise, and "contradicts" it when the premise holds a
number the claim does not. That is enough to pin what the judge does with the
model's answers: which premises it builds, which pairs it asks about, how it
aggregates them, where the threshold falls, and when it does not ask at all.
"""

from __future__ import annotations

import re

import pytest

from aegis.application.services.faithfulness import NLIFaithfulnessJudge
from aegis.domain.models.document import Document
from aegis.domain.models.faithfulness import FaithfulnessAssessment, Inference, SentenceSupport
from aegis.domain.ports.faithfulness import EntailmentModelPort

_WORDS = re.compile(r"\w{3,}")
_NUMBERS = re.compile(r"\d+")


class _StubNLI(EntailmentModelPort):
    def __init__(self, *, contradiction_on_other_numbers: bool = False) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self._contradicts = contradiction_on_other_numbers

    async def infer(self, pairs: list[tuple[str, str]]) -> list[Inference]:
        self.calls.append(pairs)
        results = []
        for premise, claim in pairs:
            words = set(_WORDS.findall(claim.lower()))
            entails = words <= set(_WORDS.findall(premise.lower()))
            other_numbers = set(_NUMBERS.findall(premise)) - set(_NUMBERS.findall(claim))
            contradicts = self._contradicts and entails and bool(other_numbers)
            results.append(Inference(0.9 if entails else 0.1, 0.8 if contradicts else 0.05))
        return results


def _chunk(text: str) -> Document:
    return Document(id=text[:12], content=text)


_CONTEXT = [
    _chunk("Para muebles y colchones el plazo es de 15 días. Las plantas no se devuelven."),
    _chunk("Sin comprobante solo se ofrece un cambio por otro producto."),
]


class TestAssess:
    async def test_supported_claims_score_one(self) -> None:
        judge = NLIFaithfulnessJudge(_StubNLI())

        result = await judge.assess("Para colchones el plazo es de 15 días.", _CONTEXT)

        assert result.score == 1.0
        assert result.fully_supported
        assert result.unsupported == ()

    async def test_an_invented_claim_is_named_and_lowers_the_score(self) -> None:
        judge = NLIFaithfulnessJudge(_StubNLI())

        result = await judge.assess(
            "Para colchones el plazo es de 15 días. La tienda paga el flete de regreso.",
            _CONTEXT,
        )

        assert result.score == 0.5
        assert not result.fully_supported
        assert result.unsupported == ("La tienda paga el flete de regreso.",)

    async def test_premises_are_the_contexts_sentences_and_neighbouring_pairs(self) -> None:
        model = _StubNLI()

        await NLIFaithfulnessJudge(model).assess("Una afirmación cualquiera aquí.", _CONTEXT)

        premises = [premise for premise, _ in model.calls[0]]
        assert premises == [
            "Para muebles y colchones el plazo es de 15 días.",
            "Las plantas no se devuelven.",
            "Para muebles y colchones el plazo es de 15 días. Las plantas no se devuelven.",
            "Sin comprobante solo se ofrece un cambio por otro producto.",
        ]

    async def test_a_claim_spanning_two_sentences_finds_their_pair(self) -> None:
        judge = NLIFaithfulnessJudge(_StubNLI())

        result = await judge.assess(
            "Colchones: plazo de 15 días; las plantas no se devuelven.", _CONTEXT
        )

        assert result.fully_supported

    async def test_a_sentence_repeated_by_overlapping_chunks_is_asked_about_once(self) -> None:
        model = _StubNLI()
        overlapping = [_chunk("Primera frase del texto. Segunda frase."), _chunk("Segunda frase.")]

        await NLIFaithfulnessJudge(model).assess("Una afirmación cualquiera aquí.", overlapping)

        premises = [premise for premise, _ in model.calls[0]]
        assert premises.count("Segunda frase.") == 1

    async def test_chunk_granularity_uses_whole_chunks_and_their_join(self) -> None:
        model = _StubNLI()

        await NLIFaithfulnessJudge(model, granularity="chunk").assess(
            "Una afirmación cualquiera aquí.", _CONTEXT
        )

        premises = [premise for premise, _ in model.calls[0]]
        assert premises == [d.content for d in _CONTEXT] + [
            "\n\n".join(d.content for d in _CONTEXT)
        ]

    async def test_contradiction_takes_support_away(self) -> None:
        """A premise that 'entails' the claim's words but holds another figure does not vouch."""
        context = [_chunk("Para colchones el plazo es de 15 días, para el resto 30 días.")]
        claim = "Para colchones el plazo es de 30 días."

        aware = await NLIFaithfulnessJudge(_StubNLI(contradiction_on_other_numbers=True)).assess(
            claim, context
        )
        unaware = await NLIFaithfulnessJudge(
            _StubNLI(contradiction_on_other_numbers=True), contradiction_aware=False
        ).assess(claim, context)

        assert not aware.fully_supported
        assert aware.sentences[0].support == pytest.approx(0.1)
        assert unaware.fully_supported

    async def test_support_never_goes_below_zero(self) -> None:
        class _Contradicting(EntailmentModelPort):
            async def infer(self, pairs: list[tuple[str, str]]) -> list[Inference]:
                return [Inference(0.05, 0.9) for _ in pairs]

        result = await NLIFaithfulnessJudge(_Contradicting()).assess(
            "Una afirmación cualquiera aquí.", _CONTEXT
        )

        assert result.sentences[0].support == 0.0

    async def test_the_model_is_asked_once_for_every_pair(self) -> None:
        model = _StubNLI()

        await NLIFaithfulnessJudge(model).assess(
            "Primera afirmación del texto. Segunda afirmación del texto.", _CONTEXT
        )

        assert len(model.calls) == 1
        assert len(model.calls[0]) == 2 * 4

    async def test_an_abstention_is_not_scored_and_not_sent_to_the_model(self) -> None:
        model = _StubNLI()

        result = await NLIFaithfulnessJudge(model).assess(
            "El contexto proporcionado no contiene información suficiente para responder.",
            _CONTEXT,
        )

        assert result.abstained
        assert result.score == 1.0
        assert model.calls == []

    async def test_no_context_supports_nothing(self) -> None:
        model = _StubNLI()

        result = await NLIFaithfulnessJudge(model).assess(
            "Para colchones el plazo es de 15 días.", []
        )

        assert result.score == 0.0
        assert model.calls == []

    async def test_an_empty_answer_has_nothing_to_judge(self) -> None:
        result = await NLIFaithfulnessJudge(_StubNLI()).assess("  ", _CONTEXT)

        assert result.sentences == ()
        assert result.score == 1.0

    async def test_the_threshold_decides_support(self) -> None:
        strict = NLIFaithfulnessJudge(_StubNLI(), threshold=0.9)

        result = await strict.assess("Para colchones el plazo es de 15 días.", _CONTEXT)

        # 0.9 entailment minus 0.05 contradiction falls short of 0.9.
        assert not result.fully_supported

    @pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
    def test_the_threshold_must_be_a_probability(self, threshold: float) -> None:
        with pytest.raises(ValueError):
            NLIFaithfulnessJudge(_StubNLI(), threshold=threshold)


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
