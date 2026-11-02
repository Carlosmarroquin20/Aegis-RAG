"""
Faithfulness judged by natural-language inference.

For every claim (sentence) in the answer, the NLI model is asked whether each
retrieved chunk entails it; the claim's support is the best of those
probabilities. A claim that combines facts from two chunks may not follow from
either alone, so the whole context, joined, is offered as one more premise (the
model truncates it to what fits).

Why NLI and not an LLM judge: an NLI model is small, local and deterministic.
The same answer and context always get the same score, which is what lets the
score gate a merge (ADR 0012, ADR 0017) and lets an air-gapped deployment
monitor its answers without sending them anywhere.
"""

from __future__ import annotations

from aegis.domain.models.document import Document
from aegis.domain.models.faithfulness import FaithfulnessAssessment, SentenceSupport
from aegis.domain.ports.faithfulness import EntailmentModelPort, FaithfulnessJudgePort
from aegis.domain.services.claims import is_abstention, split_claims

DEFAULT_SUPPORT_THRESHOLD = 0.5


class NLIFaithfulnessJudge(FaithfulnessJudgePort):
    """A claim is supported when some premise entails it with ``threshold`` or more."""

    def __init__(
        self, model: EntailmentModelPort, *, threshold: float = DEFAULT_SUPPORT_THRESHOLD
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self._model = model
        self._threshold = threshold

    async def assess(self, answer: str, context: list[Document]) -> FaithfulnessAssessment:
        if is_abstention(answer):
            return FaithfulnessAssessment(sentences=(), abstained=True)
        claims = split_claims(answer)
        if not claims:
            return FaithfulnessAssessment(sentences=())

        premises = [d.content for d in context]
        if len(premises) > 1:
            premises.append("\n\n".join(premises))
        if not premises:
            # Nothing was retrieved, so nothing the answer says is supported.
            return FaithfulnessAssessment(
                sentences=tuple(SentenceSupport(c, 0.0, False) for c in claims)
            )

        pairs = [(premise, claim) for claim in claims for premise in premises]
        probabilities = await self._model.entailment(pairs)
        per_claim = len(premises)
        sentences = []
        for index, claim in enumerate(claims):
            support = max(probabilities[index * per_claim : (index + 1) * per_claim])
            sentences.append(
                SentenceSupport(
                    text=claim, support=round(support, 4), supported=support >= self._threshold
                )
            )
        return FaithfulnessAssessment(sentences=tuple(sentences))
