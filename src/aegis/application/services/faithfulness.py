"""
Faithfulness judged by natural-language inference.

For every claim (sentence) in the answer, an NLI model reads it against each
premise taken from the retrieved chunks. Two choices follow the SummaC line of
work on NLI-based consistency checking (Laban et al., TACL 2022):

- **Sentence premises.** The chunks are split into sentences, and every pair of
  neighbouring sentences is offered too, so a claim spanning two still finds
  one premise holding both. NLI models are trained on short premises; given a
  whole chunk, they let a figure from one sentence vouch for a claim about
  another.
- **Contradiction counts against support.** A premise supports a claim by its
  entailment probability minus its contradiction probability, so a premise that
  both "entails" and "contradicts" a claim (the usual sign of a wrong detail in
  an otherwise familiar sentence) does not vouch for it.

The claim's support is the best premise's; it is supported at ``threshold`` or
more. ``granularity="chunk"`` and ``contradiction_aware=False`` restore the plain
whole-chunk entailment judge, which the evaluation keeps for comparison.

Why NLI and not an LLM judge: an NLI model is small, local and deterministic.
The same answer and context always get the same score, which is what lets the
score gate a merge (ADR 0012, ADR 0017) and lets an air-gapped deployment
monitor its answers without sending them anywhere.
"""

from __future__ import annotations

from typing import Literal

from aegis.domain.models.document import Document
from aegis.domain.models.faithfulness import FaithfulnessAssessment, Inference, SentenceSupport
from aegis.domain.ports.faithfulness import EntailmentModelPort, FaithfulnessJudgePort
from aegis.domain.services.claims import is_abstention, split_claims, split_premises

DEFAULT_SUPPORT_THRESHOLD = 0.5
Granularity = Literal["sentence", "chunk"]


class NLIFaithfulnessJudge(FaithfulnessJudgePort):
    """A claim is supported when some premise supports it with ``threshold`` or more."""

    def __init__(
        self,
        model: EntailmentModelPort,
        *,
        threshold: float = DEFAULT_SUPPORT_THRESHOLD,
        granularity: Granularity = "sentence",
        contradiction_aware: bool = True,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self._model = model
        self._threshold = threshold
        self._granularity = granularity
        self._contradiction_aware = contradiction_aware

    async def assess(self, answer: str, context: list[Document]) -> FaithfulnessAssessment:
        if is_abstention(answer):
            return FaithfulnessAssessment(sentences=(), abstained=True)
        claims = split_claims(answer)
        if not claims:
            return FaithfulnessAssessment(sentences=())

        premises = self._premises(context)
        if not premises:
            # Nothing was retrieved, so nothing the answer says is supported.
            return FaithfulnessAssessment(
                sentences=tuple(SentenceSupport(c, 0.0, False) for c in claims)
            )

        pairs = [(premise, claim) for claim in claims for premise in premises]
        inferences = await self._model.infer(pairs)
        per_claim = len(premises)
        sentences = []
        for index, claim in enumerate(claims):
            batch = inferences[index * per_claim : (index + 1) * per_claim]
            support = max(self._support(inference) for inference in batch)
            sentences.append(
                SentenceSupport(
                    text=claim, support=round(support, 4), supported=support >= self._threshold
                )
            )
        return FaithfulnessAssessment(sentences=tuple(sentences))

    def _premises(self, context: list[Document]) -> list[str]:
        if self._granularity == "chunk":
            premises = [d.content for d in context]
            if len(premises) > 1:
                premises.append("\n\n".join(premises))
            return premises
        premises = []
        for document in context:
            sentences = split_premises(document.content)
            premises.extend(sentences)
            premises.extend(f"{a} {b}" for a, b in zip(sentences, sentences[1:], strict=False))
        # Chunks overlap, so the same sentence can come twice; ask about it once.
        return list(dict.fromkeys(premises))

    def _support(self, inference: Inference) -> float:
        if not self._contradiction_aware:
            return inference.entailment
        return max(0.0, inference.entailment - inference.contradiction)
