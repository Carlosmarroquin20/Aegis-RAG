"""
How well an answer is supported by the context it was generated from.

Faithfulness is judged claim by claim: an answer is split into sentences, and
each sentence is supported when some part of the retrieved context entails it.
The score is the share of supported sentences, so one invented claim in a
three-sentence answer reads as 0.67 rather than disappearing into an average,
and ``unsupported`` names the sentences that need a human's attention.

An answer that declines ("the context does not contain enough information")
makes no claim, so it cannot be unfaithful; it is reported as an abstention
instead of being scored.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Inference:
    """An NLI model's verdict on one (premise, hypothesis) pair, as probabilities."""

    entailment: float
    contradiction: float


@dataclass(frozen=True, slots=True)
class SentenceSupport:
    text: str
    # Best support any premise of the context gives the sentence, in [0, 1].
    support: float
    supported: bool


@dataclass(frozen=True, slots=True)
class FaithfulnessAssessment:
    sentences: tuple[SentenceSupport, ...]
    abstained: bool = False

    @property
    def score(self) -> float:
        """Share of supported sentences in [0, 1]; 1.0 for an abstention or an empty answer."""
        if self.abstained or not self.sentences:
            return 1.0
        return sum(s.supported for s in self.sentences) / len(self.sentences)

    @property
    def fully_supported(self) -> bool:
        return all(s.supported for s in self.sentences) or self.abstained

    @property
    def unsupported(self) -> tuple[str, ...]:
        return tuple(s.text for s in self.sentences if not s.supported)
