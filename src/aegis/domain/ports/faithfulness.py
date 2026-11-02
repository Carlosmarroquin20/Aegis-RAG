"""
Faithfulness ports.

``FaithfulnessJudgePort`` is what the query pipeline and the offline evaluation
ask: how well is this answer supported by this context. The judge shipped here
is built on ``EntailmentModelPort``, a natural-language-inference model that
scores whether one text follows from another, so the model can be swapped (a
larger NLI model, a hosted one) without touching the judge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aegis.domain.models.document import Document
from aegis.domain.models.faithfulness import FaithfulnessAssessment


class EntailmentModelPort(ABC):
    """Natural-language inference: does a premise entail a hypothesis?"""

    @abstractmethod
    async def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        For each ``(premise, hypothesis)`` pair, the probability in [0, 1] that the
        premise entails the hypothesis, in the same order.
        """


class FaithfulnessJudgePort(ABC):
    """Judges whether an answer is supported by the context it was generated from."""

    @abstractmethod
    async def assess(self, answer: str, context: list[Document]) -> FaithfulnessAssessment:
        """Splits ``answer`` into claims and reports the support each finds in ``context``."""
