"""
Decides whether a question is simple enough for a small model (ADR 0026).

Most customer questions are a single lookup ("Do you deliver on Sundays?"),
and a small model answers those from the retrieved passage as well as a large
one, for a fraction of the cost and latency. Some are not: comparisons,
explanations, calculations, several questions in one, summaries across
policies. Those go to the large model.

The two mistakes are not equally bad. Sending a simple question to the large
model costs money; sending a complex one to the small model costs a wrong or
shallow answer. So the classifier is asymmetric on purpose: a question is
SIMPLE only when nothing in it looks complex, and any one signal is enough to
route it to the large model. There is no score to tune and no threshold to
drift; every decision names the signals that made it.

One signal points the other way. With no retrieved context the answer can only
be an abstention ("the documents do not cover this"), and the small model says
that as well as the large one.

The rules are matched on a folded view of the question: lower case, accents
removed, so "explícame", "explicame" and "EXPLICAME" are the same word, in
English and Spanish (voseo included: "compará", "explicá"). Like the security
gateway, every pattern stays linear on non-matching input; this runs on the
event loop for every query.

Pure, no I/O: the API and the offline evaluation run exactly this code.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class Tier(StrEnum):
    SIMPLE = "simple"
    COMPLEX = "complex"


@dataclass(frozen=True, slots=True)
class ComplexityAssessment:
    tier: Tier
    # The rules that fired, in a fixed order. Empty for a simple question, or
    # ("no_context",) when the context decided it. Names are a fixed set, so
    # they are safe as metric labels.
    signals: tuple[str, ...]

    @property
    def reason(self) -> str:
        """One label for metrics: the first signal, or why it is simple."""
        return self.signals[0] if self.signals else "plain_lookup"


# Longer than this and a question usually carries several conditions or asks
# several things; a one-fact lookup is rarely over 30 words.
MAX_SIMPLE_WORDS: Final = 40

# Each rule is a set of word prefixes anchored at a word boundary: no
# open-ended quantifier, so a failed match costs one pass over the text.
_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "comparison",
        re.compile(
            r"\b(?:compar|versus\b|vs\b|differ|diferenc|distint|better\b|worse\b"
            r"|cheaper\b|conviene|mas barat|mas car[oa]\b|which is (?:best|better)"
            r"|(?:cual|que) es mejor|cual (?:sale mejor|me conviene)|cual sale mas)"
        ),
    ),
    (
        "explanation",
        re.compile(
            r"\b(?:why\b|explain|explic|reason|razon\b|razones\b|analy[sz]|analiz"
            r"|justif|walk me through|por que\b)"
        ),
    ),
    (
        "calculation",
        re.compile(
            r"\b(?:calculat|calcul|compute\b|how much would|how much will i"
            r"|in total\b|en total\b|cuanto (?:me )?(?:saldria|costaria|pagaria)"
            r"|cuanto (?:seria|serian)\b|monthly payment|cuota mensual)"
        ),
    ),
    (
        "synthesis",
        re.compile(
            r"\b(?:summari[sz]|summary\b|overview\b|resum|en resumen\b|step by step"
            r"|paso a paso|everything (?:i|you)|todo lo que|in detail\b|en detalle\b"
            r"|is it worth|worth it\b|vale la pena)"
        ),
    ),
    (
        "multi_part",
        re.compile(
            r"\b(?:and|y|e)\s(?:how|what|when|where|which|who|cuanto|cuanta|cuantos|cuantas"
            r"|como|cuando|donde|cual|cuales|quien|quienes)\b"
        ),
    ),
)

_FIGURE: Final = re.compile(r"\d+(?:[.,]\d+)*")
_WORD: Final = re.compile(r"\w+")


def fold(text: str) -> str:
    """Lower case with accents removed: the view every rule is matched on."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def assess_complexity(question: str, *, context_size: int) -> ComplexityAssessment:
    """
    ``context_size`` is how many passages were retrieved for the question.
    """
    if context_size == 0:
        return ComplexityAssessment(Tier.SIMPLE, ("no_context",))

    view = fold(question)
    signals = [name for name, pattern in _RULES if pattern.search(view)]
    if view.count("?") >= 2:
        signals.append("multiple_questions")
    if len(set(_FIGURE.findall(view))) >= 2:
        signals.append("several_figures")
    if len(_WORD.findall(view)) > MAX_SIMPLE_WORDS:
        signals.append("long_question")

    if signals:
        return ComplexityAssessment(Tier.COMPLEX, tuple(signals))
    return ComplexityAssessment(Tier.SIMPLE, ())
