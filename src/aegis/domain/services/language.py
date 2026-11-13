"""
Tells Spanish from English in a short text, by its function words (ADR 0027).

The system prompt asks the model to answer in the question's language. Measured
live, the small model of the router answered 11 of 65 English questions in
Spanish, the language of the corpus it read; the large model never did. That
is checkable without a model: articles, pronouns and prepositions are frequent
enough that a sentence of a few words carries several, and the two lists
barely overlap.

The detector only has to separate the two languages the service supports, and
only has to be right when it answers. When it cannot tell, a terse "Q55." or a
tie, it returns None, and the caller treats that as "no evidence of a
mismatch". It never guesses.

Pure and linear: one pass of a bounded word pattern.
"""

from __future__ import annotations

import re
from typing import Final

SPANISH: Final = "es"
ENGLISH: Final = "en"

_SPANISH_WORDS: Final = frozenset(
    "el la los las del que qué en sí se es son puedes puede para por con una un su sus tu tus"
    " está están hay cuando cuándo como cómo más mi mis me lo al y pero también cuánto cuánta"
    " cuántos cuántas dónde cuál cuáles tengo tienen usted ustedes nos le les ya".split()
)
_ENGLISH_WORDS: Final = frozenset(
    "the is are you your can to of and for in it not yes be will with on at this that we our"
    " do does did i my what how when where which there have has an if or but they their".split()
)
# Letters only, so "Q1,000" and "15:00" contribute nothing either way.
_WORD: Final = re.compile(r"[a-záéíóúñü]+")


def detect_language(text: str) -> str | None:
    """'es', 'en', or None when the function words do not decide it."""
    spanish = english = 0
    for word in _WORD.findall(text.lower()):
        if word in _SPANISH_WORDS:
            spanish += 1
        elif word in _ENGLISH_WORDS:
            english += 1
    if spanish == english:
        return None
    return SPANISH if spanish > english else ENGLISH


def languages_differ(question: str, answer: str) -> bool:
    """True only when both texts have a language and they are not the same."""
    asked, answered = detect_language(question), detect_language(answer)
    return asked is not None and answered is not None and asked != answered
