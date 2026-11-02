"""
Splits an answer into the claims a faithfulness judge checks one by one, and
the context into the premises it checks them against.

A sentence is the unit: small enough that one invented detail does not hide
behind a true one, large enough to carry its own subject and verb. Splitting is
deliberately conservative. A full stop only ends a sentence when whitespace and
the start of a new one follow it, so "Q0.10", "1.5 puntos" and "S.A. es" stay
whole. Line breaks and list markers also separate claims, because the model is
told to answer in plain text but may still list items. A fragment too short to
be checked on its own ("Sí.", "Efectivo.") is joined to the next claim, or to
the previous one at the end, since it means nothing without its neighbour.

Pure functions, no I/O: they run identically in the API and in the offline
evaluation.
"""

from __future__ import annotations

import re
import unicodedata

# End of sentence: terminal punctuation (optionally closed by a quote or a
# parenthesis), whitespace, then what can start a sentence in Spanish or English.
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]?\s+(?=[\"'(¿¡A-ZÁÉÍÓÚÑÜ0-9])")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+")
_HAS_WORD = re.compile(r"[^\W\d_]{2,}")
_WORD = re.compile(r"\w+")
# Fewer words than this and a fragment cannot be judged on its own.
MIN_CLAIM_WORDS = 3

# How the system prompt tells the model to decline (infrastructure/llm/prompting.py),
# and the usual ways a model paraphrases it, accents and case removed.
_ABSTENTION_MARKERS = (
    "no contiene informacion suficiente",
    "no contiene suficiente informacion",
    "no hay informacion suficiente",
    "no tengo informacion suficiente",
    "does not contain enough information",
    "does not contain sufficient information",
    "not enough information to answer",
)


def split_claims(text: str) -> list[str]:
    """The sentences of ``text`` that carry at least one word, in order."""
    fragments: list[str] = []
    for line in text.splitlines():
        line = _LIST_MARKER.sub("", line).strip()
        if not line:
            continue
        for sentence in _SENTENCE_END.split(line):
            sentence = sentence.strip()
            if _HAS_WORD.search(sentence):
                fragments.append(sentence)

    claims: list[str] = []
    pending = ""
    for fragment in fragments:
        joined = f"{pending} {fragment}".strip()
        if len(_WORD.findall(joined)) < MIN_CLAIM_WORDS:
            pending = joined
            continue
        claims.append(joined)
        pending = ""
    if pending:
        if claims:
            claims[-1] = f"{claims[-1]} {pending}"
        else:
            claims.append(pending)
    return claims


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if _HAS_WORD.search(s)]


def split_premises(text: str) -> list[str]:
    """
    The context as NLI premises: sentences, and table rows on their own.

    NLI models are trained on premises of a sentence or two; a whole chunk, with
    several figures and exceptions, is where they confuse "30 días" for the
    general rule with "15 días" for mattresses. Stored chunks keep the source's
    hard line breaks inside sentences, so lines are rejoined before splitting,
    except table rows ("Domingo | 10:00 a 19:00"), which are facts by themselves.
    """
    segments: list[str] = []
    prose: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if " | " in line:
            segments.extend(_sentences(" ".join(prose)))
            prose.clear()
            segments.append(line.replace(" | ", ": "))
        elif line:
            prose.append(line)
    segments.extend(_sentences(" ".join(prose)))
    return segments


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).split())


def is_abstention(text: str) -> bool:
    """
    True when the answer is a refusal to answer from the context. Only short
    answers count: a refusal followed by a paragraph of claims is still judged.
    """
    folded = _fold(text)
    return any(marker in folded for marker in _ABSTENTION_MARKERS) and len(split_claims(text)) <= 2
