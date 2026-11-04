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

Two further helpers cover what NLI models get wrong (ADR 0022). A leading
reply word ("No, the stores are closed...") answers the question, which the
judge never sees, and a model reads it as negating the rest of the sentence, so
``strip_reply_word`` drops it before judging. A figure a claim states that
appears nowhere in the context is invented, however fluent the sentence around
it, so ``ungrounded_figures`` names them for the judge to count against the
claim.

Pure functions, no I/O: they run identically in the API and in the offline
evaluation.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation

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


# A word that answers a yes/no question before the claim itself, with its
# punctuation: "Sí, ...", "No. ...", "Yes, ...". "No hay entregas" is a negation,
# not a reply, and keeps its "No" because no punctuation follows it.
_REPLY_WORD = re.compile(r"^\s*(?:s[ií]|no|yes)\s*[,.;:!]\s+", re.IGNORECASE)

# A figure: digits, with the separators that join them in a price, a decimal or
# a time ("Q1,000", "0.10", "1.5", "20:00").
_FIGURE = re.compile(r"\d+(?:[.,:]\d+)*")
_TIME = re.compile(r"^(\d{1,2}):(\d{2})$")
_THOUSANDS = re.compile(r"^\d{1,3}(?:[.,]\d{3})+$")
# Figures the context may spell out. Only the context's words count: a claim is
# checked on the figures it writes as digits, so "dos referencias" in the
# context grounds a claim's "2 references", while a claim's "five years" is left
# to the NLI model. "un"/"una"/"a" are left out, being articles far more often.
_NUMBER_WORDS = {
    **dict.fromkeys(("cero", "zero"), "0"),
    **dict.fromkeys(("uno", "one"), "1"),
    **dict.fromkeys(("dos", "two"), "2"),
    **dict.fromkeys(("tres", "three"), "3"),
    **dict.fromkeys(("cuatro", "four"), "4"),
    **dict.fromkeys(("cinco", "five"), "5"),
    **dict.fromkeys(("seis", "six"), "6"),
    **dict.fromkeys(("siete", "seven"), "7"),
    **dict.fromkeys(("ocho", "eight"), "8"),
    **dict.fromkeys(("nueve", "nine"), "9"),
    **dict.fromkeys(("diez", "ten"), "10"),
    **dict.fromkeys(("once", "eleven"), "11"),
    **dict.fromkeys(("doce", "twelve"), "12"),
    **dict.fromkeys(("quince", "fifteen"), "15"),
    **dict.fromkeys(("veinte", "twenty"), "20"),
    **dict.fromkeys(("treinta", "thirty"), "30"),
    **dict.fromkeys(("cuarenta", "forty"), "40"),
    **dict.fromkeys(("cincuenta", "fifty"), "50"),
    **dict.fromkeys(("cien", "ciento", "hundred"), "100"),
    **dict.fromkeys(("mil", "thousand"), "1000"),
}


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


def strip_reply_word(claim: str) -> str:
    """
    ``claim`` without a leading yes or no that answers the question. Kept as it
    is when nothing worth judging would be left.
    """
    stripped = _REPLY_WORD.sub("", claim, count=1)
    if stripped == claim or len(_WORD.findall(stripped)) < MIN_CLAIM_WORDS:
        return claim
    return stripped[:1].upper() + stripped[1:]


def _readings(figure: str) -> set[str]:
    """
    The value a written figure means, canonical, so the same value written two
    ways matches: "1,000" and "1.000" are a thousand (a separator followed by
    exactly three digits groups thousands in either convention), "0.10" and
    "0,1" are a tenth, and a time keeps its form. Empty for what is not a value.
    """
    time = _TIME.match(figure)
    if time:
        return {f"{int(time.group(1))}:{time.group(2)}"}
    if _THOUSANDS.match(figure):
        candidate = figure.replace(",", "").replace(".", "")
    else:
        candidate = figure.replace(",", ".")
    try:
        value = Decimal(candidate)
    except InvalidOperation:
        return set()
    return {format(value.normalize(), "f")}


def grounding_figures(texts: Iterable[str]) -> frozenset[str]:
    """
    Every figure the texts state, in every reading, for ``ungrounded_figures``
    to check claims against: digits, spelled-out numbers, and each time also as
    its hour and on a 12-hour clock ("20:00" grounds "20", "8:00" and "8").
    """
    grounded: set[str] = set()
    for text in texts:
        for figure in _FIGURE.findall(text):
            grounded.update(_readings(figure))
            time = _TIME.match(figure)
            if time:
                hours, minutes = int(time.group(1)), time.group(2)
                grounded.add(str(hours))
                if hours > 12:
                    grounded.update({f"{hours - 12}:{minutes}", str(hours - 12)})
        for word in _WORD.findall(_fold(text)):
            if word in _NUMBER_WORDS:
                grounded.add(_NUMBER_WORDS[word])
    return frozenset(grounded)


def ungrounded_figures(claim: str, grounded: frozenset[str]) -> list[str]:
    """The figures ``claim`` writes as digits that no reading of ``grounded`` matches."""
    return [figure for figure in _FIGURE.findall(claim) if not _readings(figure) & grounded]
