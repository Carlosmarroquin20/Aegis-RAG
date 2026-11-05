"""
The text some embedding models expect in front of a query and of a passage.

Asymmetric retrieval models are trained with a marker on each side: the E5
family reads "query: " before a question and "passage: " before a stored text,
and without them its vectors are measurably worse. Symmetric models, such as the
paraphrase MiniLM this service defaults to, take the text as it is. Choosing the
markers from the model's name keeps the API and the offline evaluation
(scripts/evaluate_retrieval.py) embedding the same way.
"""

from __future__ import annotations

from typing import NamedTuple


class EmbeddingPrompts(NamedTuple):
    query: str
    passage: str


NO_PROMPTS = EmbeddingPrompts("", "")

# Matched against the model name in lower case, by substring. First match wins.
_KNOWN: tuple[tuple[str, EmbeddingPrompts], ...] = (
    ("intfloat/multilingual-e5-", EmbeddingPrompts("query: ", "passage: ")),
    ("intfloat/e5-", EmbeddingPrompts("query: ", "passage: ")),
)


def embedding_prompts(model_name: str) -> EmbeddingPrompts:
    """The query and passage markers ``model_name`` was trained with; none if unknown."""
    name = model_name.lower()
    for pattern, prompts in _KNOWN:
        if pattern in name:
            return prompts
    return NO_PROMPTS
