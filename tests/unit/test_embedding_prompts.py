"""The query and passage markers chosen for an embedding model."""

from __future__ import annotations

import pytest

from aegis.infrastructure.vector_stores.embedding_prompts import (
    NO_PROMPTS,
    EmbeddingPrompts,
    embedding_prompts,
)


@pytest.mark.parametrize(
    "model",
    ["intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base", "INTFLOAT/E5-LARGE-V2"],
)
def test_e5_models_get_their_markers(model: str) -> None:
    assert embedding_prompts(model) == EmbeddingPrompts("query: ", "passage: ")


@pytest.mark.parametrize(
    "model",
    [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "BAAI/bge-m3",
        "a/local-model",
    ],
)
def test_other_models_take_the_text_as_it_is(model: str) -> None:
    assert embedding_prompts(model) == NO_PROMPTS
