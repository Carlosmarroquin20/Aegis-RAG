"""
The embedding function handed to ChromaDB: documents embedded as passages,
searches as queries, with the markers the configured model was trained with.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from aegis.infrastructure.vector_stores.embedding_function import (
    PromptedSentenceTransformerEmbeddingFunction,
)


class _RecordingModel:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        self.texts.extend(texts)
        return np.ones((len(texts), 3))


@pytest.fixture()
def model(monkeypatch: pytest.MonkeyPatch) -> _RecordingModel:
    """Stands in for the SentenceTransformer the parent class would load."""
    recording = _RecordingModel()

    def fake_init(self: Any, model_name: str, device: str = "cpu", **kwargs: Any) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = False
        self.kwargs = kwargs
        self._model = recording

    monkeypatch.setattr(SentenceTransformerEmbeddingFunction, "__init__", fake_init)
    return recording


def test_an_e5_model_embeds_chunks_as_passages_and_searches_as_queries(
    model: _RecordingModel,
) -> None:
    function = PromptedSentenceTransformerEmbeddingFunction("intfloat/multilingual-e5-small")

    stored = function(["Para colchones el plazo es de 15 días."])
    searched = function.embed_query(["¿Cuántos días tengo?"])

    assert model.texts == [
        "passage: Para colchones el plazo es de 15 días.",
        "query: ¿Cuántos días tengo?",
    ]
    assert len(stored) == len(searched) == 1
    assert stored[0].dtype == np.float32


def test_a_symmetric_model_embeds_the_text_as_it_is(model: _RecordingModel) -> None:
    function = PromptedSentenceTransformerEmbeddingFunction(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    function(["Un fragmento."])
    function.embed_query(["Una pregunta."])

    assert model.texts == ["Un fragmento.", "Una pregunta."]


def test_the_collection_records_the_same_function_type_as_before(model: _RecordingModel) -> None:
    """A different name() would make ChromaDB refuse every existing collection."""
    function = PromptedSentenceTransformerEmbeddingFunction("intfloat/multilingual-e5-small")

    assert function.name() == SentenceTransformerEmbeddingFunction.name()
    assert function.get_config()["model_name"] == "intfloat/multilingual-e5-small"
