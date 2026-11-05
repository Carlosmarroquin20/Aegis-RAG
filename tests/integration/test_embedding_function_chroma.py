"""
The prompted embedding function inside a real (in-process) ChromaDB.

The unit tests call the function directly; this checks what ChromaDB does with
it: that it accepts the subclass for a collection, embeds stored documents
through ``__call__`` (as passages) and query texts through ``embed_query`` (as
queries). Were ChromaDB to embed queries with ``__call__``, an E5 model would
search with "passage: " markers and quietly rank worse. The model is a stand-in,
so no weights are needed; the ranking check only shows the query vector is used.
"""

from __future__ import annotations

import uuid
from typing import Any

import chromadb
import numpy as np
import pytest
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from aegis.infrastructure.vector_stores.embedding_function import (
    PromptedSentenceTransformerEmbeddingFunction,
)


class _MarkerModel:
    """Records what it is asked to embed; texts about "plazo" point one way, the rest another."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        self.texts.extend(texts)
        return np.array([[1.0, 0.1] if "plazo" in text else [0.1, 1.0] for text in texts])


@pytest.fixture()
def model(monkeypatch: pytest.MonkeyPatch) -> _MarkerModel:
    recording = _MarkerModel()

    def fake_init(self: Any, model_name: str, device: str = "cpu", **kwargs: Any) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = False
        self.kwargs = kwargs
        self._model = recording

    monkeypatch.setattr(SentenceTransformerEmbeddingFunction, "__init__", fake_init)
    return recording


def test_chromadb_embeds_documents_as_passages_and_queries_as_queries(
    model: _MarkerModel,
) -> None:
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name=f"prompted-{uuid.uuid4().hex[:8]}",
        embedding_function=PromptedSentenceTransformerEmbeddingFunction(  # type: ignore[arg-type]
            "intfloat/multilingual-e5-small"
        ),
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(ids=["a", "b"], documents=["El plazo es de 15 días.", "Abrimos a las 9."])
    result = collection.query(query_texts=["¿Cuál es el plazo?"], n_results=1)

    assert model.texts == [
        "passage: El plazo es de 15 días.",
        "passage: Abrimos a las 9.",
        "query: ¿Cuál es el plazo?",
    ]
    assert result["documents"] is not None and result["documents"][0] == ["El plazo es de 15 días."]
