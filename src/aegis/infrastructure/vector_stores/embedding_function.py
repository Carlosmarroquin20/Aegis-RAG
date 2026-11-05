"""
The embedding function the API hands to ChromaDB.

It is ChromaDB's SentenceTransformers function plus the markers asymmetric
models expect (embedding_prompts.py): stored chunks are embedded as passages,
and searches through ``embed_query``, which ChromaDB calls for query texts, as
queries. For a symmetric model the markers are empty and nothing changes.

It keeps the parent's ``name()`` and configuration, so a collection records the
same embedding-function type as before. The model itself is recorded and checked
separately (chromadb_adapter.EMBEDDING_MODEL_KEY), because ChromaDB would accept
a different model of the same type in silence.

Imported lazily (interface/api/dependencies.py): importing it imports chromadb.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from chromadb.api.types import Documents, Embeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from aegis.infrastructure.vector_stores.embedding_prompts import embedding_prompts


class PromptedSentenceTransformerEmbeddingFunction(SentenceTransformerEmbeddingFunction):
    """Embeds documents as passages and searches as queries, as the model was trained."""

    def __init__(self, model_name: str, device: str = "cpu", **kwargs: Any) -> None:
        super().__init__(model_name=model_name, device=device, **kwargs)
        self._prompts = embedding_prompts(model_name)

    # ``input`` is ChromaDB's name for the argument: it calls embed_query(input=...).
    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        return self._encode([self._prompts.passage + text for text in input])

    def embed_query(self, input: Documents) -> Embeddings:  # noqa: A002
        return self._encode([self._prompts.query + text for text in input])

    def _encode(self, texts: Sequence[str]) -> Embeddings:
        vectors = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings
        )
        return [np.array(vector, dtype=np.float32) for vector in vectors]
