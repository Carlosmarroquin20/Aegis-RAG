"""
Tests for the retrieval quality gate in scripts/evaluate_retrieval.py.

The gate itself runs in CI with the real embedding model; these tests pin the
threshold arithmetic without one, so a mistake there (an off-by-one cutoff, a
comparison the wrong way round) cannot make the gate pass or fail silently.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from aegis.domain.models.document import Document

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_retrieval.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_retrieval", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: dataclasses resolve their own module through
    # sys.modules while the class body is being processed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_script()


class TestShortfalls:
    def test_no_thresholds_never_fail(self) -> None:
        assert ev.shortfalls([50, 50], min_hit5=None, min_mrr=None) == []

    def test_rank_five_counts_as_a_hit(self) -> None:
        assert ev.shortfalls([5, 5], min_hit5=1.0, min_mrr=None) == []

    def test_rank_six_is_a_miss(self) -> None:
        problems = ev.shortfalls([1, 6], min_hit5=0.94, min_mrr=None)
        assert problems == ["hit@5 0.500 is below the required 0.940"]

    def test_exactly_meeting_a_threshold_passes(self) -> None:
        # MRR of [1, 2] is 0.75.
        assert ev.shortfalls([1, 2], min_hit5=None, min_mrr=0.75) == []

    def test_each_missed_threshold_is_reported(self) -> None:
        problems = ev.shortfalls([10, 10], min_hit5=0.5, min_mrr=0.5)
        assert len(problems) == 2
        assert problems[1].startswith("MRR 0.100")


_QUESTIONS = [
    json.loads(line)
    for line in ev.DEFAULT_QUESTIONS.read_text(encoding="utf-8").splitlines()
    if line.strip()
]


class TestQuestionSet:
    def test_every_expected_document_exists_in_the_corpus(self) -> None:
        """A renamed document would otherwise surface as a crash, not as a gate result."""
        corpus = {p.name for p in ev.DEFAULT_CORPUS.iterdir()}
        assert {q["expected_source"] for q in _QUESTIONS} <= corpus

    def test_ids_and_questions_are_unique(self) -> None:
        assert len({q["id"] for q in _QUESTIONS}) == len(_QUESTIONS)
        assert len({q["question"] for q in _QUESTIONS}) == len(_QUESTIONS)

    def test_every_question_has_a_known_language_and_kind(self) -> None:
        assert {q["lang"] for q in _QUESTIONS} == {"es", "en"}
        assert {q["kind"] for q in _QUESTIONS} == {"direct", "paraphrase", "scenario"}

    def test_every_document_is_asked_about(self) -> None:
        """A document with a handful of questions would hide a regression in it."""
        corpus = {p.name for p in ev.DEFAULT_CORPUS.iterdir()}
        per_document = {name: 0 for name in corpus}
        for question in _QUESTIONS:
            per_document[question["expected_source"]] += 1
        assert min(per_document.values()) >= 15, per_document

    def test_english_is_large_enough_to_read_on_its_own(self) -> None:
        english = sum(q["lang"] == "en" for q in _QUESTIONS)
        assert english >= 50 and english / len(_QUESTIONS) >= 0.3


class TestRanking:
    def _doc(self, source: str) -> Document:
        return Document(id=source, content="text", metadata={"source": source})

    def test_rank_is_the_first_chunk_of_the_expected_document(self) -> None:
        results = [self._doc("a.md"), self._doc("b.md"), self._doc("b.md")]
        assert ev._rank(results, "b.md") == 2

    def test_absent_document_is_a_miss(self) -> None:
        assert ev._rank([self._doc("a.md")], "b.md") == ev.MISS

    def test_a_miss_counts_against_both_metrics(self) -> None:
        problems = ev.shortfalls([1, ev.MISS], min_hit5=0.6, min_mrr=0.6)
        assert problems == [
            "hit@5 0.500 is below the required 0.600",
            "MRR 0.500 is below the required 0.600",
        ]


def _chunk_count(out: str) -> int:
    """The chunk count the script reports; other output (ingestion logs) may surround it."""
    found = re.search(r"^(\d+) chunks from", out, re.MULTILINE)
    assert found, out
    return int(found.group(1))


class _RecordingEncoder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        self.texts.extend(texts)
        return np.ones((len(texts), 4)) / 2


class TestMemoryStore:
    async def test_an_e5_model_sees_its_query_and_passage_markers(self) -> None:
        store = ev.MemoryStore()
        await store.add_documents([Document(id="c", content="Texto del fragmento.")])
        encoder = _RecordingEncoder()

        store.embed_with(encoder, "intfloat/multilingual-e5-small")
        await store.similarity_search("¿Una pregunta?")

        assert encoder.texts == ["passage: Texto del fragmento.", "query: ¿Una pregunta?"]

    async def test_a_symmetric_model_sees_the_text_as_it_is(self) -> None:
        store = ev.MemoryStore()
        await store.add_documents([Document(id="c", content="Texto del fragmento.")])
        encoder = _RecordingEncoder()

        store.embed_with(encoder, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        await store.similarity_search("¿Una pregunta?")

        assert encoder.texts == ["Texto del fragmento.", "¿Una pregunta?"]


class _BagOfWordsEncoder:
    """Stands in for SentenceTransformer: hashed word counts, normalized."""

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        vectors = np.zeros((len(texts), 256))
        for row, text in enumerate(texts):
            for word in text.lower().split():
                vectors[row, sum(map(ord, word)) % 256] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


class _LengthCrossEncoder:
    """Stands in for CrossEncoder: prefers shorter chunks, deterministically."""

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def predict(self, pairs: list[tuple[str, str]], **_: object) -> list[float]:
        return [1.0 / (1.0 + len(text)) for _, text in pairs]


class TestEndToEnd:
    def test_compare_runs_every_mode_over_the_real_corpus(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """
        The models are stubs, so the figures mean nothing; what is checked is that
        every mode runs through the application's own retrieval code end to end.
        """
        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            types.SimpleNamespace(
                SentenceTransformer=_BagOfWordsEncoder, CrossEncoder=_LengthCrossEncoder
            ),
        )

        assert ev.main(["--compare", "--mode", "hybrid"]) == 0

        out = capsys.readouterr().out
        assert "hybrid  (gated)" in out
        for mode in ("dense", "dense+rerank", "hybrid+rerank", "bm25"):
            assert f"\n  {mode}  " in out

    def test_the_gate_applies_to_the_gated_mode_only(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            types.SimpleNamespace(
                SentenceTransformer=_BagOfWordsEncoder, CrossEncoder=_LengthCrossEncoder
            ),
        )

        assert ev.main(["--compare", "--mode", "hybrid", "--min-hit5", "1.01"]) == 1
        assert capsys.readouterr().out.count("FAIL: hit@5") == 1

    def test_the_chunk_size_reaches_the_ingestion_pipeline(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            types.SimpleNamespace(
                SentenceTransformer=_BagOfWordsEncoder, CrossEncoder=_LengthCrossEncoder
            ),
        )

        assert ev.main(["--mode", "dense", "--chunk-size", "256", "--overlap", "32"]) == 0
        small = _chunk_count(capsys.readouterr().out)
        assert ev.main(["--mode", "dense", "--chunk-size", "1024"]) == 0
        large = _chunk_count(capsys.readouterr().out)

        assert small > large
