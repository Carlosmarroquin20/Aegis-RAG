"""
Measures how well an embedding model retrieves the right document for a question.

Usage:
    uv run python scripts/evaluate_retrieval.py
    uv run python scripts/evaluate_retrieval.py --model sentence-transformers/all-MiniLM-L6-v2
    uv run python scripts/evaluate_retrieval.py --corpus examples/atencion-cliente \\
        --questions evals/retrieval/atencion-cliente.jsonl

Retrieval is the half of RAG that decides whether the model ever sees the
answer: a question whose document is not in the top results is answered with
"not enough information" however good the LLM is. This script ingests a corpus
through the application's own pipeline (type detection, parsing, chunking), so
the chunks are the ones the API would store, embeds them with the given model,
and reports for each question where the first chunk of the expected document
ranks.

It needs the embedding model locally (downloaded on first use) and is not run
in CI. Results are printed rather than written, because they depend on the
model files and not only on the code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import structlog

from aegis.application.dtos.ingestion_dtos import IngestRequest, UploadedFile
from aegis.application.use_cases.ingest_documents import IngestDocumentsUseCase
from aegis.config import Settings
from aegis.domain.models.document import Document
from aegis.domain.ports.vector_store import VectorStorePort
from aegis.infrastructure.parsers.parser_registry import ParserRegistry

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CORPUS = ROOT / "examples" / "atencion-cliente"
DEFAULT_QUESTIONS = ROOT / "evals" / "retrieval" / "atencion-cliente.jsonl"
CUTOFFS = (1, 3, 5)


class _CollectingStore(VectorStorePort):
    """Keeps the chunks the ingest use case produces; retrieval is done here, not in a store."""

    def __init__(self) -> None:
        self.chunks: list[Document] = []

    async def similarity_search(
        self, query: str, k: int = 5, score_threshold: float = 0.0
    ) -> list[Document]:
        raise NotImplementedError

    async def add_documents(self, documents: list[Document]) -> None:
        self.chunks.extend(documents)

    async def delete_documents(self, ids: list[str]) -> None:
        raise NotImplementedError

    async def existing_ids(self, ids: list[str]) -> set[str]:
        return set()

    async def list_documents(self, limit: int = 20, offset: int = 0) -> list[Document]:
        return self.chunks[offset : offset + limit]

    async def health_check(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    lang: str
    text: str
    expected_source: str


async def _chunk_corpus(corpus: Path) -> list[Document]:
    store = _CollectingStore()
    use_case = IngestDocumentsUseCase(
        vector_store=store, parser_registry=ParserRegistry(), collection="retrieval-eval"
    )
    request = IngestRequest()
    for path in sorted(corpus.rglob("*")):
        if path.is_file():
            upload = UploadedFile(
                filename=path.relative_to(corpus).as_posix(), content=path.read_bytes()
            )
            await use_case.execute(upload, request)
    return store.chunks


def _load_questions(path: Path) -> list[Question]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions.append(
                Question(
                    record["id"], record["lang"], record["question"], record["expected_source"]
                )
            )
    return questions


def _ranks(model_name: str, chunks: Sequence[Document], questions: Sequence[Question]) -> list[int]:
    """1-based rank of the first chunk from the expected document, per question."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    # Normalized vectors make the dot product the cosine similarity, which is the
    # metric the Chroma collection is created with ("hnsw:space": "cosine").
    chunk_vectors = model.encode([c.content for c in chunks], normalize_embeddings=True)
    question_vectors = model.encode([q.text for q in questions], normalize_embeddings=True)

    ranks = []
    for question, vector in zip(questions, question_vectors, strict=True):
        scores = chunk_vectors @ vector
        order = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))
        sources = [chunks[i].metadata["source"] for i in order]
        ranks.append(sources.index(question.expected_source) + 1)
    return ranks


def _summary(ranks: Sequence[int]) -> str:
    hits = "  ".join(f"hit@{k} {sum(r <= k for r in ranks) / len(ranks):6.1%}" for k in CUTOFFS)
    mrr = statistics.fmean(1 / r for r in ranks)
    return f"{hits}  MRR {mrr:.3f}  (n={len(ranks)})"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--model", action="append", help="repeatable; default: the configured model"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--show-misses", action="store_true", help="list questions ranked past 5")
    args = parser.parse_args(argv)

    # Ingestion logs one line per stage per file; the result would be lost in them.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    models = args.model or [Settings(_env_file=None).embedding_model]
    chunks = asyncio.run(_chunk_corpus(args.corpus))
    questions = _load_questions(args.questions)
    print(f"{len(chunks)} chunks from {args.corpus.name}, {len(questions)} questions")

    for model_name in models:
        started = time.perf_counter()
        ranks = _ranks(model_name, chunks, questions)
        print(f"\n{model_name}  ({time.perf_counter() - started:.1f} s)")
        for lang in sorted({q.lang for q in questions}):
            subset = [r for r, q in zip(ranks, questions, strict=True) if q.lang == lang]
            print(f"  {lang}: {_summary(subset)}")
        print(f"  all: {_summary(ranks)}")
        if args.show_misses:
            for rank, question in zip(ranks, questions, strict=True):
                if rank > max(CUTOFFS):
                    print(f"    rank {rank:2d}  {question.id}  {question.text}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
