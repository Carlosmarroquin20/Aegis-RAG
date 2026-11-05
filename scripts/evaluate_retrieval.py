"""
Measures how well the retrieval pipeline finds the right document for a question.

Usage:
    uv run python scripts/evaluate_retrieval.py
    uv run python scripts/evaluate_retrieval.py --compare
    uv run python scripts/evaluate_retrieval.py --mode hybrid+rerank
    uv run python scripts/evaluate_retrieval.py \\
        --model sentence-transformers/all-MiniLM-L6-v2 \\
        --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    uv run python scripts/evaluate_retrieval.py --corpus examples/atencion-cliente \\
        --questions evals/retrieval/atencion-cliente.jsonl
    uv run python scripts/evaluate_retrieval.py --min-hit5 0.94 --min-mrr 0.82

Retrieval is the half of RAG that decides whether the model ever sees the
answer: a question whose document is not in the top results is answered with
"not enough information" however good the LLM is. This script ingests a corpus
through the application's own pipeline (type detection, parsing, chunking), so
the chunks are the ones the API would store, and retrieves them through the
application's own ``Retriever``, BM25 index and reranker, so the ranking is the
one the API would produce. Only the vector store is replaced, by an in-memory
one that embeds with the same model and the same cosine similarity. For each
question it reports where the first chunk of the expected document ranks among
the top 20; one ranked lower counts as a miss. Figures are broken down by
language and by kind of question: ``direct`` ones use the document's own terms
(or their translation), ``paraphrase`` ones ask the same thing in other words,
and ``scenario`` ones describe a situation the policy has to be inferred from.

``--mode`` picks the retrieval stages (default: the configured RETRIEVAL_MODE)
and is the one the thresholds apply to. ``--compare`` also prints every other
mode, plus BM25 alone, for reference.

It needs the embedding model, and for the rerank modes the reranker, locally
(downloaded on first use). Results are printed rather than written, because they
depend on the model files and not only on the code. With --min-hit5 and
--min-mrr it is a quality gate: it exits non-zero when the gated mode scores
below either threshold over all questions. CI runs it that way, so a change to
parsing, chunking, retrieval or the default models that makes retrieval worse
fails the build instead of reaching users as unanswered questions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from aegis.application.dtos.ingestion_dtos import IngestRequest, UploadedFile
from aegis.application.services.retrieval import Retriever
from aegis.application.use_cases.ingest_documents import IngestDocumentsUseCase
from aegis.config import Settings
from aegis.domain.models.document import Document
from aegis.domain.ports.vector_store import VectorStorePort
from aegis.infrastructure.parsers.parser_registry import ParserRegistry
from aegis.infrastructure.retrieval.bm25 import StoreBackedLexicalIndex
from aegis.infrastructure.retrieval.cross_encoder import CrossEncoderReranker
from aegis.infrastructure.vector_stores.embedding_prompts import embedding_prompts

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CORPUS = ROOT / "examples" / "atencion-cliente"
DEFAULT_QUESTIONS = ROOT / "evals" / "retrieval" / "atencion-cliente.jsonl"
CUTOFFS = (1, 3, 5)


# Ranks are looked for among this many results; deeper is a miss.
DEPTH = 20
MISS = math.inf
MODES = ("dense", "hybrid", "dense+rerank", "hybrid+rerank")
LEXICAL_ONLY = "bm25"


class MemoryStore(VectorStorePort):
    """
    Collects the chunks the ingest use case produces, then serves dense search
    over them with the given model: normalized vectors, so the dot product is the
    cosine similarity the Chroma collection uses ("hnsw:space": "cosine").
    """

    def __init__(self) -> None:
        self.chunks: list[Document] = []
        self._model: Any | None = None
        self._vectors: Any | None = None
        self._query_prompt = ""

    def embed_with(self, model: Any, model_name: str = "") -> None:
        """Embeds every chunk with ``model``, marked the way ``model_name`` expects."""
        prompts = embedding_prompts(model_name)
        self._model = model
        self._query_prompt = prompts.query
        self._vectors = model.encode(
            [prompts.passage + c.content for c in self.chunks], normalize_embeddings=True
        )

    async def similarity_search(
        self, query: str, k: int = 5, score_threshold: float = 0.0
    ) -> list[Document]:
        assert self._model is not None and self._vectors is not None, "call embed_with() first"
        vector = self._model.encode([self._query_prompt + query], normalize_embeddings=True)[0]
        scores = self._vectors @ vector
        order = sorted(range(len(self.chunks)), key=lambda i: -float(scores[i]))[:k]
        return [
            self.chunks[i].model_copy(
                update={"relevance_score": round(max(0.0, min(1.0, float(scores[i]))), 4)}
            )
            for i in order
        ]

    async def add_documents(self, documents: list[Document]) -> None:
        self.chunks.extend(documents)

    async def delete_documents(self, ids: list[str]) -> None:
        raise NotImplementedError

    async def existing_ids(self, ids: list[str]) -> set[str]:
        return set()

    async def count(self) -> int:
        return len(self.chunks)

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
    kind: str = ""


async def chunk_corpus(corpus: Path, request: IngestRequest | None = None) -> MemoryStore:
    """The corpus, chunked by the application's ingestion pipeline with ``request``'s sizes."""
    store = MemoryStore()
    use_case = IngestDocumentsUseCase(
        vector_store=store, parser_registry=ParserRegistry(), collection="retrieval-eval"
    )
    request = request or IngestRequest()
    for path in sorted(corpus.rglob("*")):
        if path.is_file():
            upload = UploadedFile(
                filename=path.relative_to(corpus).as_posix(), content=path.read_bytes()
            )
            await use_case.execute(upload, request)
    return store


def _load_questions(path: Path) -> list[Question]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions.append(
                Question(
                    record["id"],
                    record["lang"],
                    record["question"],
                    record["expected_source"],
                    record.get("kind", ""),
                )
            )
    return questions


def _rank(results: Sequence[Document], expected_source: str) -> float:
    """1-based rank of the first result from the expected document, or MISS."""
    for position, document in enumerate(results, start=1):
        if document.metadata.get("source") == expected_source:
            return position
    return MISS


async def _ranks(
    mode: str,
    store: MemoryStore,
    questions: Sequence[Question],
    *,
    reranker: CrossEncoderReranker | None,
    candidates: int,
) -> list[float]:
    """The rank of the expected document for every question, retrieving with ``mode``."""
    lexical = StoreBackedLexicalIndex(store)
    if mode == LEXICAL_ONLY:
        return [_rank(await lexical.search(q.text, DEPTH), q.expected_source) for q in questions]
    retriever = Retriever(
        store,
        lexical_index=lexical if mode.startswith("hybrid") else None,
        reranker=reranker if mode.endswith("+rerank") else None,
        candidates=candidates,
    )
    return [_rank(await retriever.retrieve(q.text, DEPTH), q.expected_source) for q in questions]


def _summary(ranks: Sequence[float]) -> str:
    hits = "  ".join(f"hit@{k} {sum(r <= k for r in ranks) / len(ranks):6.1%}" for k in CUTOFFS)
    mrr = statistics.fmean(1 / r for r in ranks)
    return f"{hits}  MRR {mrr:.3f}  (n={len(ranks)})"


def shortfalls(
    ranks: Sequence[float], *, min_hit5: float | None, min_mrr: float | None
) -> list[str]:
    """The thresholds the ranks fall short of, as readable lines; empty when all are met."""
    problems = []
    hit5 = sum(r <= 5 for r in ranks) / len(ranks)
    mrr = statistics.fmean(1 / r for r in ranks)
    if min_hit5 is not None and hit5 < min_hit5:
        problems.append(f"hit@5 {hit5:.3f} is below the required {min_hit5:.3f}")
    if min_mrr is not None and mrr < min_mrr:
        problems.append(f"MRR {mrr:.3f} is below the required {min_mrr:.3f}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--model", action="append", help="embedding model; repeatable; default: the configured one"
    )
    parser.add_argument(
        "--mode", choices=MODES, help="retrieval stages to gate; default: RETRIEVAL_MODE"
    )
    parser.add_argument(
        "--compare", action="store_true", help="also print every other mode and BM25 alone"
    )
    parser.add_argument("--reranker-model", help="default: the configured RERANKER_MODEL")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--chunk-size", type=int, help="characters per chunk; default: the upload default"
    )
    parser.add_argument("--overlap", type=int, help="characters shared by neighbouring chunks")
    parser.add_argument("--show-misses", action="store_true", help="list questions ranked past 5")
    parser.add_argument("--min-hit5", type=float, help="fail below this hit@5 over all questions")
    parser.add_argument("--min-mrr", type=float, help="fail below this MRR over all questions")
    args = parser.parse_args(argv)

    # Ingestion logs one line per stage per file; the result would be lost in them.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    settings = Settings(_env_file=None)
    gated = args.mode or settings.retrieval_mode
    modes = [gated] + ([m for m in (*MODES, LEXICAL_ONLY) if m != gated] if args.compare else [])
    models = args.model or [settings.embedding_model]
    # One event loop for the whole run: the reranker's lock belongs to a loop.
    runner = asyncio.Runner()
    sizes = {
        key: value
        for key, value in (("chunk_size", args.chunk_size), ("overlap", args.overlap))
        if value is not None
    }
    request = IngestRequest(**sizes)
    store = runner.run(chunk_corpus(args.corpus, request))
    questions = _load_questions(args.questions)
    print(
        f"{len(store.chunks)} chunks from {args.corpus.name} (chunk size {request.chunk_size}, "
        f"overlap {request.overlap}), {len(questions)} questions"
    )

    reranker = None
    if any(m.endswith("+rerank") for m in modes):
        reranker = CrossEncoderReranker(
            args.reranker_model or settings.reranker_model, device="cpu"
        )
        print(f"reranker: {reranker.model_name}")

    failed = False
    for model_name in models:
        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        store.embed_with(SentenceTransformer(model_name, device="cpu"), model_name)
        print(f"\n{model_name}  (loaded and embedded in {time.perf_counter() - started:.1f} s)")
        for mode in modes:
            started = time.perf_counter()
            ranks = runner.run(
                _ranks(
                    mode,
                    store,
                    questions,
                    reranker=reranker,
                    candidates=settings.retrieval_candidates,
                )
            )
            per_query_ms = (time.perf_counter() - started) * 1000 / len(questions)
            label = f"{mode}  (gated)" if mode == gated else mode
            print(f"\n  {label}  {per_query_ms:.0f} ms/query")
            for lang in sorted({q.lang for q in questions}):
                subset = [r for r, q in zip(ranks, questions, strict=True) if q.lang == lang]
                print(f"    {lang}: {_summary(subset)}")
            print(f"    all: {_summary(ranks)}")
            for kind in sorted({q.kind for q in questions} - {""}):
                subset = [r for r, q in zip(ranks, questions, strict=True) if q.kind == kind]
                print(f"    {kind}: {_summary(subset)}")
            if args.show_misses:
                for rank, question in zip(ranks, questions, strict=True):
                    if rank > max(CUTOFFS):
                        shown = "  -" if rank == MISS else f"{int(rank):3d}"
                        print(f"      rank {shown}  {question.id}  {question.text}")
            if mode != gated:
                continue
            for problem in shortfalls(ranks, min_hit5=args.min_hit5, min_mrr=args.min_mrr):
                print(f"    FAIL: {problem}")
                failed = True
    runner.close()
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
