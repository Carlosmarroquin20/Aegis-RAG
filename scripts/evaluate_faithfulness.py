"""
Measures how well the faithfulness judge tells supported answers from invented ones.

Usage:
    uv run python scripts/evaluate_faithfulness.py
    uv run python scripts/evaluate_faithfulness.py --show-errors
    uv run python scripts/evaluate_faithfulness.py --compare-model <another NLI model>
    uv run python scripts/evaluate_faithfulness.py --min-recall 0.8 --max-false-flags 0.15
    uv run python scripts/evaluate_faithfulness.py --live --min-score 0.8

Default mode: the judge reads labelled answers (evals/faithfulness/), each written
against the example corpus and either faithful to it or not in a named way: a
wrong figure, a contradiction, an invented addition, a claim from outside
knowledge. The context for an answer is every chunk of its source document,
chunked by the application's own ingestion pipeline. The report says how many
unfaithful answers were caught (recall) and how many faithful ones were wrongly
flagged, per kind and per language. With --min-recall and --max-false-flags it is
a quality gate, and CI runs it that way: the judge is deterministic, so the
figures only move when the code, the model or the labels do.

--live instead runs the API's own query pipeline end to end (retrieval as
RETRIEVAL_MODE configures it, the configured LLM, the output sanitizer) over the
retrieval questions, with the faithfulness check on, and reports how faithful the
answers actually are. It needs the embedding model and a reachable LLM (Ollama
from docker compose, or a hosted provider from .env), so it is not run in CI;
--min-score turns it into a gate wherever it is run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_retrieval import (  # noqa: E402  (sibling script, not a package)
    DEFAULT_CORPUS,
    DEFAULT_QUESTIONS,
    MemoryStore,
    chunk_corpus,
)

from aegis.application.dtos.rag_dtos import QueryRequest  # noqa: E402
from aegis.application.services.faithfulness import NLIFaithfulnessJudge  # noqa: E402
from aegis.config import Settings, get_settings  # noqa: E402
from aegis.domain.models.document import Document  # noqa: E402
from aegis.domain.models.faithfulness import FaithfulnessAssessment  # noqa: E402
from aegis.domain.ports.faithfulness import FaithfulnessJudgePort  # noqa: E402
from aegis.infrastructure.faithfulness.nli_model import TransformersNLIModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWERS = ROOT / "evals" / "faithfulness" / "atencion-cliente.jsonl"
LABELS = ("faithful", "unfaithful")


@dataclass(frozen=True, slots=True)
class LabelledAnswer:
    id: str
    question: str
    lang: str
    source: str
    kind: str
    label: str
    answer: str

    @property
    def unfaithful(self) -> bool:
        return self.label == "unfaithful"


@dataclass(frozen=True, slots=True)
class Outcome:
    item: LabelledAnswer
    assessment: FaithfulnessAssessment

    @property
    def flagged(self) -> bool:
        """The judge says the answer has at least one unsupported claim."""
        return not self.assessment.fully_supported

    @property
    def correct(self) -> bool:
        return self.flagged == self.item.unfaithful


@dataclass(frozen=True, slots=True)
class Summary:
    caught: int
    unfaithful: int
    false_flags: int
    faithful: int

    @property
    def recall(self) -> float:
        return self.caught / self.unfaithful if self.unfaithful else 1.0

    @property
    def false_flag_rate(self) -> float:
        return self.false_flags / self.faithful if self.faithful else 0.0

    @property
    def accuracy(self) -> float:
        total = self.unfaithful + self.faithful
        return (self.caught + self.faithful - self.false_flags) / total if total else 1.0


def load_answers(path: Path) -> list[LabelledAnswer]:
    answers = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record["label"] not in LABELS:
            raise ValueError(f"{path.name}:{number}: label must be one of {LABELS}")
        answers.append(
            LabelledAnswer(
                id=record["id"],
                question=record["question"],
                lang=record["lang"],
                source=record["source"],
                kind=record["kind"],
                label=record["label"],
                answer=record["answer"],
            )
        )
    return answers


def summarise(outcomes: Sequence[Outcome]) -> Summary:
    return Summary(
        caught=sum(o.flagged for o in outcomes if o.item.unfaithful),
        unfaithful=sum(o.item.unfaithful for o in outcomes),
        false_flags=sum(o.flagged for o in outcomes if not o.item.unfaithful),
        faithful=sum(not o.item.unfaithful for o in outcomes),
    )


def shortfalls(
    summary: Summary, *, min_recall: float | None, max_false_flags: float | None
) -> list[str]:
    """The thresholds the judge falls short of, as readable lines; empty when all are met."""
    problems = []
    if min_recall is not None and summary.recall < min_recall:
        problems.append(f"recall {summary.recall:.3f} is below the required {min_recall:.3f}")
    if max_false_flags is not None and summary.false_flag_rate > max_false_flags:
        problems.append(
            f"false-flag rate {summary.false_flag_rate:.3f} is above the allowed "
            f"{max_false_flags:.3f}"
        )
    return problems


async def judge_labelled(
    judge: FaithfulnessJudgePort, answers: Sequence[LabelledAnswer], chunks: Sequence[Document]
) -> list[Outcome]:
    by_source: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.metadata.get("source", "")].append(chunk)
    return [
        Outcome(item, await judge.assess(item.answer, by_source[item.source])) for item in answers
    ]


def _line(summary: Summary) -> str:
    return (
        f"caught {summary.caught}/{summary.unfaithful} unfaithful (recall {summary.recall:.1%})  "
        f"flagged {summary.false_flags}/{summary.faithful} faithful "
        f"({summary.false_flag_rate:.1%})  accuracy {summary.accuracy:.1%}"
    )


def _report(outcomes: Sequence[Outcome], *, show_errors: bool) -> None:
    print(f"    all: {_line(summarise(outcomes))}")
    for lang in sorted({o.item.lang for o in outcomes}):
        print(f"    {lang}:  {_line(summarise([o for o in outcomes if o.item.lang == lang]))}")
    print("    by kind (share flagged):")
    for kind in sorted({o.item.kind for o in outcomes}):
        subset = [o for o in outcomes if o.item.kind == kind]
        flagged = sum(o.flagged for o in subset)
        label = subset[0].item.label
        print(f"      {kind:24s} {label:10s} {flagged:2d}/{len(subset):2d}")
    if show_errors:
        for outcome in outcomes:
            if outcome.correct:
                continue
            verdict = "missed" if outcome.item.unfaithful else "false flag"
            print(
                f"      {verdict:10s} {outcome.item.id} [{outcome.item.kind}] {outcome.item.answer}"
            )
            for sentence in outcome.assessment.sentences:
                mark = "ok" if sentence.supported else "--"
                print(f"        {mark} {sentence.support:.2f}  {sentence.text}")


async def _live(settings: Settings, store: MemoryStore, questions_path: Path) -> list[float]:
    """Runs the API's own query pipeline over the questions, faithfulness check on."""
    from sentence_transformers import SentenceTransformer

    from aegis.application.services.retrieval import Retriever
    from aegis.application.use_cases.query_rag import QueryRAGUseCase
    from aegis.infrastructure.observability.metrics import PrometheusQueryMetrics
    from aegis.infrastructure.retrieval.bm25 import StoreBackedLexicalIndex
    from aegis.infrastructure.retrieval.cross_encoder import CrossEncoderReranker
    from aegis.infrastructure.security.output_sanitizer import OutputSanitizer
    from aegis.infrastructure.security.security_gateway import SecurityGateway
    from aegis.interface.api.dependencies import get_llm_client

    store.embed_with(SentenceTransformer(settings.embedding_model, device="cpu"))
    retriever = Retriever(
        store,
        lexical_index=StoreBackedLexicalIndex(store) if settings.lexical_search_enabled else None,
        reranker=CrossEncoderReranker(settings.reranker_model)
        if settings.reranker_enabled
        else None,
        candidates=settings.retrieval_candidates,
    )
    judge = NLIFaithfulnessJudge(
        TransformersNLIModel(settings.faithfulness_model),
        threshold=settings.faithfulness_threshold,
    )
    llm = get_llm_client()
    use_case = QueryRAGUseCase(
        vector_store=store,
        retriever=retriever,
        llm_client=llm,
        security_gateway=SecurityGateway(strict_mode=settings.security_strict_mode),
        output_sanitizer=OutputSanitizer(),
        metrics=PrometheusQueryMetrics(),
        faithfulness_judge=judge,
    )
    scores: list[float] = []
    try:
        for line in questions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            response = await use_case.execute(QueryRequest(query=record["question"]))
            report = response.faithfulness
            assert report is not None
            label = "abstained" if report.abstained else f"{report.score:.2f}"
            print(f"  {record['id']}  {label:9s}  {record['question']}")
            for sentence in report.unsupported:
                print(f"      unsupported: {sentence}")
            if not report.abstained:
                scores.append(report.score)
    finally:
        await llm.aclose()
    return scores


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--model", help="NLI model to gate; default: FAITHFULNESS_MODEL")
    parser.add_argument(
        "--compare-model", action="append", default=[], help="also report this model; repeatable"
    )
    parser.add_argument("--threshold", type=float, help="default: FAITHFULNESS_THRESHOLD")
    parser.add_argument(
        "--granularity",
        choices=("sentence", "chunk"),
        default="sentence",
        help="premises: context sentences (default) or whole chunks",
    )
    parser.add_argument(
        "--no-contradiction",
        action="store_true",
        help="support is entailment alone, not entailment minus contradiction",
    )
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--show-errors", action="store_true", help="list every misjudged answer")
    parser.add_argument("--min-recall", type=float, help="fail below this share of caught answers")
    parser.add_argument(
        "--max-false-flags", type=float, help="fail above this share of faithful answers flagged"
    )
    parser.add_argument("--live", action="store_true", help="judge the live pipeline's answers")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--min-score", type=float, help="--live: fail below this mean score")
    args = parser.parse_args(argv)

    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    # --live talks to the deployment's own LLM, so it reads .env like the API does;
    # the labelled evaluation must not depend on anyone's .env.
    settings = get_settings() if args.live else Settings(_env_file=None)
    if args.threshold is not None:
        settings = settings.model_copy(update={"faithfulness_threshold": args.threshold})
    if args.model:
        settings = settings.model_copy(update={"faithfulness_model": args.model})

    runner = asyncio.Runner()
    store = runner.run(chunk_corpus(args.corpus))
    try:
        if args.live:
            return _main_live(runner, settings, store, args)
        return _main_labelled(runner, settings, store, args)
    finally:
        runner.close()


def _main_labelled(
    runner: asyncio.Runner, settings: Settings, store: MemoryStore, args: argparse.Namespace
) -> int:
    answers = load_answers(args.answers)
    print(
        f"{len(answers)} labelled answers over {args.corpus.name} "
        f"({len(store.chunks)} chunks), threshold {settings.faithfulness_threshold}, "
        f"{args.granularity} premises, "
        f"{'entailment only' if args.no_contradiction else 'entailment minus contradiction'}"
    )
    failed = False
    gated = settings.faithfulness_model
    for model_name in [gated, *args.compare_model]:
        judge = NLIFaithfulnessJudge(
            TransformersNLIModel(model_name),
            threshold=settings.faithfulness_threshold,
            granularity=args.granularity,
            contradiction_aware=not args.no_contradiction,
        )
        started = time.perf_counter()
        try:
            outcomes = runner.run(judge_labelled(judge, answers, store.chunks))
        except Exception as exc:
            if model_name == gated:
                raise
            # A reference model that cannot load must not hide the gated result.
            print(f"\n  {model_name}  could not be evaluated: {exc!r}")
            continue
        per_answer_ms = (time.perf_counter() - started) * 1000 / len(answers)
        label = f"{model_name}  (gated)" if model_name == gated else model_name
        print(f"\n  {label}  {per_answer_ms:.0f} ms/answer")
        _report(outcomes, show_errors=args.show_errors)
        if model_name != gated:
            continue
        summary = summarise(outcomes)
        for problem in shortfalls(
            summary, min_recall=args.min_recall, max_false_flags=args.max_false_flags
        ):
            print(f"    FAIL: {problem}")
            failed = True
    return 1 if failed else 0


def _main_live(
    runner: asyncio.Runner, settings: Settings, store: MemoryStore, args: argparse.Namespace
) -> int:
    print(
        f"live pipeline: retrieval {settings.retrieval_mode}, judge {settings.faithfulness_model}"
    )
    scores = runner.run(_live(settings, store, args.questions))
    if not scores:
        print("\nevery answer abstained; nothing to score")
        return 0
    mean = statistics.fmean(scores)
    full = sum(score == 1.0 for score in scores)
    print(f"\nmean faithfulness {mean:.3f}  fully supported {full}/{len(scores)}")
    if args.min_score is not None and mean < args.min_score:
        print(f"FAIL: mean faithfulness {mean:.3f} is below the required {args.min_score:.3f}")
        return 1
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
