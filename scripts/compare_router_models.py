"""
Compares the router's small model with the large one, live, on what it would send the small one.

Usage:
    uv run python scripts/compare_router_models.py --small openai/gpt-oss-20b \\
        --out evals/routing/live/gpt-oss-20b.jsonl
    uv run python scripts/compare_router_models.py --summary evals/routing/live/gpt-oss-20b.jsonl

scripts/evaluate_router.py measures the routing decision; this measures what it
costs in answers (ADR 0026). Every question in evals/routing/dataset.jsonl that
``assess_complexity`` routes to the small model is answered twice through the
API's own query pipeline, from .env (gateway, retrieval as RETRIEVAL_MODE sets
it, the output sanitizer and the faithfulness judge): once by the small model,
once by LLM_MODEL. Both see the same retrieved context.

Per question it records, for each model: the answer, whether it abstained, its
faithfulness to the context (the NLI judge of ADR 0017), latency and tokens.
Then the same judge compares the two answers with each other, with no third
model involved: how much of the large model's answer the small one also says
(coverage), and how much of the small one's the large one supports
(agreement). The large model is the reference, not ground truth.

The NLI comparison is cheap but blind to terse answers ("Q55." against "El
envío cuesta Q55." reads as uncovered). --judge adds a second opinion on every
pair both models answered: a model of another family (not the large model
judging its own answers) says whether the small answer is equivalent to the
large one, omits a detail, omits a condition that could change what the
customer does, or contradicts it. The answer's language is checked separately
and deterministically, with the router's own check (ADR 0027).

Results are appended one line per question as they arrive, and a rerun skips
the questions already in the file, so a run cut short by a provider's daily
limit resumes where it stopped. A rate-limited call waits as long as the
provider's Retry-After asks and tries again.

It sends the questions and the example corpus's excerpts to the provider in
.env; it is not run in CI.
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
from typing import Any

import httpx
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_retrieval import DEFAULT_CORPUS, MemoryStore, chunk_corpus  # noqa: E402

from aegis.application.dtos.rag_dtos import QueryRequest  # noqa: E402
from aegis.application.services.faithfulness import NLIFaithfulnessJudge  # noqa: E402
from aegis.application.use_cases.query_rag import (  # noqa: E402
    QueryRAGUseCase,
    SecurityViolationError,
)
from aegis.config import Settings, get_settings  # noqa: E402
from aegis.domain.models.document import Document  # noqa: E402
from aegis.domain.models.generation import Generation  # noqa: E402
from aegis.domain.ports.llm_client import (  # noqa: E402
    LLMClientPort,
    LLMRateLimitedError,
    LLMUnavailableError,
)
from aegis.domain.services.language import detect_language  # noqa: E402
from aegis.domain.services.query_complexity import Tier, assess_complexity  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evals" / "routing" / "dataset.jsonl"
_MAX_ATTEMPTS = 8

VERDICTS = ("equivalent", "omits_detail", "omits_condition", "contradicts")

_JUDGE_PROMPT = """You compare two answers to a customer's question about a store's policies.
The reference answer is taken as correct. Classify the candidate answer:
- "equivalent": the same facts, and every condition or limit that matters to the customer,
  in other words.
- "omits_detail": it leaves out something from the reference that would not change what the
  customer does.
- "omits_condition": it leaves out a condition, limit or exception from the reference that
  could change what the customer does.
- "contradicts": it states something the reference contradicts.
Ignore language, tone and length. Use only the two answers, no outside knowledge.
Reply with JSON: {"verdict": "<one of the four>", "note": "<at most 15 words>"}"""


class _Recording(LLMClientPort):
    """Delegates to ``inner`` and keeps the last generation and how long it took."""

    def __init__(self, inner: LLMClientPort) -> None:
        self.inner = inner
        self.last: Generation | None = None
        self.seconds = 0.0

    async def generate(
        self,
        query: str,
        context: list[Document],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Generation:
        start = time.perf_counter()
        self.last = await self.inner.generate(
            query, context, max_tokens=max_tokens, temperature=temperature
        )
        self.seconds = time.perf_counter() - start
        return self.last

    async def health_check(self) -> bool:
        return await self.inner.health_check()

    async def aclose(self) -> None:
        await self.inner.aclose()


@dataclass
class _Side:
    model: str
    llm: _Recording
    use_case: QueryRAGUseCase


async def _answer(side: _Side, question: str) -> dict[str, Any]:
    for _ in range(_MAX_ATTEMPTS):
        try:
            response = await side.use_case.execute(QueryRequest(query=question))
        except LLMRateLimitedError as exc:
            wait = min(90.0, (exc.retry_after or 10.0) + 1.0)
            print(f"    {side.model}: rate limited, waiting {wait:.0f}s", flush=True)
            await asyncio.sleep(wait)
            continue
        except SecurityViolationError:
            return {"error": "blocked_by_gateway"}
        except LLMUnavailableError as exc:
            return {"error": f"unavailable: {exc}"[:200]}
        report = response.faithfulness
        generation = side.llm.last
        assert report is not None and generation is not None
        usage = generation.usage
        return {
            "answer": response.answer,
            "abstained": report.abstained,
            "faithfulness": report.score,
            "unsupported": report.unsupported,
            "seconds": round(side.llm.seconds, 3),
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
        }
    return {"error": "rate_limited"}


async def _agreement(
    judge: NLIFaithfulnessJudge, question: str, small: dict[str, Any], large: dict[str, Any]
) -> dict[str, float | None]:
    if "answer" not in small or "answer" not in large:
        return {"coverage": None, "agreement": None}
    if small["abstained"] or large["abstained"]:
        both = small["abstained"] and large["abstained"]
        return {"coverage": 1.0 if both else 0.0, "agreement": 1.0 if both else 0.0}

    def _as_context(text: str) -> list[Document]:
        # Read with the question it answers: a terse "15 días." entails nothing
        # on its own, but "¿Cuántos días tengo…? 15 días." does.
        return [Document(id="reference", content=f"{question} {text}", metadata={})]

    covered = await judge.assess(large["answer"], _as_context(small["answer"]), question=question)
    agreed = await judge.assess(small["answer"], _as_context(large["answer"]), question=question)
    return {"coverage": covered.score, "agreement": agreed.score}


async def _judge_pair(
    http: httpx.AsyncClient, model: str, record: dict[str, Any]
) -> dict[str, str]:
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 400,
        "reasoning_format": "hidden",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _JUDGE_PROMPT},
            {
                "role": "user",
                "content": f"Question: {record['question']}\n"
                f"Reference answer: {record['large']['answer']}\n"
                f"Candidate answer: {record['small']['answer']}",
            },
        ],
    }
    for _ in range(_MAX_ATTEMPTS):
        response = await http.post("/chat/completions", json=body)
        if response.status_code == 429:
            wait = min(90.0, float(response.headers.get("retry-after", "10")) + 1.0)
            print(f"    judge: rate limited, waiting {wait:.0f}s", flush=True)
            await asyncio.sleep(wait)
            continue
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {"verdict": "unparsed", "note": content[:120]}
        verdict = str(parsed.get("verdict", ""))
        return {
            "verdict": verdict if verdict in VERDICTS else "unparsed",
            "note": str(parsed.get("note", ""))[:200],
        }
    return {"verdict": "rate_limited", "note": ""}


def _judgeable(record: dict[str, Any]) -> bool:
    small, large = record["small"], record["large"]
    return (
        "answer" in small
        and "answer" in large
        and not small["abstained"]
        and not large["abstained"]
    )


async def _annotate(settings: Settings, path: Path, model: str) -> None:
    records = [json.loads(line) for line in _lines(path)]
    pending = [r for r in records if _judgeable(r) and "judge" not in r]
    print(f"judge {model}: {len(pending)} pairs to classify")
    async with httpx.AsyncClient(
        base_url=settings.llm_base_url,
        headers={"Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}"},
        timeout=60,
    ) as http:
        for number, record in enumerate(pending, start=1):
            record["judge"] = {"model": model, **await _judge_pair(http, model, record)}
            print(f"  [{number}/{len(pending)}] {record['id']} {record['judge']['verdict']}")
            # Rewritten after every verdict, so an interrupted run keeps them.
            path.write_bytes(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
            )


def _routed_small(dataset: Path) -> list[dict[str, Any]]:
    rows = []
    for line in dataset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assessment = assess_complexity(row["question"], context_size=5)
        if assessment.tier is Tier.SIMPLE:
            rows.append(row)
    return rows


async def _run(settings: Settings, args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer

    from aegis.application.services.retrieval import Retriever
    from aegis.infrastructure.faithfulness.nli_model import TransformersNLIModel
    from aegis.infrastructure.llm.openai_compatible_adapter import OpenAICompatibleAdapter
    from aegis.infrastructure.observability.metrics import PrometheusQueryMetrics
    from aegis.infrastructure.retrieval.bm25 import StoreBackedLexicalIndex
    from aegis.infrastructure.retrieval.cross_encoder import CrossEncoderReranker
    from aegis.infrastructure.security.output_sanitizer import OutputSanitizer
    from aegis.infrastructure.security.security_gateway import SecurityGateway

    store: MemoryStore = await chunk_corpus(DEFAULT_CORPUS)
    store.embed_with(
        SentenceTransformer(settings.embedding_model, device="cpu"), settings.embedding_model
    )
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

    def side(model: str) -> _Side:
        llm = _Recording(
            OpenAICompatibleAdapter(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key.get_secret_value(),
                model=model,
                timeout_seconds=settings.llm_timeout_seconds,
            )
        )
        use_case = QueryRAGUseCase(
            vector_store=store,
            retriever=retriever,
            llm_client=llm,
            security_gateway=SecurityGateway(strict_mode=settings.security_strict_mode),
            output_sanitizer=OutputSanitizer(),
            metrics=PrometheusQueryMetrics(),
            faithfulness_judge=judge,
        )
        return _Side(model, llm, use_case)

    small, large = side(args.small), side(settings.llm_model)
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in _lines(out)}
    questions = [q for q in _routed_small(args.dataset) if q["id"] not in done]
    if args.limit:
        questions = questions[: args.limit]
    print(f"small {small.model} vs large {large.model}: {len(questions)} questions to run")

    try:
        for number, row in enumerate(questions, start=1):
            small_result, large_result = await asyncio.gather(
                _answer(small, row["question"]), _answer(large, row["question"])
            )
            record = {
                "id": row["id"],
                "lang": row["lang"],
                "label": row["label"],
                "kind": row["kind"],
                "question": row["question"],
                "small_model": small.model,
                "large_model": large.model,
                "small": small_result,
                "large": large_result,
                **await _agreement(judge, row["question"], small_result, large_result),
            }
            with out.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"  [{number}/{len(questions)}] {row['id']} "
                f"coverage={_fmt(record['coverage'])} agreement={_fmt(record['agreement'])}",
                flush=True,
            )
    finally:
        await small.llm.aclose()
        await large.llm.aclose()


def _lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.1%} ({part}/{whole})" if whole else "n/a"


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def summarise(records: Sequence[dict[str, Any]]) -> str:
    """The comparison as plain text; also what the documentation quotes."""
    ok = [r for r in records if "answer" in r["small"] and "answer" in r["large"]]
    lines = [f"questions: {len(records)}, answered by both: {len(ok)}"]
    failed = len(records) - len(ok)
    if failed:
        lines.append(f"  failed or blocked on either side: {failed}")
    if not ok:
        return "\n".join(lines)

    for name in ("small", "large"):
        answers = [r[name] for r in ok]
        scored = [a["faithfulness"] for a in answers if not a["abstained"]]
        latency = [a["seconds"] for a in answers]
        output = [a["output_tokens"] for a in answers if a["output_tokens"] is not None]
        inputs = [a["input_tokens"] for a in answers if a["input_tokens"] is not None]
        lines += [
            f"{name} ({answers and ok[0][name + '_model']}):",
            f"  abstained:            {_pct(sum(a['abstained'] for a in answers), len(answers))}",
            f"  mean faithfulness:    {statistics.fmean(scored):.3f} over {len(scored)} answers"
            if scored
            else "  mean faithfulness:    n/a",
            f"  fully supported:      {_pct(sum(s == 1.0 for s in scored), len(scored))}",
            f"  latency median / p95: {statistics.median(latency):.2f}s / "
            f"{_quantile(latency, 0.95):.2f}s",
            f"  tokens per answer:    {statistics.fmean(inputs):.0f} in, "
            f"{statistics.fmean(output):.0f} out"
            if inputs and output
            else "  tokens per answer:    n/a",
        ]

    def abstained(record: dict[str, Any], side: str) -> bool:
        return bool(record[side]["abstained"])

    small_only_abstained = [r for r in ok if abstained(r, "small") and not abstained(r, "large")]
    large_only_abstained = [r for r in ok if abstained(r, "large") and not abstained(r, "small")]
    both_answered = [
        r
        for r in ok
        if r["coverage"] is not None and not abstained(r, "small") and not abstained(r, "large")
    ]
    lines += [
        "small vs large:",
        f"  small abstained, large answered: {_pct(len(small_only_abstained), len(ok))}",
        f"  large abstained, small answered: {_pct(len(large_only_abstained), len(ok))}",
    ]
    if both_answered:
        coverage = [r["coverage"] for r in both_answered]
        agreement = [r["agreement"] for r in both_answered]
        lines += [
            f"  where both answered ({len(both_answered)}):",
            f"    mean coverage of the large answer:  {statistics.fmean(coverage):.3f}",
            f"    mean agreement with the large one:  {statistics.fmean(agreement):.3f}",
            f"    small says less than half of large: "
            f"{_pct(sum(c < 0.5 for c in coverage), len(coverage))}",
        ]
    english = [r for r in ok if r["lang"] == "en"]
    for name in ("small", "large"):
        wrong = [r for r in ok if detect_language(r[name]["answer"]) not in (None, r["lang"])]
        lines.append(
            f"  {name} answered in another language: {_pct(len(wrong), len(ok))}"
            f" ({sum(r['lang'] == 'en' for r in wrong)} of {len(english)} English questions)"
        )
    judged = [r for r in both_answered if "judge" in r]
    if judged:
        lines.append(f"  judged by {judged[0]['judge']['model']} ({len(judged)} pairs):")
        for verdict in (*VERDICTS, "unparsed", "rate_limited"):
            count = sum(r["judge"]["verdict"] == verdict for r in judged)
            if count:
                lines.append(f"    {verdict:<16} {_pct(count, len(judged))}")
    for label in ("simple", "complex"):
        subset = [r for r in both_answered if r["label"] == label]
        if subset:
            lines.append(
                f"  {label} questions (labelled), mean coverage: "
                f"{statistics.fmean(r['coverage'] for r in subset):.3f} over {len(subset)}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--small", help="the small model, served by LLM_BASE_URL")
    parser.add_argument("--out", type=Path, help="results file (JSON lines); appended to")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, help="run at most this many new questions")
    parser.add_argument("--summary", type=Path, help="summarise a results file and exit")
    parser.add_argument("--judge", help="classify the answer pairs in --out with this model")
    args = parser.parse_args(argv)

    if args.summary:
        print(summarise([json.loads(line) for line in _lines(args.summary)]))
        return 0
    if args.judge:
        if not args.out:
            parser.error("--judge needs --out, the results file to annotate")
        asyncio.run(_annotate(get_settings(), args.out, args.judge))
        print(summarise([json.loads(line) for line in _lines(args.out)]))
        return 0
    if not args.small or not args.out:
        parser.error("--small and --out are required to run the comparison")

    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    settings = get_settings()
    if settings.llm_provider != "openai_compatible":
        parser.error("LLM_PROVIDER must be openai_compatible: both models are hosted here")
    if settings.faithfulness_model == "":
        parser.error("FAITHFULNESS_MODEL is required")
    asyncio.run(_run(settings, args))
    print()
    print(summarise([json.loads(line) for line in _lines(args.out)]))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
