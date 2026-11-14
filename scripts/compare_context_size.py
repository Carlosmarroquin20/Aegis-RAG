"""
Compares answers generated from 5 and from 10 retrieved passages, live.

Usage:
    uv run python scripts/compare_context_size.py --out evals/multihop/live/context-5-vs-10.jsonl
    uv run python scripts/compare_context_size.py --judge qwen/qwen3.8-27b --out <same file>
    uv run python scripts/compare_context_size.py --summary <same file>

ADR 0028 found that 10 passages bring every document a multi-document question
needs, where 5 miss one in seven. Whether that makes answers better, or dilutes
them for the single-document questions that are most of the traffic, is about
answers, so it is measured here before the default changes.

Questions: the 39 of evals/multihop/dataset.jsonl (29 that need two or three
documents, 10 controls) and 40 single-document questions drawn with a fixed
seed from the retrieval evaluation set. Each is answered twice by LLM_MODEL
through the API's own pipeline (gateway, hybrid retrieval, output sanitizer,
faithfulness judge), with top_k 5 and top_k 10; nothing else differs.

--judge then asks a model of another family to grade both answers against the
full text of the documents the question needs, which is the ground truth here:
correct, incomplete (true but missing a condition or a second policy) or wrong
(contradicts the documents, or says they do not cover what they do). It also
says which answer serves the customer better. The two answers are shown as A
and B in an order fixed per question by a seeded shuffle, so position cannot
favour one size.

Results are appended one line per question and a rerun skips those already in
the file. Sends the questions and the example corpus to the provider in .env;
not run in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_router_models import _Recording  # noqa: E402
from evaluate_retrieval import DEFAULT_CORPUS, chunk_corpus  # noqa: E402

from aegis.application.dtos.rag_dtos import QueryRequest  # noqa: E402
from aegis.application.use_cases.query_rag import (  # noqa: E402
    QueryRAGUseCase,
    SecurityViolationError,
)
from aegis.config import Settings, get_settings  # noqa: E402
from aegis.domain.ports.llm_client import LLMRateLimitedError, LLMUnavailableError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MULTIHOP_DATASET = ROOT / "evals" / "multihop" / "dataset.jsonl"
RETRIEVAL_DATASET = ROOT / "evals" / "retrieval" / "atencion-cliente.jsonl"
SIZES = (5, 10)
SINGLE_SAMPLE = 40
SEED = 28
_MAX_ATTEMPTS = 8
GRADES = ("correct", "incomplete", "wrong")

_JUDGE_PROMPT = """You grade two answers to a customer's question about a store, using the store's
policy documents below as the only ground truth.
For each answer, give one grade:
- "correct": it gives what the customer needs from these documents, including every condition,
  limit or second policy that changes what they should do, and nothing the documents contradict.
- "incomplete": what it says is true, but it leaves out a condition, limit or second policy that
  could change what the customer does.
- "wrong": it states something the documents contradict, or says the information is not
  available when the documents cover it.
Ignore style, length and language. Then say which answer serves the customer better: "A", "B"
or "tie".
Reply with JSON: {"A": "<grade>", "B": "<grade>", "better": "A|B|tie", "note": "<15 words>"}"""


def questions() -> list[dict[str, Any]]:
    """The multi-hop set plus a fixed sample of single-document retrieval questions."""
    rows = []
    for line in MULTIHOP_DATASET.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            rows.append(
                {
                    "id": record["id"],
                    "lang": record["lang"],
                    "kind": record["kind"],
                    "question": record["question"],
                    "sources": record["required_sources"],
                }
            )
    retrieval = [
        json.loads(line)
        for line in RETRIEVAL_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in random.Random(SEED).sample(retrieval, SINGLE_SAMPLE):
        rows.append(
            {
                "id": record["id"],
                "lang": record["lang"],
                "kind": "single",
                "question": record["question"],
                "sources": [record["expected_source"]],
            }
        )
    return rows


async def _answer(
    use_case: QueryRAGUseCase, llm: _Recording, question: str, k: int
) -> dict[str, Any]:
    for _ in range(_MAX_ATTEMPTS):
        try:
            response = await use_case.execute(QueryRequest(query=question, top_k=k))
        except LLMRateLimitedError as exc:
            wait = min(90.0, (exc.retry_after or 10.0) + 1.0)
            print(f"    top_k={k}: rate limited, waiting {wait:.0f}s", flush=True)
            await asyncio.sleep(wait)
            continue
        except SecurityViolationError:
            return {"error": "blocked_by_gateway"}
        except LLMUnavailableError as exc:
            return {"error": f"unavailable: {exc}"[:200]}
        report = response.faithfulness
        generation = llm.last
        assert report is not None and generation is not None
        usage = generation.usage
        return {
            "answer": response.answer,
            "sources": sorted({s.metadata.get("source", "") for s in response.sources}),
            "abstained": report.abstained,
            "faithfulness": report.score,
            "seconds": round(llm.seconds, 3),
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
        }
    return {"error": "rate_limited"}


async def _run(settings: Settings, out: Path, limit: int | None) -> None:
    from sentence_transformers import SentenceTransformer

    from aegis.application.services.faithfulness import NLIFaithfulnessJudge
    from aegis.application.services.retrieval import Retriever
    from aegis.infrastructure.faithfulness.nli_model import TransformersNLIModel
    from aegis.infrastructure.llm.openai_compatible_adapter import OpenAICompatibleAdapter
    from aegis.infrastructure.observability.metrics import PrometheusQueryMetrics
    from aegis.infrastructure.retrieval.bm25 import StoreBackedLexicalIndex
    from aegis.infrastructure.security.output_sanitizer import OutputSanitizer
    from aegis.infrastructure.security.security_gateway import SecurityGateway

    store = await chunk_corpus(DEFAULT_CORPUS)
    store.embed_with(
        SentenceTransformer(settings.embedding_model, device="cpu"), settings.embedding_model
    )
    retriever = Retriever(
        store,
        lexical_index=StoreBackedLexicalIndex(store) if settings.lexical_search_enabled else None,
        candidates=settings.retrieval_candidates,
    )
    judge = NLIFaithfulnessJudge(
        TransformersNLIModel(settings.faithfulness_model),
        threshold=settings.faithfulness_threshold,
    )
    llm = _Recording(
        OpenAICompatibleAdapter(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
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
    done = {json.loads(line)["id"] for line in _lines(out)}
    pending = [q for q in questions() if q["id"] not in done]
    if limit:
        pending = pending[:limit]
    print(f"model {settings.llm_model}: {len(pending)} questions, top_k {SIZES}")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for number, row in enumerate(pending, start=1):
            record = {**row, "model": settings.llm_model}
            for k in SIZES:
                # Sequential on purpose: the recording client keeps one generation.
                record[f"k{k}"] = await _answer(use_case, llm, row["question"], k)
            if any(record[f"k{k}"].get("error") == "rate_limited" for k in SIZES):
                # Usually the provider's daily quota. Stop without writing the
                # question, so a rerun once the quota frees up resumes from it.
                print(f"  {row['id']}: still rate limited, stopping; rerun to resume")
                return
            with out.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  [{number}/{len(pending)}] {row['id']}", flush=True)
    finally:
        await llm.aclose()


def _documents(sources: Sequence[str]) -> str:
    return "\n\n".join(
        f"=== {name} ===\n{(DEFAULT_CORPUS / name).read_text(encoding='utf-8')}" for name in sources
    )


def _order(record_id: str) -> tuple[int, int]:
    """Which size is shown as A, fixed per question."""
    first, second = SIZES
    return (
        (first, second) if random.Random(f"{SEED}:{record_id}").random() < 0.5 else (second, first)
    )


async def _judge_one(http: httpx.AsyncClient, model: str, record: dict[str, Any]) -> dict[str, Any]:
    a_size, b_size = _order(record["id"])
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
                "content": f"{_documents(record['sources'])}\n\n"
                f"Question: {record['question']}\n"
                f"Answer A: {record[f'k{a_size}']['answer']}\n"
                f"Answer B: {record[f'k{b_size}']['answer']}",
            },
        ],
    }
    for _ in range(_MAX_ATTEMPTS):
        response = await http.post("/chat/completions", json=body)
        if response.status_code == 429:
            await asyncio.sleep(min(90.0, float(response.headers.get("retry-after", "10")) + 1))
            continue
        response.raise_for_status()
        content = response.json()["choices"][0]["message"].get("content") or ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {"model": model, "unparsed": content[:200]}
        grades = {
            # The judge sometimes names the keys "grade_A"/"grade_B" despite the prompt.
            size: str(parsed.get(label) or parsed.get(f"grade_{label}") or "").strip().lower()
            for size, label in ((a_size, "A"), (b_size, "B"))
        }
        better = str(parsed.get("better", "")).strip()
        better = "tie" if better.lower() == "tie" else better.upper()
        winner = {"A": f"k{a_size}", "B": f"k{b_size}", "tie": "tie"}.get(better, "unparsed")
        return {
            "model": model,
            "shown_as_a": f"k{a_size}",
            **{
                f"k{size}": grade if grade in GRADES else "unparsed"
                for size, grade in grades.items()
            },
            "better": winner,
            "note": str(parsed.get("note", ""))[:200],
        }
    return {"model": model, "unparsed": "rate_limited"}


async def _annotate(settings: Settings, path: Path, model: str) -> None:
    records = [json.loads(line) for line in _lines(path)]
    pending = [
        r for r in records if "judge" not in r and all("answer" in r[f"k{k}"] for k in SIZES)
    ]
    print(f"judge {model}: {len(pending)} questions")
    async with httpx.AsyncClient(
        base_url=settings.llm_base_url,
        headers={"Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}"},
        timeout=60,
    ) as http:
        for number, record in enumerate(pending, start=1):
            record["judge"] = await _judge_one(http, model, record)
            print(f"  [{number}/{len(pending)}] {record['id']} {record['judge'].get('better')}")
            path.write_bytes(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
            )


def _lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.1%} ({part}/{whole})" if whole else "n/a"


def summarise(records: Sequence[dict[str, Any]]) -> str:
    ok = [r for r in records if all("answer" in r[f"k{k}"] for k in SIZES)]
    lines = [f"questions: {len(records)}, answered at both sizes: {len(ok)}"]
    for kind in ("multi", "single"):
        subset = [r for r in ok if r["kind"] == kind]
        if not subset:
            continue
        lines.append(f"\n{kind}-document questions ({len(subset)}):")
        for k in SIZES:
            answers = [r[f"k{k}"] for r in subset]
            scored = [a["faithfulness"] for a in answers if not a["abstained"]]
            needed_all = sum(set(r["sources"]) <= set(r[f"k{k}"]["sources"]) for r in subset)
            line = (
                f"  top_k={k:>2}: needed documents in sources {_pct(needed_all, len(subset))}, "
                f"abstained {_pct(sum(a['abstained'] for a in answers), len(answers))}, "
                f"faithfulness {statistics.fmean(scored):.3f}, "
                f"tokens in {statistics.fmean(a['input_tokens'] or 0 for a in answers):.0f}, "
                f"latency median {statistics.median(a['seconds'] for a in answers):.2f}s"
            )
            lines.append(line)
        judged = [r for r in subset if "judge" in r and "unparsed" not in r["judge"]]
        if judged:
            for k in SIZES:
                grades = Counter(r["judge"][f"k{k}"] for r in judged)
                lines.append(
                    f"  judged top_k={k:>2}: "
                    + ", ".join(f"{g} {_pct(grades[g], len(judged))}" for g in GRADES)
                )
            better = Counter(r["judge"]["better"] for r in judged)
            lines.append(
                "  better: "
                + ", ".join(f"{w} {_pct(better[w], len(judged))}" for w in ("k5", "k10", "tie"))
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--out", type=Path, help="results file (JSON lines)")
    parser.add_argument("--limit", type=int, help="run at most this many new questions")
    parser.add_argument("--judge", help="grade the answers in --out with this model")
    parser.add_argument("--summary", type=Path, help="summarise a results file and exit")
    args = parser.parse_args(argv)

    if args.summary:
        print(summarise([json.loads(line) for line in _lines(args.summary)]))
        return 0
    if not args.out:
        parser.error("--out is required")
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    settings = get_settings()
    if settings.llm_provider != "openai_compatible":
        parser.error("LLM_PROVIDER must be openai_compatible")
    if args.judge:
        asyncio.run(_annotate(settings, args.out, args.judge))
    else:
        asyncio.run(_run(settings, args.out, args.limit))
    print()
    print(summarise([json.loads(line) for line in _lines(args.out)]))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
