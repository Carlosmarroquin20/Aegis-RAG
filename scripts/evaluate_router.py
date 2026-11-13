"""
Measures the model router's complexity classifier against labelled questions.

Usage:
    uv run python scripts/evaluate_router.py            # print the summary
    uv run python scripts/evaluate_router.py --write    # regenerate the report
    uv run python scripts/evaluate_router.py --check    # fail if the report is stale

The router sends a question to the small model only when the classifier calls
it simple (ADR 0026). Two numbers say whether that is safe and whether it is
worth it: how many complex questions stay on the large model (the quality
guard), and how many simple ones move to the small model (the saving). This
script runs every question in ``evals/routing/dataset.jsonl`` through
``assess_complexity`` exactly as the router does, with retrieved context
present, and reports both with the individual misses.

Labelling rubric, fixed before the first run: a question is **complex** if it
asks more than one distinct thing, compares options, needs a figure computed
from several others, asks why or for an explanation, asks for a summary or a
walk-through, or needs two separate policies combined to answer. Anything else
is **simple**, including a lookup that applies one rule to one figure ("Can I
split a Q1,500 purchase into instalments?").

The report (``docs/ROUTER_EVALUATION.md``) is a pure function of the dataset and
the classifier, and a unit test fails when it is stale.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from aegis.domain.services.query_complexity import ComplexityAssessment, Tier, assess_complexity

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evals" / "routing" / "dataset.jsonl"
REPORT_PATH = ROOT / "docs" / "ROUTER_EVALUATION.md"

# The router only consults the classifier with passages in hand; with none it
# routes to the small model whatever the question says.
_CONTEXT_SIZE = 5

# Results of the first run that were bugs in the rules rather than limits of
# the approach. The baseline figures are in the history of the report.
_FIXED_DEFECTS: tuple[str, ...] = (
    'The comparison rule matched `differen`, which misses the verb itself: "How do store '
    'hours differ between…" went to the small model. It now matches `differ`.',
    'The comparison rule matched "cuál es mejor" but not "qué es mejor", the more common '
    "Spanish way to ask which of two options is better. Both are matched now.",
)


@dataclass(frozen=True, slots=True)
class Sample:
    id: str
    lang: str
    origin: str
    kind: str
    label: Tier
    question: str


@dataclass(frozen=True, slots=True)
class Verdict:
    sample: Sample
    assessment: ComplexityAssessment

    @property
    def correct(self) -> bool:
        return self.assessment.tier is self.sample.label


@dataclass(frozen=True, slots=True)
class Rate:
    hits: int
    total: int

    @property
    def value(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def interval(self) -> tuple[float, float]:
        """95% Wilson score interval; sound at rates of 0 or 1 and small counts."""
        if not self.total:
            return (0.0, 0.0)
        z = 1.959964
        n, p = self.total, self.value
        denominator = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denominator
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        return (max(0.0, centre - margin), min(1.0, centre + margin))

    def __str__(self) -> str:
        low, high = self.interval()
        return f"{self.value:.1%} ({self.hits}/{self.total}, 95% CI {low:.0%}–{high:.0%})"


class DatasetError(ValueError):
    """The dataset is malformed; raised with the offending line number."""


def load_dataset(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            sample = Sample(
                id=record["id"],
                lang=record["lang"],
                origin=record["origin"],
                kind=record["kind"],
                label=Tier(record["label"]),
                question=record["question"],
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise DatasetError(f"{path.name}:{number}: {exc}") from exc
        if sample.id in seen:
            raise DatasetError(f"{path.name}:{number}: duplicate id {sample.id!r}")
        seen.add(sample.id)
        samples.append(sample)
    return samples


def evaluate(samples: Iterable[Sample]) -> list[Verdict]:
    return [Verdict(s, assess_complexity(s.question, context_size=_CONTEXT_SIZE)) for s in samples]


def _routed(verdicts: Iterable[Verdict], label: Tier, tier: Tier) -> Rate:
    of_label = [v for v in verdicts if v.sample.label is label]
    return Rate(sum(v.assessment.tier is tier for v in of_label), len(of_label))


def complex_kept_large(verdicts: Sequence[Verdict]) -> Rate:
    return _routed(verdicts, Tier.COMPLEX, Tier.COMPLEX)


def simple_moved_small(verdicts: Sequence[Verdict]) -> Rate:
    return _routed(verdicts, Tier.SIMPLE, Tier.SIMPLE)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _breakdown(verdicts: Sequence[Verdict], label: Tier, key: str) -> list[str]:
    groups: dict[str, list[Verdict]] = {}
    for v in verdicts:
        if v.sample.label is label:
            groups.setdefault(getattr(v.sample, key), []).append(v)
    return [
        f"| {name} | {Rate(sum(v.correct for v in group), len(group))} |"
        for name, group in sorted(groups.items())
    ]


def render_report(samples: Sequence[Sample]) -> str:
    verdicts = evaluate(samples)
    kept = complex_kept_large(verdicts)
    moved = simple_moved_small(verdicts)
    to_small = [v for v in verdicts if v.assessment.tier is Tier.SIMPLE]
    complex_missed = [v for v in verdicts if v.sample.label is Tier.COMPLEX and not v.correct]
    simple_kept = [v for v in verdicts if v.sample.label is Tier.SIMPLE and not v.correct]
    retrieval_set = [v for v in verdicts if v.sample.origin == "retrieval_set"]

    out = [
        "# Model router evaluation",
        "",
        "<!-- Generated by scripts/evaluate_router.py --write. Do not edit by hand:",
        "     tests/unit/test_router_evaluation.py fails if this file is stale. -->",
        "",
        "How well the router's complexity classifier keeps hard questions on the large model",
        "while moving easy ones to the small one ([ADR 0026](adr/0026-route-simple-questions-"
        "to-a-small-model.md)). The questions live in",
        "[`evals/routing/dataset.jsonl`](../evals/routing/dataset.jsonl); every number below is",
        "reproduced by `uv run python scripts/evaluate_router.py`.",
        "",
        f"**Dataset:** {len(samples)} questions, "
        f"{sum(s.label is Tier.SIMPLE for s in samples)} simple and "
        f"{sum(s.label is Tier.COMPLEX for s in samples)} complex. "
        f"{len(retrieval_set)} are the retrieval evaluation's questions, written before the "
        "router existed; the rest were written for this evaluation.",
        "",
        "## Headline",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Complex questions kept on the large model (quality guard) | {kept} |",
        f"| Simple questions moved to the small model (saving) | {moved} |",
        f"| Retrieval-set questions sent to the small model | "
        f"{Rate(sum(v.assessment.tier is Tier.SIMPLE for v in retrieval_set), len(retrieval_set))}"
        " |",
        f"| All questions sent to the small model | {Rate(len(to_small), len(verdicts))} |",
        "",
        "The first row is the one that protects answer quality: a complex question routed to",
        "the small model is the expensive mistake. The retrieval-set row is the best guide to",
        "real traffic, which is mostly single lookups.",
        "",
        "## Complex questions, by kind",
        "",
        "| Kind | Kept on the large model |",
        "|---|---|",
        *_breakdown(verdicts, Tier.COMPLEX, "kind"),
        "",
        "## By language",
        "",
        "| Language | Complex kept large |",
        "|---|---|",
        *_breakdown(verdicts, Tier.COMPLEX, "lang"),
        "",
        "| Language | Simple moved small |",
        "|---|---|",
        *_breakdown(verdicts, Tier.SIMPLE, "lang"),
        "",
        f"## Complex questions sent to the small model ({len(complex_missed)})",
        "",
        "| ID | Kind | Question |",
        "|---|---|---|",
        *(
            f"| {v.sample.id} | {v.sample.kind} | {_cell(v.sample.question)} |"
            for v in complex_missed
        ),
        "",
        f"## Simple questions kept on the large model ({len(simple_kept)})",
        "",
        "| ID | Signals | Question |",
        "|---|---|---|",
        *(
            f"| {v.sample.id} | {', '.join(v.assessment.signals)} | {_cell(v.sample.question)} |"
            for v in simple_kept
        ),
        "",
        "## Defects this evaluation found",
        "",
        "Most misses above are the limit of reading a question's wording. These were not; they",
        "were bugs in the rules, and they were fixed:",
        "",
        *(f"- {finding}" for finding in _FIXED_DEFECTS),
        "",
        "## Method and limits",
        "",
        "- Each question is classified with retrieved context present, as the router sees it.",
        "  With no passages the router uses the small model regardless, since the only correct",
        "  answer is an abstention.",
        "- **This measures the routing decision, not the answers.** Whether the small model",
        "  answers the questions it receives as well as the large one is a separate, live",
        "  measurement against the configured models: see",
        "  [`ROUTER_LIVE_EVALUATION.md`](ROUTER_LIVE_EVALUATION.md).",
        "- **Questions that combine two policies without saying so are the weak spot.** \"I'm a",
        '  Gold member in Quetzaltenango. Do I pay for installation?" has no comparison, no',
        "  calculation and one question mark; only knowing the documents shows it needs two",
        "  of them. Reading the retrieved passages (several documents with close scores) is",
        "  the candidate signal for these.",
        "- The rules are not tuned against this file. Apart from the defects above, a miss is",
        "  a finding to weigh: a rule per missed question would raise the figure here without",
        "  routing unseen questions any better.",
        "- **The questions written for this evaluation were written by the author of the",
        "  rules,** after the rubric was fixed but knowing the rules. Read their figures as an",
        "  upper bound; the retrieval-set questions were not written with routing in mind.",
        "- Intervals are 95% Wilson score intervals; per-kind counts are small.",
        "",
    ]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help=f"regenerate {REPORT_PATH.name}")
    group.add_argument("--check", action="store_true", help="fail if the report is stale")
    args = parser.parse_args(argv)

    samples = load_dataset(args.dataset)
    report = render_report(samples)
    if args.write:
        REPORT_PATH.write_bytes(report.encode("utf-8"))
        print(f"wrote {REPORT_PATH.relative_to(ROOT)}")
        return 0
    if args.check:
        current = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
        if current != report:
            print(f"{REPORT_PATH.name} is stale; run with --write", file=sys.stderr)
            return 1
        print(f"{REPORT_PATH.name} is up to date")
        return 0

    verdicts = evaluate(samples)
    print(f"complex kept on the large model: {complex_kept_large(verdicts)}")
    print(f"simple moved to the small model: {simple_moved_small(verdicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
