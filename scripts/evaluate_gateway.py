"""
Measures the SecurityGateway against a labelled corpus of queries.

Usage:
    uv run python scripts/evaluate_gateway.py            # print the summary
    uv run python scripts/evaluate_gateway.py --write    # regenerate the report
    uv run python scripts/evaluate_gateway.py --check    # fail if the report is stale
    uv run python scripts/evaluate_gateway.py --dataset other.jsonl

The gateway is a heuristic, so "it blocks prompt injection" is a claim that
needs a number next to it, and the number needs to include the cases it gets
wrong. This script runs every sample in ``evals/gateway/dataset.jsonl`` through
the real gateway, in both of its modes, and reports detection and false-positive
rates per category together with the individual misses.

The report it writes (``docs/GATEWAY_EVALUATION.md``) is a pure function of the
dataset and the gateway. A unit test regenerates it and compares, so a change to
a signature that moves a number cannot land without the document moving with it.
Timing is printed but never written: it varies per machine and would make the
report non-reproducible.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import re
import statistics
import sys
import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import structlog

from aegis.domain.models.query import RawQuery
from aegis.infrastructure.security.security_gateway import SecurityGateway

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evals" / "gateway" / "dataset.jsonl"
REPORT_PATH = ROOT / "docs" / "GATEWAY_EVALUATION.md"

ATTACK = "attack"
BENIGN = "benign"

# Attack categories the gateway does not claim to stop. The threat model lists
# them as residual risk; they are in the corpus so that the overall detection
# rate reflects what an attacker can actually send, not only what the signatures
# were written for.
DOCUMENTED_GAPS: frozenset[str] = frozenset({"paraphrase", "non_english", "character_substitution"})

_PREVIEW_CHARS = 90


@dataclass(frozen=True, slots=True)
class Sample:
    id: str
    label: str
    category: str
    text: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class Verdict:
    sample: Sample
    blocked: bool
    score: int
    rules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Rate:
    """A proportion with the count behind it, so small samples stay visible."""

    hits: int
    total: int

    @property
    def value(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def interval(self) -> tuple[float, float]:
        """
        95% Wilson score interval.

        Preferred over the normal approximation because several categories have
        fewer than twenty samples and rates of exactly 0 or 1, where the normal
        interval collapses to zero width and overstates the certainty.
        """
        if not self.total:
            return (0.0, 0.0)
        z = 1.959964
        n = self.total
        p = self.value
        denominator = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denominator
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        return (max(0.0, centre - margin), min(1.0, centre + margin))

    def __str__(self) -> str:
        low, high = self.interval()
        return f"{self.value:.1%} ({self.hits}/{self.total}, 95% CI {low:.0%}–{high:.0%})"


class DatasetError(ValueError):
    """The corpus is malformed; raised with the offending line number."""


def load_dataset(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            sample = Sample(
                id=record["id"],
                label=record["label"],
                category=record["category"],
                text=record["text"],
                note=record.get("note", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise DatasetError(f"{path.name}:{number}: {exc!r}") from exc

        if sample.label not in (ATTACK, BENIGN):
            raise DatasetError(f"{path.name}:{number}: unknown label {sample.label!r}")
        if sample.id in seen_ids:
            raise DatasetError(f"{path.name}:{number}: duplicate id {sample.id!r}")
        # A repeated text would be counted twice and quietly weight the result.
        if sample.text in seen_texts:
            raise DatasetError(f"{path.name}:{number}: duplicate text ({sample.id})")
        seen_ids.add(sample.id)
        seen_texts.add(sample.text)
        samples.append(sample)

    if not samples:
        raise DatasetError(f"{path.name}: no samples")
    return samples


def evaluate(samples: Iterable[Sample], *, strict_mode: bool) -> list[Verdict]:
    gateway = SecurityGateway(strict_mode=strict_mode)
    verdicts: list[Verdict] = []
    for sample in samples:
        result = gateway.evaluate(RawQuery(text=sample.text))
        verdicts.append(
            Verdict(
                sample=sample,
                blocked=result.blocked,
                score=result.assessment.score,
                rules=tuple(result.assessment.triggered_rules),
            )
        )
    return verdicts


def blocked_rate(verdicts: Iterable[Verdict]) -> Rate:
    """Share of the given verdicts that were blocked: detection for attacks, FPR for benign."""
    items = list(verdicts)
    return Rate(sum(v.blocked for v in items), len(items))


def _attacks(verdicts: Sequence[Verdict]) -> list[Verdict]:
    return [v for v in verdicts if v.sample.label == ATTACK]


def _benign(verdicts: Sequence[Verdict]) -> list[Verdict]:
    return [v for v in verdicts if v.sample.label == BENIGN]


def precision(verdicts: Sequence[Verdict]) -> Rate:
    """Of everything blocked, the share that was really an attack."""
    blocked = [v for v in verdicts if v.blocked]
    return Rate(sum(v.sample.label == ATTACK for v in blocked), len(blocked))


def _by_category(verdicts: Sequence[Verdict]) -> dict[str, list[Verdict]]:
    grouped: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        grouped.setdefault(verdict.sample.category, []).append(verdict)
    return grouped


# ── Markdown rendering ────────────────────────────────────────────────────────

_MARKDOWN_SPECIALS = re.compile(r"([\\`*_\[\]|~])")


def _cell(text: str) -> str:
    """
    Makes arbitrary sample text safe to place in a Markdown table cell.

    The corpus is hostile by construction: it contains HTML, pipes, backticks
    and invisible characters. Each of those would either break the table or be
    interpreted by the renderer, so the cell shows the text, never executes it.
    """
    flat = " ".join(text.split())
    if len(flat) > _PREVIEW_CHARS:
        flat = flat[:_PREVIEW_CHARS].rstrip() + "…"
    visible = "".join(
        f"\\u{ord(ch):04x}" if unicodedata.category(ch) in ("Cc", "Cf") else ch for ch in flat
    )
    return _MARKDOWN_SPECIALS.sub(r"\\\1", html.escape(visible, quote=False))


def _label(category: str) -> str:
    return category.replace("_", " ")


def _summary_rows(verdicts: Sequence[Verdict]) -> list[tuple[str, Rate]]:
    attacks = _attacks(verdicts)
    targeted = [v for v in attacks if v.sample.category not in DOCUMENTED_GAPS]
    gaps = [v for v in attacks if v.sample.category in DOCUMENTED_GAPS]
    return [
        ("Attacks blocked — categories the gateway targets", blocked_rate(targeted)),
        ("Attacks blocked — documented gaps", blocked_rate(gaps)),
        ("Attacks blocked — whole corpus", blocked_rate(attacks)),
        ("Legitimate queries blocked (false positives)", blocked_rate(_benign(verdicts))),
        ("Precision of a block", precision(verdicts)),
    ]


def _category_table(verdicts: Sequence[Verdict], header: str) -> list[str]:
    lines = [f"| Category | Samples | {header} |", "|---|---:|---|"]
    for category, items in _by_category(verdicts).items():
        gap = " †" if category in DOCUMENTED_GAPS and items[0].sample.label == ATTACK else ""
        lines.append(f"| {_label(category)}{gap} | {len(items)} | {blocked_rate(items)} |")
    return lines


def render_report(samples: Sequence[Sample]) -> str:
    strict = evaluate(samples, strict_mode=True)
    permissive = evaluate(samples, strict_mode=False)

    attacks = _attacks(strict)
    benign = _benign(strict)
    missed = [v for v in attacks if not v.blocked]
    false_positives = [v for v in benign if v.blocked]

    out: list[str] = [
        "# Input gateway evaluation",
        "",
        "<!-- Generated by scripts/evaluate_gateway.py --write. Do not edit by hand:",
        "     tests/unit/test_gateway_evaluation.py fails if this file is stale. -->",
        "",
        "How well the `SecurityGateway` separates prompt-injection attempts from legitimate",
        "questions, measured on a labelled corpus rather than asserted. The corpus lives in",
        "[`evals/gateway/dataset.jsonl`](../evals/gateway/dataset.jsonl); every number below is",
        "reproduced by `uv run python scripts/evaluate_gateway.py`.",
        "",
        f"**Corpus:** {len(samples)} queries — {len(attacks)} attacks in "
        f"{len(_by_category(attacks))} categories and {len(benign)} legitimate queries in "
        f"{len(_by_category(benign))} categories.",
        "",
        "## Headline",
        "",
        "Strict mode is the shipped default (`SECURITY_STRICT_MODE=true`): it blocks at a",
        "score of 60. Permissive mode blocks at 80.",
        "",
        "| Metric | Strict (default) | Permissive |",
        "|---|---|---|",
    ]
    for (name, strict_rate), (_, permissive_rate) in zip(
        _summary_rows(strict), _summary_rows(permissive), strict=True
    ):
        out.append(f"| {name} | {strict_rate} | {permissive_rate} |")

    out += [
        "",
        "Read the first three rows together. The gateway is a signature engine: it is",
        "effective against the phrasings and obfuscations it was built for, and close to blind",
        "to an attacker who rewords the request or writes it in another language. The whole-corpus",
        "figure is the honest one to quote, and it is the reason the gateway is one layer of",
        "several — see [the threat model](THREAT_MODEL.md#6-residual-risks-and-non-goals).",
        "",
        "## Attacks, by category (strict mode)",
        "",
        *_category_table(attacks, "Blocked"),
        "",
        "† Documented gap: the gateway does not claim to stop this class.",
        "",
        "## Legitimate queries, by category (strict mode)",
        "",
        *_category_table(benign, "Wrongly blocked"),
        "",
        "`hard negative` queries use the vocabulary of an attack for an ordinary purpose",
        '("how do I override the default timeout?"). `technical` queries contain code, markup',
        "and template syntax. Both are deliberately adversarial to the detector; the",
        "`knowledge base` category is what everyday traffic looks like.",
        "",
        f"## Attacks that got through ({len(missed)})",
        "",
        "| ID | Category | Score | Query | Technique |",
        "|---|---|---:|---|---|",
    ]
    out += [
        f"| {v.sample.id} | {_label(v.sample.category)} | {v.score} | {_cell(v.sample.text)} "
        f"| {_cell(v.sample.note)} |"
        for v in missed
    ]

    out += [
        "",
        f"## Legitimate queries that were blocked ({len(false_positives)})",
        "",
        "| ID | Category | Score | Rule | Query |",
        "|---|---|---:|---|---|",
    ]
    out += [
        f"| {v.sample.id} | {_label(v.sample.category)} | {v.score} "
        f"| {', '.join(_label(rule) for rule in v.rules)} | {_cell(v.sample.text)} |"
        for v in false_positives
    ]

    out += [
        "",
        "## Method and limits",
        "",
        "- Each query is passed to `SecurityGateway.evaluate()` exactly as the query route",
        "  does. An attack counts as detected only if the gateway **blocks** it; a query that",
        "  is let through with its markup stripped counts as a miss.",
        "- Intervals are 95% Wilson score intervals. Several categories have fewer than",
        "  twenty samples, so the intervals are wide and per-category figures are indicative.",
        "- **The corpus was written by the author of the gateway.** It mixes published",
        "  injection strings with original ones, and it was fixed before the first run, but it",
        "  is not an independent benchmark and should not be read as one. Use `--dataset` to",
        "  run the same measurement over a third-party corpus.",
        "- The gateway is not tuned against this file. A miss listed above is a finding to",
        "  weigh, not a test to make pass: adding a signature per missed string would raise",
        "  the number here without making the system harder to attack.",
        "- This measures the input gateway alone. A query that passes it still meets the",
        "  hardened system prompt, retrieval scoped to the indexed corpus and the output",
        "  sanitizer's reflection check; none of those are credited here.",
        "",
    ]
    return "\n".join(out)


# ── Command line ──────────────────────────────────────────────────────────────


def _silence_gateway_logs() -> None:
    # The gateway logs one line per query; a few hundred of them would bury the result.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))


def _latency_ms(samples: Sequence[Sample], *, rounds: int = 5) -> list[float]:
    gateway = SecurityGateway(strict_mode=True)
    timings: list[float] = []
    for sample in samples:
        query = RawQuery(text=sample.text)
        best = math.inf
        for _ in range(rounds):
            start = time.perf_counter()
            gateway.evaluate(query)
            best = min(best, time.perf_counter() - start)
        timings.append(best * 1000)
    return timings


def _print_summary(samples: Sequence[Sample]) -> None:
    for mode, strict_mode in (("strict", True), ("permissive", False)):
        verdicts = evaluate(samples, strict_mode=strict_mode)
        print(f"\n{mode} mode")
        for name, rate in _summary_rows(verdicts):
            print(f"  {name:<52} {rate}")

    timings = sorted(_latency_ms(samples))
    p95 = timings[min(len(timings) - 1, math.ceil(0.95 * len(timings)) - 1)]
    print(
        f"\nlatency per query (this machine): median {statistics.median(timings):.3f} ms, "
        f"p95 {p95:.3f} ms, max {timings[-1]:.3f} ms"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help=f"regenerate {REPORT_PATH.name}")
    group.add_argument("--check", action="store_true", help="exit 1 if the report is stale")
    args = parser.parse_args(argv)

    _silence_gateway_logs()
    try:
        samples = load_dataset(args.dataset)
    except (OSError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.write or args.check:
        report = render_report(samples)
        if args.write:
            REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")
            print(f"wrote {REPORT_PATH.relative_to(ROOT)}")
            return 0
        current = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
        if current != report:
            print(f"{REPORT_PATH.relative_to(ROOT)} is stale; run with --write", file=sys.stderr)
            return 1
        print("report is up to date")
        return 0

    _print_summary(samples)
    return 0


if __name__ == "__main__":
    # Windows consoles default to a legacy code page that cannot encode the
    # interval dash; the report itself is always written as UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
