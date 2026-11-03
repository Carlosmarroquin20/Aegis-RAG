"""
Compares the SecurityGateway with an ML prompt-injection classifier on the same corpus.

Usage:
    uv run python scripts/compare_injection_classifier.py
    uv run python scripts/compare_injection_classifier.py --model deepset/deberta-v3-base-injection
    uv run python scripts/compare_injection_classifier.py --threshold 0.9 --show-misses

The gateway's own report (docs/GATEWAY_EVALUATION.md) is generated without any
model and checked by a unit test. This script answers the next question: what
would a classifier add? It runs the 326 labelled queries through the gateway (strict
mode, as shipped), through the classifier, and through both combined (blocked
if either blocks), and prints the same figures for each: attacks blocked in the
categories the gateway targets, in its documented gaps (paraphrase, other
languages, character substitution) and overall, false positives, and precision.

It needs the classifier's weights (downloaded on first use), so it runs in CI
as a measurement rather than inside the unit tests.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_gateway import (  # noqa: E402  (sibling script, not a package)
    ATTACK,
    BENIGN,
    DEFAULT_DATASET,
    DOCUMENTED_GAPS,
    Rate,
    Sample,
    evaluate,
    load_dataset,
)

from aegis.infrastructure.security.injection_classifier import (  # noqa: E402
    DEFAULT_INJECTION_CLASSIFIER,
    TransformersInjectionClassifier,
)

DEFAULT_THRESHOLD = 0.5


def _rate(flags: Sequence[bool]) -> Rate:
    return Rate(sum(flags), len(flags))


def summary(samples: Sequence[Sample], blocked: Sequence[bool]) -> dict[str, Rate]:
    """The headline figures for one detector's verdicts, in the gateway report's terms."""
    pairs = list(zip(samples, blocked, strict=True))
    attacks = [b for s, b in pairs if s.label == ATTACK]
    targeted = [b for s, b in pairs if s.label == ATTACK and s.category not in DOCUMENTED_GAPS]
    gaps = [b for s, b in pairs if s.label == ATTACK and s.category in DOCUMENTED_GAPS]
    benign = [b for s, b in pairs if s.label == BENIGN]
    flagged = [s.label == ATTACK for s, b in pairs if b]
    return {
        "targeted": _rate(targeted),
        "gaps": _rate(gaps),
        "attacks": _rate(attacks),
        "false_positives": _rate(benign),
        "precision": _rate(flagged),
    }


def by_category(samples: Sequence[Sample], blocked: Sequence[bool]) -> dict[str, Rate]:
    grouped: dict[str, list[bool]] = {}
    for sample, flag in zip(samples, blocked, strict=True):
        grouped.setdefault(f"{sample.label}/{sample.category}", []).append(flag)
    return {key: _rate(flags) for key, flags in sorted(grouped.items())}


_ROWS = (
    ("targeted", "Attacks blocked — categories the gateway targets"),
    ("gaps", "Attacks blocked — gateway's documented gaps"),
    ("attacks", "Attacks blocked — whole corpus"),
    ("false_positives", "Legitimate queries blocked (false positives)"),
    ("precision", "Precision of a block"),
)


def _print_table(columns: dict[str, dict[str, Rate]]) -> None:
    names = list(columns)
    print("\n| Metric | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for key, label in _ROWS:
        cells = " | ".join(str(columns[name][key]) for name in names)
        print(f"| {label} | {cells} |")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--model", default=DEFAULT_INJECTION_CLASSIFIER)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--show-misses", action="store_true", help="list the classifier's errors")
    args = parser.parse_args(argv)

    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    samples = load_dataset(args.dataset)
    gateway = [v.blocked for v in evaluate(samples, strict_mode=True)]

    classifier = TransformersInjectionClassifier(args.model)
    started = time.perf_counter()
    classifier.load()
    loaded = time.perf_counter()
    scores = classifier.scores([s.text for s in samples])
    per_query_ms = (time.perf_counter() - loaded) * 1000 / len(samples)
    model_flags = [score >= args.threshold for score in scores]
    combined = [g or m for g, m in zip(gateway, model_flags, strict=True)]

    print(
        f"{len(samples)} queries; classifier {args.model} at threshold {args.threshold}: "
        f"loaded in {loaded - started:.1f} s, {per_query_ms:.0f} ms/query on this machine"
    )
    _print_table(
        {
            "Gateway (strict)": summary(samples, gateway),
            "Classifier": summary(samples, model_flags),
            "Either": summary(samples, combined),
        }
    )

    print("\nBy category (gateway / classifier / either):")
    per_gateway = by_category(samples, gateway)
    per_model = by_category(samples, model_flags)
    per_either = by_category(samples, combined)
    for key in per_gateway:
        gap = " †" if key.split("/", 1)[1] in DOCUMENTED_GAPS and key.startswith(ATTACK) else ""
        print(
            f"  {key + gap:42s} {per_gateway[key].hits:3d} / {per_model[key].hits:3d} / "
            f"{per_either[key].hits:3d}  of {per_gateway[key].total}"
        )

    if args.show_misses:
        print("\nClassifier errors:")
        for sample, score, flag in zip(samples, scores, model_flags, strict=True):
            wrong = flag != (sample.label == ATTACK)
            if wrong:
                kind = "missed" if sample.label == ATTACK else "false positive"
                print(
                    f"  {kind:14s} {score:.2f}  {sample.id} [{sample.category}] {sample.text[:90]}"
                )
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
