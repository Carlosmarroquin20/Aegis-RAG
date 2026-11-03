"""
Keeps the Grafana dashboards and the alert rules tied to the code.

A panel or an alert that queries a metric nobody exports does not fail: it
shows "No data", which reads the same as a quiet system. These tests read every
query in the provisioned dashboards and alert rules and check that each Aegis
metric it names is registered, and that each span the trace dashboard filters on
is one the code actually opens.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from aegis.infrastructure.observability.metrics import registry

_ROOT = Path(__file__).resolve().parents[2]
_DASHBOARDS = sorted((_ROOT / "infra" / "grafana" / "dashboards").glob("*.json"))
_ALERTS = _ROOT / "infra" / "prometheus" / "alerts.yml"

_METRIC = re.compile(r"\baegis_[a-z_]+\b")
_SAMPLE_SUFFIX = re.compile(r"_(bucket|sum|count)$")
# Label names are not metrics: drop {matchers} and by/without (...) clauses first.
# Span attributes such as aegis.blocked become labels named aegis_blocked.
_LABEL_CONTEXT = re.compile(r"\{[^}]*\}|\b(?:by|without)\s*\([^)]*\)")
_SPAN_FILTER = re.compile(r'span_name=~?"([^"]+)"')
_SPAN_IN_CODE = re.compile(r'start_(?:as_current_)?span\(\s*"([a-z_.]+)"')


def _exported_metrics() -> set[str]:
    """
    Every series name the registry can expose. Taken from the metric types rather
    than from samples, because a labelled metric has no samples until it is first
    used, and a counter's family is named without the ``_total`` it is scraped as.
    """
    suffixes = {"counter": ("_total",), "histogram": ("_bucket", "_sum", "_count")}
    names: set[str] = set()
    for family in registry.collect():
        names.add(family.name)
        names.update(family.name + suffix for suffix in suffixes.get(family.type, ()))
    return names


def _queries(dashboard: Path) -> list[str]:
    queries: list[str] = []

    def walk(panels: list[dict[str, object]]) -> None:
        for panel in panels:
            for target in panel.get("targets", []) or []:  # type: ignore[union-attr]
                queries.append(str(target.get("expr", "")))  # type: ignore[union-attr]
            walk(panel.get("panels", []) or [])  # type: ignore[arg-type]

    walk(json.loads(dashboard.read_text(encoding="utf-8"))["panels"])
    return queries


def test_the_dashboards_are_found() -> None:
    assert len(_DASHBOARDS) >= 2


@pytest.mark.parametrize("dashboard", _DASHBOARDS, ids=lambda p: p.name)
def test_every_metric_a_dashboard_queries_is_exported(dashboard: Path) -> None:
    exported = _exported_metrics()
    used = {
        name
        for query in _queries(dashboard)
        for name in _METRIC.findall(_LABEL_CONTEXT.sub(" ", query))
    }
    missing = sorted(
        name
        for name in used
        if name not in exported and _SAMPLE_SUFFIX.sub("", name) not in exported
    )
    assert not missing, f"{dashboard.name} queries metrics no one exports: {missing}"


def test_every_metric_an_alert_uses_is_exported() -> None:
    exported = _exported_metrics()
    used = set(_METRIC.findall(_ALERTS.read_text(encoding="utf-8")))
    # Annotations may mention a metric in prose; only names that look like series count.
    missing = sorted(
        name
        for name in used
        if name not in exported and _SAMPLE_SUFFIX.sub("", name) not in exported
    )
    assert not missing, f"alerts.yml uses metrics no one exports: {missing}"


def test_every_span_the_trace_dashboard_filters_on_is_opened_by_the_code() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (_ROOT / "src" / "aegis").rglob("*.py")
    )
    opened = set(_SPAN_IN_CODE.findall(source))
    traces = _ROOT / "infra" / "grafana" / "dashboards" / "aegis-rag-traces.json"
    filtered = {
        span
        for query in _queries(traces)
        for pattern in _SPAN_FILTER.findall(query)
        for span in pattern.split("|")
    }

    assert filtered, "the trace dashboard filters on no span names; has its format changed?"
    missing = sorted(filtered - opened)
    assert not missing, f"the trace dashboard filters on spans the code never opens: {missing}"


def _panel_ids(panels: list[dict[str, object]]) -> list[int]:
    ids: list[int] = []
    for panel in panels:
        ids.append(int(panel["id"]))  # type: ignore[call-overload]
        ids.extend(_panel_ids(panel.get("panels", []) or []))  # type: ignore[arg-type]
    return ids


@pytest.mark.parametrize("dashboard", _DASHBOARDS, ids=lambda p: p.name)
def test_panel_ids_are_unique(dashboard: Path) -> None:
    ids = _panel_ids(json.loads(dashboard.read_text(encoding="utf-8"))["panels"])
    assert len(ids) == len(set(ids)), f"{dashboard.name} reuses a panel id"
