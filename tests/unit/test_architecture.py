"""
Architecture test: the dependency rule of the hexagonal layout.

    interface  →  application  →  domain  ←  infrastructure

Each layer may import only from the layers it points to. The rule is easy to
state and easy to erode one convenient import at a time, so it is enforced here
by parsing every module's imports rather than left to code review.

Imports are read from the syntax tree (not by importing the modules), so the
check also sees imports that are deferred inside functions or guarded by
``TYPE_CHECKING`` — the usual places a layering violation hides.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "aegis"

# layer → layers it must never import from.
_FORBIDDEN: dict[str, frozenset[str]] = {
    "domain": frozenset({"application", "infrastructure", "interface"}),
    "application": frozenset({"infrastructure", "interface"}),
    "infrastructure": frozenset({"application", "interface"}),
}


def _imported_aegis_layers(module: Path) -> set[tuple[str, int]]:
    """Returns (layer, line) for every ``aegis.<layer>`` import in the module."""
    found: set[tuple[str, int]] = set()
    for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "aegis" and len(parts) > 1:
                found.add((parts[1], node.lineno))
    return found


@pytest.mark.parametrize("layer", sorted(_FORBIDDEN))
def test_layer_only_depends_inward(layer: str) -> None:
    violations = [
        f"{module.relative_to(_PACKAGE_ROOT.parent)}:{line} imports aegis.{imported}"
        for module in sorted((_PACKAGE_ROOT / layer).rglob("*.py"))
        for imported, line in sorted(_imported_aegis_layers(module))
        if imported in _FORBIDDEN[layer]
    ]
    assert not violations, "Dependency rule violated:\n  " + "\n  ".join(violations)


def test_the_check_actually_inspects_modules() -> None:
    """Guards against the rule passing vacuously if the source tree moves."""
    for layer in _FORBIDDEN:
        assert any((_PACKAGE_ROOT / layer).rglob("*.py")), f"no modules found for {layer}"

    use_case = _PACKAGE_ROOT / "application" / "use_cases" / "query_rag.py"
    assert "domain" in {layer for layer, _ in _imported_aegis_layers(use_case)}
