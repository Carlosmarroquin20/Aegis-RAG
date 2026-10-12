"""
Keeps docs/THREAT_MODEL.md tied to the code.

The threat model claims that every control it lists is backed by a named test
and lives in a named file. That claim is only worth making if something fails
when it stops being true — a renamed test or a moved module would otherwise
leave the document confidently pointing at nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_DOC = (_ROOT / "docs" / "THREAT_MODEL.md").read_text(encoding="utf-8")

# Directories a short module reference in the document may be relative to.
_PACKAGE = _ROOT / "src" / "aegis"
_SOURCE_ROOTS = [_ROOT, _PACKAGE, *(p for p in _PACKAGE.rglob("*") if p.is_dir())]


def _cited(pattern: str) -> list[str]:
    return sorted(set(re.findall(pattern, _DOC)))


def test_every_cited_test_exists() -> None:
    test_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (_ROOT / "tests").rglob("*.py")
    )
    cited = _cited(r"`((?:Test[A-Za-z]+|test_[a-z0-9_]+))`")

    missing = [
        name
        for name in cited
        if not re.search(
            rf"class {name}\b" if name.startswith("Test") else rf"def {name}\b", test_sources
        )
    ]
    assert cited, "the document cites no tests; the extraction pattern is probably stale"
    assert not missing, f"THREAT_MODEL.md cites tests that do not exist: {missing}"


def test_every_cited_file_exists() -> None:
    cited = _cited(r"`([A-Za-z0-9_./-]+\.(?:py|yml|yaml|md|lock))`")

    missing = [
        reference
        for reference in cited
        if not any((root / reference).exists() for root in _SOURCE_ROOTS)
    ]
    assert cited, "the document cites no files; the extraction pattern is probably stale"
    assert not missing, f"THREAT_MODEL.md cites files that do not exist: {missing}"


def test_threat_identifiers_are_unique_and_sequential() -> None:
    """Identifiers are referenced from commits and issues, so they must stay stable."""
    identifiers = re.findall(r"^\| (T-\d{2}) \|", _DOC, flags=re.MULTILINE)
    assert identifiers == [f"T-{n:02d}" for n in range(1, len(identifiers) + 1)]
