"""
Keeps the architecture decision records consistent.

ADRs are only useful if they can be found and trusted: every record must appear
in the index, numbering must have no gaps (a missing number reads as a deleted
decision), and the names they cite — settings, metrics, alerts, tests — must
still exist in the repository.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_ADR_DIR = _ROOT / "docs" / "adr"
_RECORDS = sorted(_ADR_DIR.glob("[0-9][0-9][0-9][0-9]-*.md"))
_INDEX = (_ADR_DIR / "README.md").read_text(encoding="utf-8")


def test_records_are_numbered_without_gaps() -> None:
    numbers = [int(path.name[:4]) for path in _RECORDS]
    assert numbers == list(range(1, len(numbers) + 1))


def test_every_record_is_linked_from_the_index() -> None:
    linked = set(re.findall(r"\]\((\d{4}-[a-z0-9-]+\.md)\)", _INDEX))
    assert linked == {path.name for path in _RECORDS}


def test_every_record_has_the_required_sections() -> None:
    required = ("**Status:**", "## Context", "## Decision", "## Consequences", "## Alternatives")
    for path in _RECORDS:
        text = path.read_text(encoding="utf-8")
        missing = [section for section in required if section not in text]
        assert not missing, f"{path.name} is missing {missing}"
        assert text.startswith(f"# {path.name[:4]} — "), f"{path.name}: title must carry its number"


def test_cited_identifiers_exist_in_the_repository() -> None:
    """Settings, metrics, alerts and file names quoted in an ADR must be real."""
    searchable = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for pattern in ("src/**/*.py", "tests/**/*.py", "infra/**/*.yml", ".env.example")
        for path in _ROOT.glob(pattern)
    )
    stale: list[str] = []
    for path in _RECORDS:
        text = path.read_text(encoding="utf-8")
        cited = set(re.findall(r"`([A-Z][A-Z0-9_]{5,}(?:=[a-z_]+)?)`", text))  # SETTINGS
        cited |= set(re.findall(r"`(aegis_[a-z_]+)`", text))  # metrics
        cited |= set(re.findall(r"`(Aegis[A-Za-z]+)`", text))  # alerts
        cited |= set(re.findall(r"`([A-Z][A-Za-z]+(?:Port|Error))`", text))  # ports, errors
        for name in sorted(cited):
            if name.split("=")[0] not in searchable:
                stale.append(f"{path.name}: {name}")
        for reference in re.findall(r"`((?:tests|src|interface)/[A-Za-z0-9_./-]+\.py)`", text):
            if not ((_ROOT / reference).exists() or (_ROOT / "src" / "aegis" / reference).exists()):
                stale.append(f"{path.name}: {reference}")
    assert not stale, f"ADRs cite names that no longer exist: {stale}"
