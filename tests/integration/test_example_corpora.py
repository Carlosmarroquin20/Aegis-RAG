"""
Integration tests for the example corpora under examples/.

A corpus is only useful if it loads cleanly and the questions advertised next to
it actually get through. These run each corpus through the real seed path and
parsers, and each advertised question through the real gateway, so a parser
change or a new signature cannot quietly break the demo.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from aegis.application.use_cases.ingest_documents import IngestDocumentsUseCase
from aegis.domain.models.query import RawQuery
from aegis.infrastructure.parsers.parser_registry import ParserRegistry
from aegis.infrastructure.security.security_gateway import SecurityGateway
from tests.conftest import FakeVectorStore

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _ROOT / "examples"
_GUIDE = (_EXAMPLES / "README.md").read_text(encoding="utf-8")
_CORPORA = sorted(path for path in _EXAMPLES.iterdir() if path.is_dir())


def _load_seed_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "seed_documents", _ROOT / "scripts" / "seed_documents.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _questions(heading: str) -> list[str]:
    """The bullet list under a ``### heading`` of examples/README.md."""
    section = re.search(rf"^### {re.escape(heading)}\n(.*?)(?=^#|\Z)", _GUIDE, re.M | re.S)
    assert section, f"examples/README.md has no '### {heading}' section"
    return re.findall(r"^- (.+)$", section.group(1), re.M)


def test_every_corpus_is_documented() -> None:
    for corpus in _CORPORA:
        assert f"`{corpus.name}/`" in _GUIDE, f"{corpus.name} is not described in the guide"


@pytest.mark.parametrize("corpus", _CORPORA, ids=lambda path: path.name)
async def test_corpus_loads_without_failures(corpus: Path) -> None:
    store = FakeVectorStore()
    use_case = IngestDocumentsUseCase(
        vector_store=store, parser_registry=ParserRegistry(), collection="examples"
    )

    summary = await _load_seed_script().seed(corpus, use_case)

    assert summary.failures == []
    assert summary.files_indexed == len(list(corpus.glob("*.md")))
    # Every document should split into several retrievable chunks, not one blob.
    assert summary.chunks_created > summary.files_indexed


@pytest.mark.parametrize("question", _questions("Questions to try"))
def test_advertised_questions_are_let_through(question: str) -> None:
    assert not SecurityGateway(strict_mode=True).evaluate(RawQuery(text=question)).blocked


@pytest.mark.parametrize("question", _questions("Questions the gateway refuses"))
def test_advertised_attacks_are_refused(question: str) -> None:
    assert SecurityGateway(strict_mode=True).evaluate(RawQuery(text=question)).blocked
