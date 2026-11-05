"""
Consistency checks for the public demo image (deploy/huggingface).

The demo is configured across files that nothing else ties together: the
public key is a build argument in the Dockerfile, its digest lives in the
tenants file, and the read-only and throttling settings are image ENV lines.
A mismatch would only show once the Space is built and every query gets a 403,
so the pieces are checked against each other here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aegis.config import Settings
from aegis.infrastructure.tenancy.registry import TenantRegistry

_DEPLOY = Path(__file__).resolve().parents[2] / "deploy" / "huggingface"
_DOCKERFILE = (_DEPLOY / "Dockerfile").read_text(encoding="utf-8")


def _arg_default(name: str) -> str:
    match = re.search(rf"^ARG {name}=(\S+)$", _DOCKERFILE, re.MULTILINE)
    assert match, f"ARG {name} has no default in the demo Dockerfile"
    return match.group(1)


def _env(name: str) -> str:
    match = re.search(rf'^\s+{name}="([^"]*)"', _DOCKERFILE, re.MULTILINE)
    assert match, f"{name} is not set in the demo image"
    return match.group(1)


@pytest.fixture(scope="module")
def registry() -> TenantRegistry:
    return TenantRegistry.from_file(_DEPLOY / "tenants.toml")


class TestDemoTenant:
    def test_the_key_baked_into_the_console_authenticates(self, registry: TenantRegistry) -> None:
        tenant = registry.resolve(_arg_default("DEMO_API_KEY"))
        assert tenant is not None
        assert tenant.id == "demo"

    def test_the_public_key_is_limited(self, registry: TenantRegistry) -> None:
        """Everyone shares the key, so its limits are what bounds the demo's cost."""
        (tenant,) = registry.tenants
        assert tenant.rate_limit_requests is not None
        assert tenant.daily_token_budget is not None


class TestDemoImage:
    def test_the_corpus_is_read_only(self) -> None:
        assert _env("DOCUMENT_WRITES_ENABLED") == "false"

    def test_chroma_runs_in_process(self) -> None:
        assert _env("CHROMA_MODE") == "embedded"
        assert _env("CHROMA_PATH")

    def test_the_seed_step_indexes_the_tenant_the_api_serves(self) -> None:
        assert "--tenant demo" in _DOCKERFILE
        assert _env("TENANTS_FILE") == "/app/deploy/tenants.toml"

    def test_the_baked_embedding_model_is_the_default(self) -> None:
        """
        The demo pins the model it bakes and indexes with. When the default
        changes, the demo must follow, or its corpus is embedded with a model
        the rest of the project no longer evaluates.
        """
        assert _arg_default("EMBEDDING_MODEL") == Settings.model_fields["embedding_model"].default

    def test_no_llm_key_is_baked_in(self) -> None:
        """The provider key is a platform secret; the API must refuse to start without it."""
        assert "LLM_API_KEY=" not in _DOCKERFILE.replace("-e LLM_API_KEY=...", "")
