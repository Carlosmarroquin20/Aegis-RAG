"""
Fixtures for end-to-end tests against the running docker-compose stack.

E2E tests assume the stack is reachable on the host network:

    docker compose up -d
    pytest -m e2e --no-cov

Both the base URL and the API key can be overridden via environment so the
same test suite can run against a remote staging environment if desired:

    AEGIS_E2E_BASE_URL=https://staging.example.com \\
    AEGIS_E2E_API_KEY=secret-key-here \\
    pytest -m e2e --no-cov

A module-level autouse fixture probes ``/health`` once per session and
skips every e2e test cleanly if the stack is unreachable, so a forgotten
``docker compose up`` produces clear output instead of cryptic timeouts.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import httpx
import pytest

# ── Configuration via environment ─────────────────────────────────────────────

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_API_KEY = "dev-key-change-in-production"  # matches docker-compose.yml
_HEALTH_TIMEOUT_SECONDS = 30
_HEALTH_POLL_INTERVAL_SECONDS = 2


# ── Session-scope fixtures ────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def e2e_base_url() -> str:
    return os.environ.get("AEGIS_E2E_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


@pytest.fixture(scope="session")
def e2e_api_key() -> str:
    return os.environ.get("AEGIS_E2E_API_KEY", _DEFAULT_API_KEY)


@pytest.fixture(scope="session")
def e2e_auth_headers(e2e_api_key: str) -> dict[str, str]:
    return {"X-API-Key": e2e_api_key}


@pytest.fixture(scope="session")
def e2e_client(e2e_base_url: str) -> Iterator[httpx.Client]:
    """Reusable httpx Client with a generous timeout for LLM-bound calls."""
    with httpx.Client(base_url=e2e_base_url, timeout=httpx.Timeout(60.0)) as client:
        yield client


# ── Stack-readiness gate ──────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def _require_stack_up(e2e_client: httpx.Client, request: pytest.FixtureRequest) -> None:
    """
    Probes /health until it answers 200 or the timeout expires. If the stack
    is not reachable, every e2e test is skipped with a clear message.

    This fixture is autouse so individual tests do not need to opt in.
    """
    # Only enforce when we are actually running e2e tests; otherwise this
    # fixture is a no-op and the wait does not happen for unit-only runs.
    if not _e2e_tests_selected(request):
        return

    deadline = time.monotonic() + _HEALTH_TIMEOUT_SECONDS
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            response = e2e_client.get("/health")
            if response.status_code == 200:
                return
            last_error = f"/health returned {response.status_code}"
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(_HEALTH_POLL_INTERVAL_SECONDS)

    pytest.skip(
        "Aegis-RAG stack is not reachable at "
        f"{e2e_client.base_url} ({last_error}). "
        "Run `docker compose up -d` and try again."
    )


@pytest.fixture(scope="session")
def stack_is_ready(e2e_client: httpx.Client) -> bool:
    """
    True only if the deep readiness check passes (ChromaDB + Ollama up).

    Tests that exercise the full RAG path should depend on this fixture
    and skip themselves when the LLM/vector store are not ready, since on
    first startup Ollama may still be pulling the model.
    """
    try:
        response = e2e_client.get("/ready")
    except httpx.HTTPError:
        return False
    return response.status_code == 200


# ── Internals ─────────────────────────────────────────────────────────────────


def _e2e_tests_selected(request: pytest.FixtureRequest) -> bool:
    """
    Returns True if the current pytest invocation is going to execute any
    e2e-marked tests. Avoids paying the 30s health-probe wait on plain unit runs.
    """
    for item in request.session.items:
        if item.get_closest_marker("e2e") is not None:
            return True
    return False
