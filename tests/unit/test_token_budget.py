"""
Tests for per-tenant daily token budgets: the ledgers and the use case's check.

The budget is a cost control, so the cases that matter are the ones that would
let cost through (no charge recorded, a day that never resets) and the ones
that would refuse paid-for work (a ledger outage turned into an outage).
"""

from __future__ import annotations

from datetime import UTC, datetime

import fakeredis
import pytest

from aegis.application.dtos.rag_dtos import QueryRequest
from aegis.application.use_cases.query_rag import QueryRAGUseCase
from aegis.domain.models.tenant import Tenant
from aegis.domain.ports.token_ledger import (
    DailyUsage,
    TokenBudgetExhaustedError,
    TokenLedgerPort,
)
from aegis.infrastructure.observability.metrics import PrometheusQueryMetrics, registry
from aegis.infrastructure.security.output_sanitizer import OutputSanitizer
from aegis.infrastructure.security.security_gateway import SecurityGateway
from aegis.infrastructure.tenancy.token_ledger import InMemoryTokenLedger, RedisTokenLedger
from tests.conftest import FakeLLMClient, FakeVectorStore


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 28, 23, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture()
def clock() -> _Clock:
    return _Clock()


@pytest.fixture(params=["memory", "redis"])
def ledger(request: pytest.FixtureRequest, clock: _Clock) -> TokenLedgerPort:
    if request.param == "redis":
        return RedisTokenLedger(client=fakeredis.FakeStrictRedis(), clock=clock)
    return InMemoryTokenLedger(clock=clock)


# ── Ledgers ───────────────────────────────────────────────────────────────────


class TestLedgers:
    async def test_charges_add_up_per_tenant(self, ledger: TokenLedgerPort) -> None:
        await ledger.charge("support", 100)
        await ledger.charge("support", 50)
        await ledger.charge("billing", 7)

        assert (await ledger.used_today("support")).tokens == 150
        assert (await ledger.used_today("billing")).tokens == 7
        assert (await ledger.used_today("nobody")).tokens == 0

    async def test_the_count_resets_at_utc_midnight(
        self, ledger: TokenLedgerPort, clock: _Clock
    ) -> None:
        await ledger.charge("support", 100)
        clock.now = datetime(2026, 10, 29, 0, 0, 1, tzinfo=UTC)
        assert (await ledger.used_today("support")).tokens == 0

    async def test_reports_when_the_budget_resets(self, ledger: TokenLedgerPort) -> None:
        # The clock reads 23:00 UTC: one hour to midnight.
        assert (await ledger.used_today("support")).seconds_to_reset == 3600

    async def test_redis_counters_expire_on_their_own(self, clock: _Clock) -> None:
        client = fakeredis.FakeStrictRedis()
        ledger = RedisTokenLedger(client=client, clock=clock)
        await ledger.charge("support", 1)
        ttl_after_first = client.ttl("aegis:tokens:support:2026-10-28")
        await ledger.charge("support", 1)

        assert 0 < ttl_after_first <= 2 * 24 * 3600
        # Set once, when the key is created: later charges do not extend it.
        assert client.ttl("aegis:tokens:support:2026-10-28") <= ttl_after_first


# ── The use case ──────────────────────────────────────────────────────────────


class _BrokenLedger(TokenLedgerPort):
    async def used_today(self, tenant_id: str) -> DailyUsage:
        raise ConnectionError("redis refused")

    async def charge(self, tenant_id: str, tokens: int) -> None:
        raise ConnectionError("redis refused")


_BUDGETED = Tenant(id="support", collection="support-docs", daily_token_budget=200)
_UNLIMITED = Tenant(id="billing", collection="billing-docs")


def _use_case(ledger: TokenLedgerPort, llm: FakeLLMClient) -> QueryRAGUseCase:
    return QueryRAGUseCase(
        vector_store=FakeVectorStore(),
        llm_client=llm,
        security_gateway=SecurityGateway(strict_mode=True),
        output_sanitizer=OutputSanitizer(),
        metrics=PrometheusQueryMetrics(),
        token_ledger=ledger,
    )


def _ask(use_case: QueryRAGUseCase, tenant: Tenant | None) -> object:
    return use_case.execute(
        QueryRequest(query="How many days do I have to return a mattress?"),
        client=tenant.id if tenant else "unknown",
        tenant=tenant,
    )


@pytest.fixture()
def llm() -> FakeLLMClient:
    return FakeLLMClient()  # every answer uses 120 + 30 tokens


@pytest.fixture(autouse=True)
def _clean_metrics(reset_prometheus_registry: None) -> None:
    return None


class TestBudgetEnforcement:
    async def test_every_answer_is_charged_to_the_tenant(self, llm: FakeLLMClient) -> None:
        ledger = InMemoryTokenLedger()
        await _ask(_use_case(ledger, llm), _BUDGETED)  # type: ignore[misc]
        assert (await ledger.used_today("support")).tokens == 150

    async def test_a_spent_budget_refuses_before_any_cost(self, llm: FakeLLMClient) -> None:
        ledger = InMemoryTokenLedger()
        use_case = _use_case(ledger, llm)
        await _ask(use_case, _BUDGETED)  # type: ignore[misc]
        await _ask(use_case, _BUDGETED)  # type: ignore[misc]  # 150 < 200: still admitted

        with pytest.raises(TokenBudgetExhaustedError) as refused:
            await _ask(use_case, _BUDGETED)  # type: ignore[misc]

        assert len(llm.calls) == 2  # the third never reached the model
        assert refused.value.retry_after > 0
        rejections = registry.get_sample_value(
            "aegis_token_budget_rejections_total", {"tenant": "support"}
        )
        assert rejections == 1

    async def test_a_tenant_without_a_budget_is_neither_checked_nor_charged(
        self, llm: FakeLLMClient
    ) -> None:
        ledger = InMemoryTokenLedger()
        for _ in range(5):
            await _ask(_use_case(ledger, llm), _UNLIMITED)  # type: ignore[misc]
        assert (await ledger.used_today("billing")).tokens == 0

    async def test_no_tenant_means_no_budget(self, llm: FakeLLMClient) -> None:
        await _ask(_use_case(_BrokenLedger(), llm), None)  # type: ignore[misc]
        assert registry.get_sample_value("aegis_token_ledger_errors_total") in (None, 0)

    async def test_a_ledger_outage_does_not_refuse_queries(self, llm: FakeLLMClient) -> None:
        response = await _ask(_use_case(_BrokenLedger(), llm), _BUDGETED)  # type: ignore[misc]

        assert response.answer  # type: ignore[attr-defined]
        # Once for the check, once for the charge: visible, not silent.
        assert registry.get_sample_value("aegis_token_ledger_errors_total") == 2
