"""
Tests for tenant definitions and the registry that maps API keys to them.

The failure cases matter more than the happy path here: a registry that loads
with two tenants on one collection, or one key under two tenants, would leak
documents between teams without any error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aegis.domain.models.tenant import InvalidTenantError, Tenant
from aegis.infrastructure.tenancy.registry import (
    LEGACY_TENANT_ID_HEX_CHARS,
    TenantRegistry,
    key_digest,
)

_SUPPORT_KEY = "support-test-key"
_BILLING_KEY = "billing-test-key"


def _entry(tenant_id: str, collection: str, *keys: str, **extra: object) -> dict[str, object]:
    return {
        "id": tenant_id,
        "collection": collection,
        "api_key_sha256": [key_digest(k).hex() for k in keys],
        **extra,
    }


def _registry(*entries: dict[str, object]) -> TenantRegistry:
    return TenantRegistry.from_mapping({"tenant": list(entries)})


class TestTenant:
    @pytest.mark.parametrize("tenant_id", ["", "Support", "has space", "a" * 41, "-leading"])
    def test_ids_are_short_plain_and_lowercase(self, tenant_id: str) -> None:
        with pytest.raises(InvalidTenantError, match="tenant id"):
            Tenant(id=tenant_id, collection="docs")

    @pytest.mark.parametrize("collection", ["ab", "has space", "-bad", "x" * 64])
    def test_collections_follow_chroma_naming(self, collection: str) -> None:
        with pytest.raises(InvalidTenantError, match="collection"):
            Tenant(id="support", collection=collection)

    @pytest.mark.parametrize("field", ["rate_limit_requests", "daily_token_budget"])
    def test_limits_are_positive(self, field: str) -> None:
        with pytest.raises(InvalidTenantError, match=field):
            Tenant(id="support", collection="docs", **{field: 0})  # type: ignore[arg-type]


class TestResolution:
    def test_each_key_resolves_to_its_own_tenant(self) -> None:
        registry = _registry(
            _entry("support", "support-docs", _SUPPORT_KEY),
            _entry("billing", "billing-docs", _BILLING_KEY),
        )
        support = registry.resolve(_SUPPORT_KEY)
        billing = registry.resolve(_BILLING_KEY)

        assert support is not None and support.collection == "support-docs"
        assert billing is not None and billing.collection == "billing-docs"

    def test_a_tenant_may_hold_several_keys(self) -> None:
        """So a key can be rotated: add the new one, roll clients over, drop the old."""
        registry = _registry(_entry("support", "support-docs", _SUPPORT_KEY, "rotated-key-xyz"))
        assert registry.resolve(_SUPPORT_KEY) == registry.resolve("rotated-key-xyz")

    @pytest.mark.parametrize("key", ["", "wrong", _SUPPORT_KEY[:-1]])
    def test_unknown_keys_resolve_to_nobody(self, key: str) -> None:
        registry = _registry(_entry("support", "support-docs", _SUPPORT_KEY))
        assert registry.resolve(key) is None


class TestTheFileIsStrict:
    def test_a_key_cannot_belong_to_two_tenants(self) -> None:
        with pytest.raises(InvalidTenantError, match="more than once"):
            _registry(
                _entry("support", "support-docs", _SUPPORT_KEY),
                _entry("billing", "billing-docs", _SUPPORT_KEY),
            )

    def test_two_tenants_cannot_share_a_collection(self) -> None:
        with pytest.raises(InvalidTenantError, match="share collection"):
            _registry(
                _entry("support", "shared-docs", _SUPPORT_KEY),
                _entry("billing", "shared-docs", _BILLING_KEY),
            )

    def test_a_tenant_id_cannot_be_declared_twice(self) -> None:
        with pytest.raises(InvalidTenantError, match="declared twice"):
            _registry(
                _entry("support", "support-docs", _SUPPORT_KEY),
                _entry("support", "other-docs", _BILLING_KEY),
            )

    def test_a_misspelt_field_is_an_error_not_a_default(self) -> None:
        with pytest.raises(InvalidTenantError, match="daily_token_budjet"):
            _registry(_entry("support", "support-docs", _SUPPORT_KEY, daily_token_budjet=10))

    @pytest.mark.parametrize("value", ["100", 1.5, True])
    def test_limits_must_be_integers(self, value: object) -> None:
        with pytest.raises(InvalidTenantError, match="must be an integer"):
            _registry(_entry("support", "support-docs", _SUPPORT_KEY, rate_limit_requests=value))

    @pytest.mark.parametrize("digest", ["abc", "z" * 64, key_digest("k").hex()[:-2]])
    def test_digests_must_be_sha256_hex(self, digest: str) -> None:
        entry = {"id": "support", "collection": "support-docs", "api_key_sha256": [digest]}
        with pytest.raises(InvalidTenantError, match="64 hex"):
            _registry(entry)

    def test_every_tenant_needs_a_key(self) -> None:
        with pytest.raises(InvalidTenantError, match="no api_key_sha256"):
            _registry({"id": "support", "collection": "support-docs"})

    def test_an_empty_file_is_an_error(self) -> None:
        with pytest.raises(InvalidTenantError, match="no \\[\\[tenant\\]\\]"):
            TenantRegistry.from_mapping({})


class TestLoadingFromToml:
    def test_reads_the_documented_format(self, tmp_path: Path) -> None:
        path = tmp_path / "tenants.toml"
        path.write_text(
            f"""
[[tenant]]
id = "support"
collection = "support-docs"
api_key_sha256 = ["{key_digest(_SUPPORT_KEY).hex().upper()}"]
rate_limit_requests = 120
daily_token_budget = 2_000_000
""",
            encoding="utf-8",
        )
        tenant = TenantRegistry.from_file(path).resolve(_SUPPORT_KEY)

        assert tenant == Tenant(
            id="support",
            collection="support-docs",
            rate_limit_requests=120,
            daily_token_budget=2_000_000,
        )

    def test_an_unreadable_file_names_itself(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.toml"
        with pytest.raises(InvalidTenantError, match="missing.toml"):
            TenantRegistry.from_file(missing)

    def test_invalid_toml_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "tenants.toml"
        path.write_text("[[tenant]\nid = ", encoding="utf-8")
        with pytest.raises(InvalidTenantError, match="cannot read"):
            TenantRegistry.from_file(path)


class TestSingleTeamCompatibility:
    def test_every_key_is_a_tenant_named_as_before(self) -> None:
        """The id equals the client id the API used to derive, so buckets and labels carry over."""
        registry = TenantRegistry.from_api_keys([_SUPPORT_KEY, _BILLING_KEY], "aegis_documents")
        tenant = registry.resolve(_SUPPORT_KEY)

        assert tenant is not None
        assert tenant.id == key_digest(_SUPPORT_KEY).hex()[:LEGACY_TENANT_ID_HEX_CHARS]
        assert tenant.collection == "aegis_documents"

    def test_those_tenants_share_the_collection(self) -> None:
        registry = TenantRegistry.from_api_keys([_SUPPORT_KEY, _BILLING_KEY], "aegis_documents")
        assert {t.collection for t in registry.tenants} == {"aegis_documents"}
        assert len(registry.tenants) == 2

    def test_no_keys_means_no_tenants(self) -> None:
        assert len(TenantRegistry.from_api_keys([], "aegis_documents")) == 0
