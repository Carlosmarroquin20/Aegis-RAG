"""
Which tenant an API key belongs to, loaded once at startup.

Tenants are declared in a TOML file (TENANTS_FILE). The file holds the SHA-256
of each key, never the key itself, so it is configuration rather than a secret:
it can live in a config map or in version control, and leaking it does not let
anyone authenticate. Keys are handed to callers out of band; scripts/tenant_key.py
generates one together with the digest to put here.

    [[tenant]]
    id = "support"
    collection = "support-docs"
    api_key_sha256 = ["9f86d081884c7d65..."]
    rate_limit_requests = 120        # optional, per window
    daily_token_budget = 2_000_000   # optional

Without a file, every key in VALID_API_KEYS becomes its own tenant, named by
the first 16 hex characters of its digest and sharing the configured collection.
That is exactly how a single-team deployment behaved before tenants existed:
the same rate-limit buckets, the same metric labels, the same documents.

Loading is strict. Two tenants sharing a key or a collection would each see
the other's data, so either is a startup error, not a warning.
"""

from __future__ import annotations

import hashlib
import hmac
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aegis.domain.models.tenant import InvalidTenantError, Tenant

# 64 bits of the key digest: enough to keep the keys of a single-team
# deployment apart, useless for recovering or verifying a key.
LEGACY_TENANT_ID_HEX_CHARS = 16

_FIELDS = {"id", "collection", "api_key_sha256", "rate_limit_requests", "daily_token_budget"}


def key_digest(api_key: str) -> bytes:
    """
    The SHA-256 of an API key: what the registry stores and compares.

    A plain SHA-256, deliberately not a password hash such as bcrypt or Argon2.
    Slow hashes exist to make guessing low-entropy, human-chosen passwords
    expensive. API keys are random tokens (startup warns about short ones), so
    guessing one is infeasible at any hash speed. A slow hash would instead
    hand every unauthenticated caller a lever to burn CPU on each request.
    """
    return hashlib.sha256(api_key.encode()).digest()


@dataclass(frozen=True, slots=True)
class _Credential:
    digest: bytes
    tenant: Tenant


class TenantRegistry:
    """Maps API-key digests to tenants, comparing in constant time."""

    def __init__(
        self,
        credentials: Sequence[tuple[bytes, Tenant]],
        *,
        shared_collections: bool = False,
    ) -> None:
        seen_ids: dict[str, Tenant] = {}
        seen_collections: dict[str, str] = {}
        seen_digests: set[bytes] = set()
        for digest, tenant in credentials:
            if digest in seen_digests:
                raise InvalidTenantError(
                    f"an API key is listed more than once (last under tenant {tenant.id!r}); "
                    "a key must identify exactly one tenant"
                )
            seen_digests.add(digest)
            known = seen_ids.setdefault(tenant.id, tenant)
            if known != tenant:
                raise InvalidTenantError(f"tenant id {tenant.id!r} is declared twice")
            owner = seen_collections.setdefault(tenant.collection, tenant.id)
            if owner != tenant.id and not shared_collections:
                raise InvalidTenantError(
                    f"tenants {owner!r} and {tenant.id!r} share collection "
                    f"{tenant.collection!r}; each tenant needs its own"
                )
        self._credentials = tuple(_Credential(d, t) for d, t in credentials)
        self._tenants = tuple(seen_ids.values())

    @property
    def tenants(self) -> tuple[Tenant, ...]:
        return self._tenants

    def __len__(self) -> int:
        return len(self._credentials)

    def resolve(self, api_key: str) -> Tenant | None:
        """
        The tenant ``api_key`` belongs to, or None.

        Every configured digest is compared on every call, with no early exit,
        so the time taken does not reveal how close a guess came or which tenant
        it matched. Digesting first also equalizes lengths, which compare_digest
        would otherwise leak.
        """
        candidate = key_digest(api_key)
        match: Tenant | None = None
        for credential in self._credentials:
            if hmac.compare_digest(candidate, credential.digest):
                match = credential.tenant
        return match

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_api_keys(cls, keys: Iterable[str], collection: str) -> TenantRegistry:
        """One tenant per key, all on ``collection``: the single-team behaviour."""
        credentials = []
        for key in sorted(keys):
            digest = key_digest(key)
            tenant = Tenant(id=digest.hex()[:LEGACY_TENANT_ID_HEX_CHARS], collection=collection)
            credentials.append((digest, tenant))
        # Shared on purpose: these keys all belonged to one team before tenants.
        return cls(credentials, shared_collections=True)

    @classmethod
    def from_file(cls, path: str | Path) -> TenantRegistry:
        try:
            document = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise InvalidTenantError(f"cannot read tenants file {str(path)!r}: {exc}") from exc
        return cls.from_mapping(document)

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> TenantRegistry:
        entries = document.get("tenant")
        if not isinstance(entries, list) or not entries:
            raise InvalidTenantError("the tenants file declares no [[tenant]] tables")
        credentials: list[tuple[bytes, Tenant]] = []
        for entry in entries:
            unknown = set(entry) - _FIELDS
            if unknown:
                # A misspelt "daily_token_budjet" would otherwise silently mean
                # "unlimited".
                raise InvalidTenantError(f"unknown tenant field(s): {', '.join(sorted(unknown))}")
            tenant = Tenant(
                id=str(entry.get("id", "")),
                collection=str(entry.get("collection", "")),
                rate_limit_requests=_optional_int(entry, "rate_limit_requests"),
                daily_token_budget=_optional_int(entry, "daily_token_budget"),
            )
            digests = entry.get("api_key_sha256") or []
            if not isinstance(digests, list) or not digests:
                raise InvalidTenantError(f"tenant {tenant.id!r} has no api_key_sha256")
            credentials.extend((_parse_digest(tenant.id, d), tenant) for d in digests)
        return cls(credentials)


def _optional_int(entry: Mapping[str, Any], name: str) -> int | None:
    value = entry.get(name)
    # bool is an int subclass; `true` is never a meaningful limit.
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    raise InvalidTenantError(f"tenant {entry.get('id')!r}: {name} must be an integer")


def _parse_digest(tenant_id: str, value: object) -> bytes:
    text = str(value).strip().lower()
    try:
        digest = bytes.fromhex(text)
    except ValueError:
        digest = b""
    if len(digest) != hashlib.sha256().digest_size:
        raise InvalidTenantError(
            f"tenant {tenant_id!r}: api_key_sha256 entries must be 64 hex characters"
        )
    return digest
