# Security Policy

## Reporting a Vulnerability

If you discover a security issue in Aegis-RAG, please report it privately via
GitHub's **Security → Report a vulnerability** advisory flow rather than opening a
public issue. You can expect an initial acknowledgement within a few days.

## Dependency Vulnerability Management

Every push runs `pip-audit` against the fully resolved dependency set as part of
CI (see `.github/workflows/ci.yml`). The build fails on any advisory that has a
fixed release available, so patchable vulnerabilities cannot merge.

Advisories with **no fixed upstream release** are triaged individually. Where the
described vector is not reachable in this system's deployment model, the advisory
is formally risk-accepted, recorded below (VEX-style), and suppressed in CI with
an explicit `--ignore-vuln` entry. Every acceptance carries a re-evaluation date.

## Accepted Risks Register

> Deployment assumptions that back these acceptances: ChromaDB runs as an
> **internal-only** service reachable solely by the Aegis API process; it is never
> exposed to untrusted network clients. Aegis uses a **single tenant / single
> collection** with a **fixed, trusted embedding model** and never sets or forwards
> `trust_remote_code`, nor does it relay user-controlled model repositories to
> ChromaDB.

| Advisory | Package | Description | Reachable here? | Rationale |
|---|---|---|---|---|
| PYSEC-2026-311 | chromadb | Pre-auth code injection via a malicious model repository with `trust_remote_code=true` on a collections endpoint. | No | ChromaDB's HTTP API is not exposed to untrusted clients; Aegis never enables `trust_remote_code` nor forwards attacker-controlled model repositories. |
| CVE-2026-45833 | chromadb | Authenticated code injection via `trust_remote_code` with `UPDATE_COLLECTION` permission. | No | Same as above; there is a single trusted client (the Aegis API) and `trust_remote_code` is never enabled. |
| CVE-2026-45830 | chromadb | Missing authorization validation lets an authenticated user act across tenants. | No | Single-tenant, single-collection deployment; there is no second tenant to cross into, and no untrusted authenticated principals. |
| CVE-2026-45831 | chromadb | `SimpleRBACAuthorizationProvider` does not scope permissions to a tenant/database/collection. | No | Aegis does not rely on ChromaDB's RBAC provider for isolation; access is single-tenant behind the API-key-authenticated Aegis gateway. |

**Status:** No fixed ChromaDB release addresses these advisories at the time of
writing. Aegis pins no vulnerable behavior on the affected code paths.

**Next review:** 2026-12-01, or immediately upon publication of a patched
ChromaDB release — whichever comes first. When a fix ships, bump the dependency
and remove the corresponding `--ignore-vuln` entries from CI.

_Last reviewed: 2026-09-01._
