# Security Policy

## Reporting a Vulnerability

If you discover a security issue in Aegis-RAG, please report it privately via
GitHub's **Security → Report a vulnerability** advisory flow rather than opening a
public issue. You can expect an initial acknowledgement within a few days.

## Threat Model

What the system defends against, where each control is enforced and tested, and its known
limits are documented in [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

## Dependency Vulnerability Management

Every push runs `pip-audit` against the fully resolved dependency set as part of
CI (see `.github/workflows/ci.yml`). The build fails on any advisory that has a
fixed release available, so patchable vulnerabilities cannot merge.

Dependabot (`.github/dependabot.yml`) opens weekly update pull requests for the Python
lockfile, the frontend, GitHub Actions, container images and Terraform providers, and opens
security updates as soon as an advisory is published. Each one runs the full pipeline before
it can merge, so staying current is routine rather than a reaction to a red build.

Advisories with **no fixed upstream release** are triaged individually. Where the
described vector is not reachable in this system's deployment model, the advisory
is formally risk-accepted, recorded below (VEX-style), and suppressed in CI with
an explicit `--ignore-vuln` entry. Every acceptance carries a re-evaluation date.

## Container Image Scanning

The production Docker image is scanned by Trivy on every push (see the
`docker-build` job in `.github/workflows/ci.yml`), failing the build on
`CRITICAL`/`HIGH` findings. The scan is configured to fail only on issues we can
actually act on:

- **`ignore-unfixed: true`** — advisories with no fixed version are not
  remediable by a version bump. The ones that matter (ChromaDB) are risk-accepted
  in the register below; the remainder are transient base-image OS advisories that
  clear as the upstream `python:3.11-slim` image is rebuilt.
- **`.trivyignore`** — a short, commented list of advisories that report a fix but
  where the fix is not directly applicable (packages vendored inside `setuptools`'
  `_vendor` tree), each assessed as a non-reachable build-time code path.

The image itself is kept minimal (CPU-only PyTorch, multi-stage build, no build
tools in the final layer) which also keeps the scan's attack surface small.

## Supply Chain

What runs in CI and what ships are both pinned and verifiable:

- **Pinned inputs.** Every GitHub Action is referenced by full commit SHA, and both
  Dockerfiles pin their base images by digest. A tag can be re-pointed after the fact; a
  commit or a digest cannot. Dependabot moves the pins and their version comments together.
- **Static analysis.** CodeQL analyses the Python service, the TypeScript frontend and the
  workflows themselves on every push and pull request (`.github/workflows/codeql.yml`).
  OpenSSF Scorecard scores the repository's practices weekly (`scorecard.yml`).
- **Signed releases.** Each GitHub release publishes the API and frontend images to GHCR
  (`release.yml`). Every image digest is signed keylessly with Sigstore cosign, using the
  workflow's OIDC identity, so there is no long-lived signing key to leak. It also carries a
  SLSA build provenance attestation and an SPDX SBOM, which is attached to the release too.

To verify an image before running it:

```bash
IMAGE=ghcr.io/carlosmarroquin20/aegis-rag-api:1.3.0

# Keyless signature: made by this repository's release workflow, nothing else
cosign verify "$IMAGE"   --certificate-identity-regexp '^https://github.com/Carlosmarroquin20/Aegis-RAG/.github/workflows/release.yml@'   --certificate-oidc-issuer https://token.actions.githubusercontent.com

# SLSA build provenance: which commit and which workflow run built this digest
gh attestation verify "oci://$IMAGE" --repo Carlosmarroquin20/Aegis-RAG
```

## Accepted Risks Register

> Deployment assumptions that back these acceptances: ChromaDB runs as an
> **internal-only** service reachable solely by the Aegis API process; it is never
> exposed to untrusted network clients. Aegis uses a **single tenant / single
> collection** with a **fixed, trusted embedding model** and never sets or forwards
> `trust_remote_code`, nor does it relay user-controlled model repositories to
> ChromaDB.

| Advisory | Package | Description | Reachable here? | Rationale |
|---|---|---|---|---|
| PYSEC-2026-311 / CVE-2026-45829 | chromadb | Pre-auth code injection via a malicious model repository with `trust_remote_code=true` on a collections endpoint. | No | ChromaDB's HTTP API is not exposed to untrusted clients; Aegis never enables `trust_remote_code` nor forwards attacker-controlled model repositories. |
| CVE-2026-45833 | chromadb | Authenticated code injection via `trust_remote_code` with `UPDATE_COLLECTION` permission. | No | Same as above; there is a single trusted client (the Aegis API) and `trust_remote_code` is never enabled. |
| CVE-2026-45830 | chromadb | Missing authorization validation lets an authenticated user act across tenants. | No | Single-tenant, single-collection deployment; there is no second tenant to cross into, and no untrusted authenticated principals. |
| CVE-2026-45831 | chromadb | `SimpleRBACAuthorizationProvider` does not scope permissions to a tenant/database/collection. | No | Aegis does not rely on ChromaDB's RBAC provider for isolation; access is single-tenant behind the API-key-authenticated Aegis gateway. |

**Status:** No fixed ChromaDB release addresses these advisories at the time of
writing. Aegis pins no vulnerable behavior on the affected code paths.

**Next review:** 2026-12-01, or immediately upon publication of a patched
ChromaDB release — whichever comes first. When a fix ships, bump the dependency
and remove the corresponding `--ignore-vuln` entries from CI.

_Last reviewed: 2026-09-01._
