# 0011 — Pin every build input to an immutable reference, and sign releases keylessly

**Status:** Accepted

## Context

The code this repository runs in CI and ships to users is mostly not code it wrote: GitHub
Actions, base images, the toolchain. They were referenced by tag (`actions/checkout@v7`,
`python:3.11-slim`), and a tag is a pointer its owner can move. Compromised actions have had
their tags repointed to malicious commits, and a re-pushed base image changes what a rebuild
of an unchanged Dockerfile produces. On the other side, the images users would run carried
nothing to show they came from this repository's workflow.

## Decision

**Inputs are pinned to what cannot move.** Every action is referenced by full commit SHA, with
its version in a comment. Both Dockerfiles pin their base images as `tag@sha256:digest`,
spelled out in each `FROM` rather than through build arguments, which Dependabot cannot
update. Dependabot moves each pin and its comment together, so pinning does not mean falling
behind.

**Outputs are verifiable.** Each GitHub release builds the API and frontend images and pushes
them to GHCR by digest. Each digest is:

- signed with Sigstore cosign in keyless mode, using the workflow's OIDC identity;
- given a SLSA build provenance attestation naming the commit and the run that built it;
- shipped with an SPDX SBOM, both in the registry and attached to the release.

**The repository checks itself.** CodeQL analyses the Python service, the TypeScript frontend
and the workflows; OpenSSF Scorecard scores the repository's practices weekly.

## Consequences

- A tag moved upstream changes nothing here until a reviewed Dependabot PR moves the pin, with
  the full CI behind it.
- There is no signing key to store, rotate or leak. A signature proves the image was built by
  `release.yml` in this repository; `SECURITY.md` has the `cosign verify` and
  `gh attestation verify` commands that check it.
- Pins are less readable than tags, hence the version comments, and every upstream release is
  now a PR to review. That is the price of the guarantee.
- Images are built for `linux/amd64` only: the CPU-only torch wheels make an emulated arm64
  build impractically slow.
- Publishing a release now publishes images. It is a deliberate, confirmed action, not a side
  effect of a push.

## Alternatives considered

- **Floating major tags.** Convenient and common; rejected because a tag is exactly the
  reference an attacker who compromises an action's repository can move.
- **Signing with a long-lived key.** Rejected: the key becomes the most valuable secret in the
  project and needs storage, rotation and revocation. Keyless signing ties the signature to
  the workflow's identity instead.
- **An SBOM without signatures, or signatures without provenance.** Each answers a different
  question (what is inside, who built it, from what), and a consumer auditing an image needs
  all three.
