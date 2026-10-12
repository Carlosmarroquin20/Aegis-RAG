# 0003 — Documents are trusted at ingestion and untrusted at generation

**Status:** Accepted (recorded retrospectively)

## Context

A retrieved document is the classic carrier for indirect prompt injection: text that says
"ignore your instructions" ends up in the model's context because it was semantically close
to the question. The obvious response is to run uploaded documents through the same gateway
that screens queries and reject the ones that trip it.

## Decision

Do **not** screen document content at ingestion. Accept documents from authenticated
operators as source material, and treat their content as hostile at the only point where it
can do harm — generation:

- the system prompt, identical for every backend, declares the context untrusted data;
- each excerpt is framed with attribution so it reads as quoted material;
- the output guard fails the request if the answer echoes an injection (ADR 0008).

Ingestion still defends the *service* against the file: type detection from magic bytes, a
size limit enforced before buffering, an uncompressed-size budget for archives, and parsing
off the event loop.

## Consequences

- Legitimate documents are never rejected for what they discuss. A security policy, an
  incident report or this repository's own threat model all contain the phrases the gateway
  blocks; a knowledge base that cannot index its security documentation is not useful.
- The defence does not depend on recognising the payload in advance. Screening at ingestion
  only stops attacks that match a known signature, and gives a false sense that indexed
  content is clean.
- The mitigation is probabilistic. No prompt guarantees a model will not follow an
  instruction in its context; this is stated as a residual risk. Operators should not ingest
  sources they would not trust to email their users.

## Alternatives considered

- **Run the gateway on document text and reject on a match.** Rejected for the false
  positives above, and because a rejection at upload time tells an attacker exactly which
  phrasing to change.
- **Flag suspicious documents but still index them.** Reasonable and not yet built; it would
  add a review signal without changing the trust model. Listed as future work, not as a
  control.
- **Strip instruction-like sentences from documents.** Rejected: it silently alters source
  material, which defeats the purpose of grounding answers in it.
