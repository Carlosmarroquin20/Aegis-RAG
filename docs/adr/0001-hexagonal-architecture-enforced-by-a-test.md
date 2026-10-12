# 0001 — Hexagonal architecture, with the dependency rule enforced by a test

**Status:** Accepted (recorded retrospectively)

## Context

The system sits between three things that are certain to change independently: the vector
store, the LLM backend, and the security controls. A RAG prototype usually wires these
together directly — the route calls the vendor client — which is fast to write and makes
every later swap a rewrite. It also makes the security-critical orchestration impossible to
test without the real services.

## Decision

Four layers, each depending only inward:

```
interface  →  application  →  domain  ←  infrastructure
```

The domain defines ports (`VectorStorePort`, `LLMClientPort`, `QueryGuardPort`,
`OutputGuardPort`, `QueryMetricsPort`, `ParserResolverPort`); infrastructure implements
them; the object graph is assembled in exactly one module (`interface/api/dependencies.py`).

The rule is **enforced, not documented**: `tests/unit/test_architecture.py` parses the
imports of every module and fails on any that points outward, including imports deferred
inside functions or guarded by `TYPE_CHECKING`.

## Consequences

- Swapping a backend is a change to one factory. This was exercised for real: a hosted LLM
  provider was added as a second adapter (ADR 0007) without touching the use case, the
  routes or the sanitizer.
- Use cases are tested against in-memory fakes, so the security-relevant ordering
  (guard → retrieve → generate → sanitize) is verified in milliseconds with no services.
- More files and more indirection than a direct implementation. Each new capability a use
  case needs from the outside world costs a port method, a fake and an adapter.

## Alternatives considered

- **Layering by convention only.** Rejected after it failed in practice: before the test
  existed, the use cases imported the ChromaDB adapter and reached into its private
  attributes, and nothing flagged it. Deferred imports made the violation invisible to a
  reviewer reading the top of the file.
- **An import-linter dependency.** Equivalent result; a 60-line test in the suite has no
  extra dependency and its failure message names the offending line.
