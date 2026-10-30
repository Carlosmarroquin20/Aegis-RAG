# Architecture Decision Records

Each record captures one decision that had a real alternative: the context that forced a
choice, what was chosen, what it costs, and what was rejected. They are short on purpose —
the point is to make the reasoning reviewable, not to restate the code.

Records are immutable once accepted. A decision that changes gets a new record that
supersedes the old one, so the history of *why* stays intact.

| # | Decision | Status |
|---|---|---|
| [0001](0001-hexagonal-architecture-enforced-by-a-test.md) | Hexagonal architecture, with the dependency rule enforced by a test | Accepted |
| [0002](0002-heuristic-gateway-is-not-the-safety-boundary.md) | The input gateway is a heuristic first layer, not the safety boundary | Accepted |
| [0003](0003-documents-trusted-at-ingestion-untrusted-at-generation.md) | Documents are trusted at ingestion and untrusted at generation | Accepted |
| [0004](0004-rate-limiter-fails-open.md) | The rate limiter fails open when its store is unreachable | Accepted |
| [0005](0005-forwarded-headers-untrusted-by-default.md) | `X-Forwarded-For` is not trusted unless a proxy depth is configured | Accepted |
| [0006](0006-auth-lockout-applies-to-valid-keys.md) | A failed-authentication lockout also refuses valid keys | Accepted |
| [0007](0007-local-llm-by-default-hosted-opt-in.md) | Local LLM by default; hosted providers are opt-in through one adapter | Accepted |
| [0008](0008-output-reflection-fails-the-request.md) | A reflected injection fails the request instead of being cleaned up | Accepted |
| [0009](0009-llm-fallback-behind-circuit-breakers.md) | Fall back between LLM backends behind per-backend circuit breakers | Accepted |
| [0010](0010-tenants-isolated-by-collection.md) | Tenants are isolated by collection, identified by key, and budgeted in tokens | Accepted |
| [0011](0011-immutable-supply-chain-references-and-keyless-signing.md) | Pin every build input to an immutable reference, and sign releases keylessly | Accepted |
| [0012](0012-retrieval-quality-is-a-merge-gate.md) | Retrieval quality is a merge gate | Accepted |
| [0013](0013-one-worker-per-container.md) | One worker per container; scale out with replicas | Accepted |
| [0014](0014-freeze-the-startup-heap.md) | Freeze the startup heap so garbage collection stays out of the tail | Accepted |
| [0015](0015-multilingual-embeddings-recorded-per-collection.md) | A multilingual embedding model, recorded in every collection | Accepted |

The first three were made when the system was designed and are recorded here retrospectively;
the rest were recorded as they were decided: 0004 to 0008 during the security hardening pass,
0009 to 0014 while running the service as a shared platform and putting it under load.
0015 records the embedding-model change of release 1.1.0 after the fact. A summary of all of
them in Spanish is in [RESUMEN.md](RESUMEN.md).
For what the system defends against and where it stops, see the
[threat model](../THREAT_MODEL.md).
