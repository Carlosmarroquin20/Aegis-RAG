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
| [0015](0015-multilingual-embeddings-recorded-per-collection.md) | A multilingual embedding model, recorded in every collection | Accepted; model superseded by 0023 |
| [0016](0016-hybrid-retrieval-by-default-reranker-opt-in.md) | Hybrid retrieval by default; the reranker is opt-in | Accepted |
| [0017](0017-answer-faithfulness-judged-by-nli.md) | Answer faithfulness is judged by a local NLI model, sentence by sentence | Accepted |
| [0018](0018-stream-whole-sanitized-sentences.md) | Stream answers as whole, sanitized sentences | Accepted |
| [0019](0019-ml-injection-classifier-measured-not-enabled.md) | An ML injection classifier is measured, not enabled | Accepted |
| [0020](0020-injection-classifier-in-shadow-mode.md) | Run the injection classifier in shadow mode, never blocking | Accepted |
| [0021](0021-larger-evaluation-sets.md) | Larger evaluation sets, and the gates re-set on them | Accepted |
| [0022](0022-faithfulness-reply-words-and-figure-check.md) | The faithfulness judge drops reply words and checks figures against the context | Accepted |
| [0023](0023-multilingual-e5-embeddings.md) | Multilingual E5 embeddings, with their query and passage markers | Accepted |
| [0024](0024-public-demo-on-free-hosting.md) | A public demo: the gateway in the browser, the full stack in one container | Accepted |

The first three were made when the system was designed and are recorded here retrospectively;
the rest were recorded as they were decided: 0004 to 0008 during the security hardening pass,
0009 to 0014 while running the service as a shared platform and putting it under load.
0015 records the embedding-model change of release 1.1.0 after the fact, 0016 the move to
hybrid retrieval, 0017 how answers are checked against their context, 0018 how answers
stream without giving up the output guard, 0019 why an ML injection classifier was measured
but not enabled, 0020 how it runs on real traffic without blocking anything, 0021 what the
larger evaluation sets showed and how the quality gates were re-set on them, 0022 the two
corrections to the faithfulness judge that came out of it, 0023 the embedding model chosen
on them, and 0024 the public demo. A summary of all of
them in Spanish is in [RESUMEN.md](RESUMEN.md).
For what the system defends against and where it stops, see the
[threat model](../THREAT_MODEL.md).
