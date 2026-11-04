# 0020 — Run the injection classifier in shadow mode, never blocking

**Status:** Accepted

## Context

[0019](0019-ml-injection-classifier-measured-not-enabled.md) measured an ML prompt-injection
classifier on the gateway's 326 labelled queries. It closed the signature gateway's gaps, but
refused 17.4% of legitimate queries, so it was not put in the request path. The labelled corpus
is small and written by hand, though. How the classifier behaves on a deployment's real
questions, and whether real attackers send what the signatures miss, is something only real
traffic can show.

## Decision

Add `INJECTION_CLASSIFIER_MODE=shadow`, which defaults to `off`. In shadow mode:

- **Every query the gateway allows is also scored,** using its raw text (the form the classifier
  was evaluated on), by `ShadowInjectionMonitor` over the `InjectionScorerPort`.
- **The score never changes the outcome.** It is computed in the background, after the request
  has moved on, so no answer waits for it, and no score, flag or classifier failure can delay,
  block or fail a query. There is deliberately no blocking mode: the setting accepts `off` or
  `shadow` and nothing else.
- **Scores at or above `INJECTION_CLASSIFIER_THRESHOLD` are flags.** They are counted in
  `aegis_injection_shadow_total{outcome="flagged"}` and logged as
  `security.shadow_classifier_flagged` with the query hash, the classifier's score and the
  gateway's score. The query text is not logged, consistent with the rest of the service.
- **Background work is bounded.** With `INJECTION_CLASSIFIER_MAX_PENDING` scores already in
  flight, a new query is skipped and counted as `busy` instead of queued. A classifier failure is
  counted as `error`. The model loads at startup, and shutdown waits for scores in flight.

A Grafana row shows the flag rate, flags per day, skipped and failed scores, and the score
distribution.

## Consequences

- **The flag rate becomes a measurement on real traffic.** If it stays near zero, the classifier
  has little to add for this deployment. If it is high, the logged hashes let someone with access
  to the queries tell attacks the gateway misses from the classifier's false positives. That is
  the evidence a later decision to enforce would need, and it would get its own ADR.
- **No protection is added.** A flagged attack was still answered. Shadow mode measures; the other
  layers of [0002](0002-heuristic-gateway-is-not-the-safety-boundary.md) still do the defending.
- **It costs CPU, not latency.** It takes about 140 ms of CPU per allowed query (measured in CI)
  and about 700 MB of image when the model is baked in. Under load the bound turns excess traffic
  into `busy` counts rather than a backlog, so the flag rate covers the queries that were scored,
  not every query.
- **Blocked queries are not scored.** The gateway has already refused them, and comparing two
  refusals says nothing new.

## Alternatives considered

- **Score inline, concurrently with retrieval.** This hides part of the latency, but not 140 ms,
  and a slow or failing model would then sit on the request path.
- **Score in a separate worker from a queue.** It is cleaner at scale, but it needs a queue and
  another process for a measurement that an in-process bounded task already provides.
- **Log the flagged query text.** Reviewing flags would be easier, but the service logs only query
  hashes so that logs never become a store of what users asked. Flagged queries can be joined to
  any query log that is kept under its own access controls.
- **A blocking mode behind the same setting.** One mistyped value away from refusing one
  legitimate query in six. Enforcing should be a separate, measured decision.
