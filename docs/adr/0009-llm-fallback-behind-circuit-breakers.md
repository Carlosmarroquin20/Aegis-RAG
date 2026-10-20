# 0009 — Fall back between LLM backends behind per-backend circuit breakers

**Status:** Accepted

## Context

With a hosted provider as the main backend ([0007](0007-local-llm-by-default-hosted-opt-in.md)),
its quotas and outages become ours. In a live test the free Groq tier rate limited the key
after about ten questions in a row, and each refusal reached the user as a 503. The local
Ollama model was running next to the API the whole time and could have answered, more
slowly. The question is how a request should get from one to the other, and what it must
not do on the way.

## Decision

An optional second backend (`LLM_FALLBACK_PROVIDER`), tried in order behind a
`FallbackLLMClient` that implements the same `LLMClientPort`, so nothing above the wiring
changes. Each backend has a consecutive-failure circuit breaker:

- `LLM_CIRCUIT_FAILURE_THRESHOLD` failures in a row (default 3) open it; it is then skipped
  for `LLM_CIRCUIT_RESET_SECONDS` (default 30) and a single trial request decides whether
  it closes.
- A rate-limited reply opens it immediately, for the provider's `Retry-After` when that is
  longer than the reset period.
- Only backend errors (unavailable, timeout, rate limited) are routed around. Anything else
  is a defect: it is raised, and still counted against the backend so its circuit cannot
  stay half-open.
- When every backend fails, the last error keeps its type, so a timeout is still a 504.

Data locality is decided by the operator through the order, not by this code. The expected
configuration is a hosted primary with the local model behind it, which keeps data on the
host whenever it falls back. The reverse (a hosted fallback behind the local model) is
allowed and logged at startup, because in that order data leaves the host only when the
local model fails, and nothing else says so.

## Consequences

- A provider's rate limit or outage costs latency and quality instead of availability. In a
  live test with the hosted key revoked, the local llama3.2 answered in 21 to 89 seconds on
  CPU (the first one exceeded the 120 s timeout), and two of three answers contradicted the
  documents. The fallback keeps the service answering; it does not make a 3B local model as
  good as the hosted one, and an operator should size and choose it with that in mind.
- A fallback sits idle by definition, and Ollama unloads an idle model after five minutes.
  Loading llama3.2 from its volume took about 3.5 minutes on a laptop's Docker VM, so a cold
  fallback would have failed exactly when needed. The compose file keeps the model resident
  (`OLLAMA_KEEP_ALIVE=-1`) and loads it at startup, at the cost of its memory held
  permanently.
- Every answer reports the backend and model that produced it, so usage and the
  per-provider cost metrics stay correct (an answer from Ollama is not billed at the hosted
  rate), and a degraded answer can be told apart.
- A failing backend stops costing every request its timeout after the threshold is reached.
  The first requests of an outage still pay it; that is the price of not opening on a
  single transient error.
- Breaker state is per process and in memory. Each worker learns a backend's health on its
  own, at the cost of up to the threshold in failed calls per worker. Sharing it through
  Redis was not worth a network round trip on every generation.
- Two new signals: `aegis_llm_fallbacks_total{from_provider, to_provider, reason}` and
  `aegis_llm_circuit_open{provider}`, with an alert while a circuit stays open.

## Alternatives considered

- **Retrying the same backend.** Rejected as the main mechanism: a rate limit and an outage
  are exactly the cases where an immediate retry fails again, and retries multiply the load
  on a provider that is already refusing it.
- **Failing over without a breaker.** Rejected: every request would wait for the primary to
  time out before trying the fallback, turning an outage into a minute-long delay per query.
- **A shared breaker in Redis.** Rejected for now; see the per-process consequence above.
- **A gateway product in front of the providers** (LiteLLM, a cloud AI gateway). Reasonable
  at larger scale, but it moves the error handling, the credential hygiene and the cost
  accounting out of this codebase into another service to run and secure.
