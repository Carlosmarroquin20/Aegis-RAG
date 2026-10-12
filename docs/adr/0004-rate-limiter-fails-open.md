# 0004 — The rate limiter fails open when its store is unreachable

**Status:** Accepted

## Context

With the Redis backend, the per-key rate limiter makes a network call on every authenticated
request. When Redis was unreachable that call raised inside the middleware, so every request
failed with a 500 for the duration of the outage — an undefined behaviour nobody had chosen.
A policy had to be picked: serve the request without throttling, or refuse it.

## Decision

Fail **open** by default (`RATE_LIMIT_FAIL_OPEN=true`): the request is served unthrottled.
Setting it to `false` refuses with a retryable 503 instead.

In both modes the event is never silent: it is logged, counted in
`aegis_rate_limit_backend_errors_total`, and covered by the `AegisRateLimitBackendDown`
alert. The Redis client has 0.5 s connect and read timeouts so a hung store fails fast into
this policy instead of stalling requests.

## Consequences

- An outage of an auxiliary cache does not become an outage of the API. The limiter is an
  abuse control layered on top of authentication, and authentication has already succeeded
  by the time it runs.
- While the store is down, an authenticated client is unmetered. The exposure is bounded to
  holders of a valid key and is visible on a dashboard within one scrape interval.
- The default is wrong for some deployments, which is why it is a setting. Where each
  request has a direct cost — a metered hosted LLM (ADR 0007) is the obvious case — failing
  closed is the better trade.

## Alternatives considered

- **Fail closed by default.** The more conservative-sounding choice, rejected as the default
  because it makes availability of the whole service depend on a component whose loss
  compromises no confidentiality or integrity property.
- **Fall back to the in-memory store.** Rejected: with several workers each would keep its
  own window, so the limit would be silently multiplied while appearing to work.
- **Leave it as an unhandled error.** Not a policy. It was fail-closed with a misleading
  status code and no metric.
