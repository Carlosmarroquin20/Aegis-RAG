# 0005 — `X-Forwarded-For` is not trusted unless a proxy depth is configured

**Status:** Accepted

## Context

Throttling failed authentication (ADR 0006) needs a client address to bucket by. Behind a
reverse proxy the TCP peer is the proxy, and the real client is in `X-Forwarded-For`.
Reached directly, the TCP peer *is* the client, and `X-Forwarded-For` is whatever the caller
typed. The application cannot tell which situation it is in: both look identical from
inside the process, and this service is deployed both ways (behind nginx in compose,
directly on a published port in the Terraform deployment).

## Decision

Make the operator state it. `TRUSTED_PROXY_HOPS` is the number of proxies trusted to report
the client address:

- `0` (default) — use the TCP peer and ignore the header entirely.
- `N` — use the N-th entry from the right, the one appended by the outermost trusted proxy.
  Entries further left were supplied by the client and are never used.

If the header is absent or shorter than the configured depth, fall back to the peer: that
request did not come through the expected chain.

## Consequences

- The default cannot be bypassed. With no proxy configured, sending a different
  `X-Forwarded-For` on each attempt changes nothing.
- Behind a proxy, the default is wrong in a *visible* way: every client shares the proxy's
  address, so ten bad keys from anyone lock everyone out for a minute. That is noisy,
  obvious in the logs, and fixed by one setting.
- The asymmetry is the reason for the default. The opposite mistake — trusting the header
  with no proxy in front — fails *silently*: the throttle appears to work and can be evaded
  at will.

## Alternatives considered

- **Always trust the leftmost entry.** The common tutorial approach, and trivially spoofable.
- **Auto-detect a proxy from the presence of the header.** Rejected: the attacker controls
  whether the header is present.
- **A list of trusted proxy CIDRs.** More precise and the usual choice for large
  deployments. A hop count is enough for the topologies this project ships and has no
  address list to keep in sync with the infrastructure.
