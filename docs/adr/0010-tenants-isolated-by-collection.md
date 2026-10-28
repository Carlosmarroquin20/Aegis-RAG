# 0010 — Tenants are isolated by collection, identified by key, and budgeted in tokens

**Status:** Accepted

## Context

Run as an internal platform, one deployment serves several teams. Each team must see only its
own documents, must not be able to exhaust the service for the others, and must be accountable
for what it spends: on a hosted model, LLM tokens are the bill. Until now the service had a
flat list of API keys, one collection and one request limit for everyone.

## Decision

A **tenant** is declared in a TOML file (`TENANTS_FILE`): an id, its own collection, the SHA-256
of each of its keys, and optionally its own request limit and a daily token budget.

- **Identity comes from the key, and only from it.** `APIKeyMiddleware` resolves the tenant
  before any route runs and leaves it on the request. Nothing in a body, a query string or a
  header can name another tenant.
- **Isolation is by collection.** Every route reaches the collection of the caller's tenant and
  no other, so a list, a search or a delete by id cannot cross tenants. A route reached without
  a tenant raises instead of falling back to a default collection.
- **The file holds key digests, not keys.** It is configuration, not a secret: it can live in a
  config map or in version control, and leaking it does not let anyone authenticate. A tenant
  may hold several keys, so a key can be rotated without downtime.
- **Loading is strict.** A key under two tenants, two tenants on one collection, a duplicated id,
  an unknown field or a non-integer limit stops the process at startup. Each of these would
  otherwise leak data or silently lift a limit.
- **Limits are per tenant.** The rate-limit bucket is the tenant, so all its keys share one
  allowance. A daily token budget is checked after the security gateway and before retrieval,
  and charged after generation, on the rate limiter's Redis so it spans replicas. A spent
  budget answers 429 with a `Retry-After` until 00:00 UTC.
- **Without a file, nothing changes.** Each key in `VALID_API_KEYS` becomes a tenant named as
  its client id already was, on `CHROMA_COLLECTION`, with the deployment-wide limit and no
  budget: the same buckets, the same metric labels, the same documents.

## Consequences

- One tenant's documents cannot reach another's answers through retrieval, which matters more
  in a RAG system than elsewhere: whatever is retrieved is shown to the model and can be quoted
  back. An integration test drives the real application with two tenants through every path;
  with the collection hard-wired to one tenant, three of its four cases fail.
- The token budget is a **soft** limit. Concurrent requests near the limit can each pass the
  check before any of them is charged, so a tenant can overshoot by what it has in flight.
  Reserving tokens up front would need the length of an answer before it exists.
- A ledger outage serves queries without a cost limit rather than refusing them, as the rate
  limiter does for the same reason ([0004](0004-rate-limiter-fails-open.md)). It is counted and
  alerted on.
- Tenant ids are metric labels (`client` on the usage and cost series), so cost per team is a
  query, not a report. Their cardinality is bounded by the file.
- Adding or changing a tenant means editing the file and restarting. There is no admin API,
  which is one less authenticated surface to secure.
- Each collection is created and checked for its embedding model at startup, and `/ready` is
  degraded if any of them is unreachable: one tenant's broken collection is a deployment
  problem, not something to discover on its first query.

## Alternatives considered

- **One collection with a tenant field in each chunk's metadata, filtered on every query.**
  Rejected: isolation would depend on every present and future code path remembering the
  filter, and one that forgets it leaks silently. Separate collections make the safe path the
  only path.
- **A tenant header or path segment chosen by the caller, checked against the key.** Rejected:
  it adds a value that has to be validated on every route for no benefit, since the key already
  determines the tenant.
- **Plain keys in the tenants file.** Rejected: the file would become a secret, and anyone who
  can read the deployment's configuration could authenticate as any tenant.
- **A database of tenants with an admin API.** Reasonable at a larger scale; here it would add a
  privileged endpoint, its own authorization model and a datastore, for a list that changes a
  few times a year.
