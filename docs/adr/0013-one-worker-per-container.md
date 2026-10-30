# 0013 — One worker per container; scale out with replicas

**Status:** Accepted

## Context

The image runs uvicorn with a single worker, and the load tests showed what that costs: one
container saturates one core at about 70 requests per second for queries the gateway rejects.
The obvious fix is more workers per container. But several things in the process are
per-process by nature: Prometheus counters, the LLM circuit breakers, the in-memory caches,
and the embedding model, which takes most of a container's roughly 900 MB.

## Decision

Keep one worker per container and scale by running more containers. Shared state that has to
be global, the rate-limit window and the daily token counts, lives in Redis, so it already
spans replicas.

## Consequences

- Every `/metrics` scrape reports the whole of its process's traffic. With several workers in
  one container, each scrape would reach one worker at random and report a fraction of the
  counts, and the dashboards and alerts would be wrong in a way nobody would notice.
- Scaling is the orchestrator's job (replicas behind a service), which is how it is done on
  Kubernetes or ECS anyway. Three replicas in the load test shared the load evenly, and the
  rate limit held exactly across them.
- Memory per unit of throughput is higher than with workers sharing a container: each replica
  loads its own copy of the embedding model.
- Circuit breakers learn a backend's health per replica, at the cost of up to the failure
  threshold in failed calls each ([0009](0009-llm-fallback-behind-circuit-breakers.md)).

## Alternatives considered

- **Several workers with Prometheus multiprocess mode.** It works, but it needs a shared
  directory of metric files, changes how gauges aggregate, and has to be cleaned on restart;
  every metric added later has to be checked against it. One more thing to get subtly wrong,
  for what replicas already give.
- **Several workers and accept partial metrics.** Rejected: an alert on a counter that sees a
  third of the traffic is an alert that does not fire.
- **A threaded or async-only speed-up instead.** The cost per request was spread across the
  middleware stack, not one hot spot, so there was no single change to make one worker several
  times faster.
