# Load tests

k6 scenarios run against the compose stack, and what they found. Every number below was
measured on a laptop (Docker Desktop on Windows, 12 vCPUs shared with k6 and the rest of the
stack), with one uvicorn worker per API container. Absolute figures depend on that machine;
the before/after comparisons are what carry over.

## Running them

```bash
# The API as it would run in production: JSON logs at INFO, no host port, rate limit lifted.
docker compose -f docker-compose.yml -f loadtest/compose.loadtest.yml up -d --wait api

# One scenario at a time: protection, stress, ratelimit or rag.
docker compose -f docker-compose.yml -f loadtest/compose.loadtest.yml run --rm k6 \
  run -e SCENARIO=protection /scripts/aegis.js
```

| Scenario | What it sends | Pass criteria |
|---|---|---|
| `protection` | Prompt-injection attempts at a steady `RATE` (default 40/s, 60 s). The gateway blocks every one with a 400, so nothing reaches retrieval or the LLM: this is the cost of the API itself (middleware stack, authentication, the Redis rate limiter, the gateway). | every request a 400; p95 < 100 ms, p99 < 250 ms |
| `stress` | The same requests on a rate rising to 400/s, to find the ceiling. | every request a 400 (latency reported, not gated) |
| `ratelimit` | 300 requests from 30 concurrent clients on one key, at the normal limit (start the API with `RATE_LIMIT_REQUESTS=60`). | at most 70 admitted (60 + burst 10); every 429 carries `Retry-After` |
| `rag` | Questions from the example corpus through retrieval and generation, 2 clients for 2 minutes. Seed the corpus first. Dominated by the LLM. | reported, not gated |

`k6/redis_outage.js` checks a failure mode instead of a load: run it and pause Redis for
10 seconds in the middle (`docker pause aegis-rag-redis-1`, then `docker unpause`). It
passes when `/health`, which does not use Redis, keeps a p99 under 200 ms.

To measure several replicas, scale the API (`--scale api=3`); k6 reaches them through the
compose network's DNS, and they share the Redis rate-limit window.

## What the first runs found

The first `protection` run capped at **37 requests per second**, with a p95 of 6.4 s, for
requests the gateway rejects in under a millisecond. Profiling the container under load with
py-spy found three independent defects, each now fixed and covered by a test.

### 1. The rate limiter read back the whole window on every request

The Redis store returned every timestamp in the client's window (`ZRANGE 0 -1 WITHSCORES`),
and the limiter used only the count and the oldest one. A check therefore cost more the
higher the limit: with the limit lifted for the test, transferring and parsing the window
took **56% of the API's CPU**. A service account with a generous quota would hit the same
wall in production.

The store now returns the count and the oldest entry (`ZCARD` and a one-member `ZRANGE`
in the same transaction), so a check costs the same whatever the limit. The ceiling on one
worker rose from **37 to about 70 requests per second**; the rest of that cost is now spread
across the middleware stack rather than concentrated anywhere.

### 2. Garbage collection was the p99

At 40 requests per second, well under that ceiling, the median was 8 ms and the p99 almost
a second. The stalls were the garbage collector: startup leaves about **520,000** long-lived
objects (torch, the embedding model, the ChromaDB client) that every full collection walked,
and single pauses of **1.3 s** froze the event loop with every request in flight.

Startup now ends with `gc.freeze()`, which moves those objects to a generation the collector
no longer visits. Collections are timed in `aegis_gc_pause_seconds`, and `AegisLongGCPauses`
fires if full collections exceed 250 ms again.

| `protection` at 40/s, 1 replica | Before | After `gc.freeze()` |
|---|---:|---:|
| median | 8.2 ms | 7.2–8.3 ms |
| p95 | 45 ms | 9.6–15.5 ms |
| p99 | 974 ms | 13.7–73 ms |

The "after" column spans 18 one-minute runs on a freshly started Docker engine. Runs on an
engine that had been up for hours under repeated load were noisier (p99 up to 190 ms, and
once far worse, on an engine that then had to be restarted): the host, not the API, which is
why every comparison here was repeated and alternated.

### 3. A stalled Redis froze the whole process

The Redis client is synchronous and was called on the event loop. With Redis paused for 10
seconds, `/health`, which never touches it, reached a **p99 of 10.25 s**: every request in
the process waited for the limiter's socket timeouts. An orchestrator's liveness probe would
have restarted healthy replicas over a Redis blip.

Calls to a store that does I/O now run in a worker thread, as the ChromaDB adapter's already
did. With the same 10 s pause, `/health` kept a **p99 of 39 ms**. The thread hop costs nothing
measurable in normal operation: alternating the images with and without the change, nine
runs each at 40/s, gave the same latency (p95 9.6–14.7 ms with it, 10.1–15.5 ms without).

## Scaling out

| `protection` | 1 replica | 3 replicas |
|---|---:|---:|
| ceiling (`stress`) | 75 req/s | 121 req/s |
| p95 at a steady 40/s | 10–16 ms | — |
| p95 at a steady 100/s | — | 39 ms |

Each replica saturates one core, and the load spread evenly across all three. Throughput does
not triple because the three containers, k6 and Redis share one laptop. The `ratelimit`
scenario against the same three replicas admitted exactly **70 of 300** requests: the limit
holds across processes because the window lives in Redis.

The application runs one worker per container on purpose. Prometheus metrics are kept per
process, so several workers in one container would each report only part of the traffic to a
scrape; one worker per replica, scaled by the orchestrator, keeps every count exact.

## Known limits

- The remaining per-request cost is the Starlette `BaseHTTPMiddleware` machinery, which
  creates tasks and streams for each of the seven layers. Rewriting them as pure ASGI
  middleware is the next lever; it is not done because their order carries security
  guarantees that the current tests lock down.
- The `rag` scenario measures the configured LLM more than this code, and its results are not
  recorded here: in earlier live tests a hosted model answered in one to two seconds and the
  local 3B model on CPU in 20 to 90, and a free hosted tier rate limits a sustained run.
