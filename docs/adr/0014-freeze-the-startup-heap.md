# 0014 — Freeze the startup heap so garbage collection stays out of the tail

**Status:** Accepted

## Context

Under a steady 40 requests per second, well under the API's ceiling, the median latency was
8 ms and the p99 almost a second. Timing the garbage collector showed why: startup leaves about
520,000 long-lived objects (torch, the embedding model, the ChromaDB client), every full
collection walked all of them, and request traffic triggered one regularly. Single pauses of
1.3 s stopped the event loop, and every request in flight with it.

## Decision

End startup with `gc.collect()` followed by `gc.freeze()`, once everything long-lived has been
built and before the first request. Frozen objects move to a permanent generation the
collector no longer visits, so a full collection only walks what requests created.

Every collection's duration is recorded in `aegis_gc_pause_seconds` by generation, and
`AegisLongGCPauses` fires when full collections exceed 250 ms over ten minutes.

## Consequences

- At 40 requests per second the p99 fell from 974 ms to between 14 and 73 ms over 18 runs.
- Anything frozen is never collected. That is the state the process keeps until it exits, so
  nothing is lost; garbage that happened to be alive at the freeze would be kept, which is why
  the freeze follows a full collection.
- The protection holds only for objects that exist at the freeze. A dependency that later builds
  a large long-lived structure brings the pauses back; the metric and the alert exist to show
  that.
- Startup takes a little longer: one full collection over the whole heap, before any traffic.

## Alternatives considered

- **Disable the collector.** Removes the pauses and leaks every reference cycle requests create.
- **Raise the collection thresholds.** Fewer full collections, each just as long: the tail
  moves, it does not shrink.
- **More replicas.** More capacity, the same pauses on each; the p99 is set by the pause, not
  by load.
