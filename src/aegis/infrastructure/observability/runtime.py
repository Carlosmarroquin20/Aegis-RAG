"""
Keeps the garbage collector out of request latency, and measures it.

The API process holds about half a million long-lived objects once startup has
loaded torch, the embedding model and the ChromaDB client. CPython's collector
still traverses all of them on every full (generation 2) collection, and
request traffic allocates enough to trigger one regularly: under load, single
pauses of over a second froze the event loop, and every request in flight with
it. That was the API's p99 latency, not anything a request did.

``gc.freeze()`` moves everything alive at that point to a permanent generation
the collector no longer visits. Done once, after startup and before the first
request, it leaves the collector only the objects requests create. Anything
frozen is never collected, which costs nothing here: it is the long-lived state
the process keeps until it exits.

The pause histogram makes a regression visible, for example a new dependency
that builds a large structure after startup has frozen the heap.
"""

from __future__ import annotations

import gc
import time
from typing import Any

from aegis.infrastructure.observability.metrics import gc_pause_seconds

# Label children, bound when the callback is installed: it runs on every
# collection, thousands of times a minute, so it must not look labels up.
_pauses: dict[int, Any] = {}
_started_at: float | None = None


def freeze_startup_heap() -> int:
    """
    Collects once, then freezes every surviving object. Call after startup has
    built its long-lived state and before serving traffic. Returns how many
    objects were frozen.
    """
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count()


def _on_gc(phase: str, info: dict[str, Any]) -> None:
    global _started_at
    # Collections never overlap: the collector holds the GIL and is not reentrant.
    if phase == "start":
        _started_at = time.perf_counter()
    elif _started_at is not None:
        _pauses[info["generation"]].observe(time.perf_counter() - _started_at)
        _started_at = None


def install_gc_pause_metrics() -> None:
    """Records every collection's duration in ``aegis_gc_pause_seconds``. Idempotent."""
    _pauses.update({g: gc_pause_seconds.labels(generation=str(g)) for g in (0, 1, 2)})
    if _on_gc not in gc.callbacks:
        gc.callbacks.append(_on_gc)
