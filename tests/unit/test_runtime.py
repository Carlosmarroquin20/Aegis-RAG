"""
Tests for the garbage-collector controls in infrastructure/observability/runtime.py.

Freezing is process-wide state, so each test that freezes unfreezes afterwards;
otherwise every later test would run against a frozen heap.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator

import pytest

from aegis.infrastructure.observability import runtime
from aegis.infrastructure.observability.metrics import registry


@pytest.fixture()
def unfreeze_after() -> Iterator[None]:
    yield
    gc.unfreeze()


@pytest.fixture()
def gc_callback_removed_after() -> Iterator[None]:
    yield
    if runtime._on_gc in gc.callbacks:
        gc.callbacks.remove(runtime._on_gc)


def _pause_count(generation: int) -> float:
    value = registry.get_sample_value(
        "aegis_gc_pause_seconds_count", {"generation": str(generation)}
    )
    return value or 0.0


class TestFreezeStartupHeap:
    def test_moves_live_objects_out_of_the_collectors_reach(self, unfreeze_after: None) -> None:
        survivor = [object() for _ in range(1_000)]
        frozen = runtime.freeze_startup_heap()

        assert frozen == gc.get_freeze_count() >= len(survivor)
        # Frozen objects are no longer tracked by any generation.
        assert all(count == 0 for count in gc.get_count()[1:])

    def test_objects_created_afterwards_are_still_collected(self, unfreeze_after: None) -> None:
        runtime.freeze_startup_heap()

        class Node:
            def __init__(self) -> None:
                self.self_ref = self

        for _ in range(100):
            Node()  # reference cycles: only the collector can reclaim them
        assert gc.collect() >= 100


class TestGCPauseMetrics:
    def test_every_collection_is_timed_by_generation(
        self, reset_prometheus_registry: None, gc_callback_removed_after: None
    ) -> None:
        runtime.install_gc_pause_metrics()
        before = _pause_count(2)
        gc.collect(2)
        gc.collect(2)

        assert _pause_count(2) == before + 2

    def test_installing_twice_does_not_double_count(
        self, reset_prometheus_registry: None, gc_callback_removed_after: None
    ) -> None:
        runtime.install_gc_pause_metrics()
        runtime.install_gc_pause_metrics()
        before = _pause_count(0)
        gc.collect(0)

        assert _pause_count(0) == before + 1
