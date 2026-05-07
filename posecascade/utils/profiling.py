"""Frame-level profiling helpers.

Usage::

    with frame_section("scene.cull"):
        cull_visible(scene, camera)

Sections accumulate per-frame timings into a thread-local registry that the
overlay UI reads at the end of each frame.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

_NS_PER_MS = 1_000_000


@dataclass
class FrameStats:
    """Accumulated section timings for the current frame, in milliseconds."""

    sections: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, milliseconds: float) -> None:
        self.sections[name] = self.sections.get(name, 0.0) + milliseconds

    def reset(self) -> None:
        self.sections.clear()


_thread_local = threading.local()


def current_stats() -> FrameStats:
    """Return the calling thread's :class:`FrameStats` instance."""
    stats = getattr(_thread_local, "stats", None)
    if stats is None:
        stats = FrameStats()
        _thread_local.stats = stats
    return stats


@contextmanager
def frame_section(name: str) -> Iterator[None]:
    """Time the wrapped block and add its duration to the current frame stats."""
    start_ns = time.perf_counter_ns()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter_ns() - start_ns) / _NS_PER_MS
        current_stats().add(name, elapsed_ms)
