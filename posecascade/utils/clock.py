"""Per-frame clock — produces stable ``dt`` values bounded against frame hitches."""
from __future__ import annotations

import time
from dataclasses import dataclass

_DEFAULT_MAX_DT = 0.1  # cap a single hitch at 100ms so a stalled frame can't fling animations


@dataclass
class FrameClock:
    """Produces a delta-seconds value clamped against hitches.

    The clamp prevents a paused or freshly-resumed window from fast-forwarding
    every animation by N seconds in a single tick. Pass ``max_dt=0`` to disable.
    """

    max_dt: float = _DEFAULT_MAX_DT
    _last_perf: float | None = None
    _elapsed: float = 0.0

    def tick(self) -> float:
        """Advance the clock and return the clamped ``dt`` since the previous tick."""
        now = time.perf_counter()
        if self._last_perf is None:
            self._last_perf = now
            return 0.0
        dt = now - self._last_perf
        self._last_perf = now
        if self.max_dt > 0.0 and dt > self.max_dt:
            dt = self.max_dt
        self._elapsed += dt
        return dt

    @property
    def elapsed(self) -> float:
        """Total elapsed seconds since the first tick (post-clamp)."""
        return self._elapsed

    def reset(self) -> None:
        """Reset the clock (next ``tick`` returns 0)."""
        self._last_perf = None
        self._elapsed = 0.0
