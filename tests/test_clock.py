"""Tests for :class:`posecascade.utils.clock.FrameClock`."""
from __future__ import annotations

import time

from posecascade.utils.clock import FrameClock


def test_first_tick_returns_zero() -> None:
    clock = FrameClock()
    assert clock.tick() == 0.0


def test_subsequent_ticks_are_positive() -> None:
    clock = FrameClock()
    clock.tick()
    time.sleep(0.005)
    assert clock.tick() > 0.0


def test_clamps_hitch() -> None:
    clock = FrameClock(max_dt=0.05)
    clock.tick()
    clock._last_perf -= 5.0  # noqa: SLF001 — simulate a 5-second hitch
    assert clock.tick() == 0.05
    assert 0.0 < clock.elapsed <= 0.05


def test_no_clamp_when_disabled() -> None:
    clock = FrameClock(max_dt=0.0)
    clock.tick()
    clock._last_perf -= 1.0  # noqa: SLF001
    dt = clock.tick()
    assert dt > 0.5


def test_reset_zeros_state() -> None:
    clock = FrameClock()
    clock.tick()
    time.sleep(0.001)
    clock.tick()
    clock.reset()
    assert clock.elapsed == 0.0
    assert clock.tick() == 0.0
