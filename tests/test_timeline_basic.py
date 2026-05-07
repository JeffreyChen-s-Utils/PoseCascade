"""Qt smoke tests for the basic timeline dock."""
from __future__ import annotations

import pytest

from posecascade.ui.timeline_basic import TimelineDock


@pytest.fixture
def timeline(qapp: object) -> TimelineDock:
    dock = TimelineDock()
    dock.set_frame_range(0, 20)
    return dock


def test_initial_state(timeline: TimelineDock) -> None:
    assert timeline.is_playing is False
    assert timeline.current_frame() == 0
    assert timeline.fps == 30


def test_set_current_frame_clamps_to_range(timeline: TimelineDock) -> None:
    timeline.set_current_frame(5)
    assert timeline.current_frame() == 5
    timeline.set_current_frame(999)
    assert timeline.current_frame() == 20
    timeline.set_current_frame(-5)
    assert timeline.current_frame() == 0


def test_current_frame_changed_signal(timeline: TimelineDock) -> None:
    received: list[int] = []
    timeline.current_frame_changed.connect(received.append)
    timeline.set_current_frame(7)
    assert received == [7]
    # Idempotent — same value should not re-emit.
    timeline.set_current_frame(7)
    assert received == [7]


def test_play_pause_toggles_state(timeline: TimelineDock) -> None:
    states: list[bool] = []
    timeline.playback_state_changed.connect(states.append)
    timeline.play()
    assert timeline.is_playing
    timeline.play()                     # idempotent — no second emission
    assert states == [True]
    timeline.pause()
    assert not timeline.is_playing
    assert states == [True, False]


def test_toggle_flips_state(timeline: TimelineDock) -> None:
    timeline.toggle_play()
    assert timeline.is_playing
    timeline.toggle_play()
    assert not timeline.is_playing


def test_loop_wraps_to_range_start(timeline: TimelineDock) -> None:
    """Without invoking the QTimer, drive ``_on_tick`` directly to verify the
    loop wrap behaves the same way the timer-driven path does."""
    timeline.set_loop(True)
    timeline.set_current_frame(20)
    timeline._on_tick()
    assert timeline.current_frame() == 0


def test_no_loop_pauses_at_end(timeline: TimelineDock) -> None:
    timeline.set_loop(False)
    timeline.play()
    timeline.set_current_frame(20)
    timeline._on_tick()
    assert not timeline.is_playing
    assert timeline.current_frame() == 20


def test_range_start_above_end_pulls_end_along(timeline: TimelineDock) -> None:
    timeline.set_frame_range(0, 10)
    timeline._range_start.setValue(50)
    assert timeline._range_end.value() >= 50


def test_range_end_below_start_pulls_start_along(timeline: TimelineDock) -> None:
    timeline.set_frame_range(20, 30)
    timeline._range_end.setValue(5)
    assert timeline._range_start.value() <= 5


def test_set_frame_range_rejects_inverted(timeline: TimelineDock) -> None:
    with pytest.raises(ValueError, match="frame range start"):
        timeline.set_frame_range(50, 10)
