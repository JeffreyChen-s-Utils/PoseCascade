"""Tests for :class:`AudioPlayer` and :class:`TimelineDock.attach_audio`."""
from __future__ import annotations

import pytest

from posecascade.audio.clip import load_wav_bytes
from posecascade.audio.player import AudioPlayer
from posecascade.ui.timeline_basic import TimelineDock
from tests.fixtures.audio.build_wav import build_sine_wav_bytes


def _player_with_fake_clock(clip) -> tuple[AudioPlayer, list[float]]:    # noqa: ANN001 — clip type is internal
    """Build a player whose clock is a settable list-of-times."""
    holder = {"now": 0.0}

    def fake_clock() -> float:
        return holder["now"]

    player = AudioPlayer(clip=clip)
    player._set_clock_for_test(fake_clock)        # noqa: SLF001
    return player, holder    # type: ignore[return-value]


def test_fallback_clock_advances_during_play() -> None:
    clip = load_wav_bytes(build_sine_wav_bytes(duration_seconds=1.0))
    player, holder = _player_with_fake_clock(clip)
    assert player.current_time_seconds() == pytest.approx(0.0)
    player.play()
    assert player.is_playing
    holder["now"] = 0.25
    assert player.current_time_seconds() == pytest.approx(0.25, abs=1e-6)
    holder["now"] = 0.6
    assert player.current_time_seconds() == pytest.approx(0.6, abs=1e-6)


def test_pause_freezes_fallback_clock() -> None:
    clip = load_wav_bytes(build_sine_wav_bytes(duration_seconds=1.0))
    player, holder = _player_with_fake_clock(clip)
    player.play()
    holder["now"] = 0.4
    player.pause()
    assert not player.is_playing
    paused_time = player.current_time_seconds()
    assert paused_time == pytest.approx(0.4, abs=1e-6)
    # Real-time keeps advancing but the player's clock holds.
    holder["now"] = 1.0
    assert player.current_time_seconds() == pytest.approx(paused_time, abs=1e-6)


def test_seek_changes_current_time_when_paused() -> None:
    clip = load_wav_bytes(build_sine_wav_bytes(duration_seconds=2.0))
    player, _holder = _player_with_fake_clock(clip)
    player.seek(0.75)
    assert player.current_time_seconds() == pytest.approx(0.75)


def test_seek_during_play_keeps_position_consistent() -> None:
    clip = load_wav_bytes(build_sine_wav_bytes(duration_seconds=2.0))
    player, holder = _player_with_fake_clock(clip)
    player.play()
    holder["now"] = 0.2
    player.seek(1.5)
    holder["now"] = 0.4    # 0.2 wall-clock seconds after the seek
    assert player.current_time_seconds() == pytest.approx(1.7, abs=1e-6)


def test_seek_clamps_to_zero_for_negative_input() -> None:
    clip = load_wav_bytes(build_sine_wav_bytes(duration_seconds=1.0))
    player, _ = _player_with_fake_clock(clip)
    player.seek(-1.0)
    assert player.current_time_seconds() == pytest.approx(0.0)


def test_attach_qt_returns_false_when_clip_has_no_path() -> None:
    """No on-disk path → Qt media player can't load → fallback path stays."""
    clip = load_wav_bytes(build_sine_wav_bytes())
    player = AudioPlayer(clip=clip)
    assert player.attach_qt() is False
    assert player._qt_player is None     # noqa: SLF001


# ----- timeline integration --------------------------------------------
@pytest.fixture
def timeline_with_audio(qapp: object):       # noqa: ANN001 — qapp fixture object
    clip = load_wav_bytes(build_sine_wav_bytes(duration_seconds=2.0))
    player, holder = _player_with_fake_clock(clip)
    dock = TimelineDock()
    dock.set_frame_range(0, 60)              # 2 s at 30 fps
    dock.attach_audio(player)
    return dock, player, holder


def test_timeline_pulls_frame_from_audio_clock(timeline_with_audio) -> None:
    dock, player, holder = timeline_with_audio
    dock.play()
    holder["now"] = 0.5                       # fallback-clock advances 0.5 s
    dock._on_tick()                            # noqa: SLF001
    assert dock.current_frame() == int(round(0.5 * dock.fps))


def test_timeline_play_seeks_audio_to_current_frame(timeline_with_audio) -> None:
    dock, player, _holder = timeline_with_audio
    dock.set_current_frame(15)
    dock.play()
    assert player.is_playing
    assert player.current_time_seconds() == pytest.approx(15 / dock.fps, abs=1e-6)


def test_timeline_pause_pauses_audio(timeline_with_audio) -> None:
    dock, player, _holder = timeline_with_audio
    dock.play()
    assert player.is_playing
    dock.pause()
    assert not player.is_playing


def test_timeline_loop_seeks_audio_back_to_range_start(timeline_with_audio) -> None:
    dock, player, holder = timeline_with_audio
    dock.set_loop(True)
    dock.play()
    holder["now"] = 3.0                       # well past 2s clip
    dock._on_tick()                            # noqa: SLF001
    assert dock.current_frame() == 0
    assert player.current_time_seconds() == pytest.approx(0.0)


def test_timeline_detach_audio_falls_back_to_internal_tick(timeline_with_audio) -> None:
    dock, _player, _ = timeline_with_audio
    dock.detach_audio()
    dock.set_current_frame(5)
    dock._on_tick()                            # noqa: SLF001
    assert dock.current_frame() == 6
