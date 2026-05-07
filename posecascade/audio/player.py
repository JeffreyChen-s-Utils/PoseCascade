"""Audio playback wrapper with a wall-clock fallback.

When ``PySide6.QtMultimedia`` is available + the audio backend
initialises, the player drives a :class:`QMediaPlayer` for actual
output and reads ``position()`` for the master clock. When either is
unavailable (headless CI, no audio device, missing module) the player
silently falls back to advancing an internal ``time.monotonic`` clock
on each :meth:`current_time_seconds` query — the timeline still gets a
monotonic frame counter so unit tests + offline rendering work.

The fallback path is the same code path the timeline uses today via
its :class:`QTimer`; routing through :class:`AudioPlayer` lets the
caller swap between "no audio" and "wave file" without changing the
timeline's API.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from posecascade.audio.clip import AudioClip


@dataclass
class AudioPlayer:
    """Common play / pause / seek surface above QMediaPlayer + a wall clock."""

    clip: AudioClip | None = None
    _qt_player: object | None = field(default=None, init=False, repr=False)
    _qt_audio_output: object | None = field(default=None, init=False, repr=False)
    _is_playing: bool = field(default=False, init=False)
    _fallback_seek: float = field(default=0.0, init=False)
    _fallback_play_started_at: float | None = field(default=None, init=False)
    _now: Callable[[], float] = field(default=time.monotonic, init=False, repr=False)

    def attach_qt(self) -> bool:
        """Best-effort wire up Qt's media player.

        Returns ``True`` when the backend came up cleanly. Failures are
        treated as "headless mode": :meth:`current_time_seconds` still
        works through the fallback clock, the caller doesn't need to
        special-case anything.
        """
        if self.clip is None or self.clip.path is None:
            return False
        try:
            from PySide6.QtCore import QUrl  # noqa: PLC0415
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer  # noqa: PLC0415
        except ImportError:
            return False
        try:
            qt_player = QMediaPlayer()
            audio_output = QAudioOutput()
            qt_player.setAudioOutput(audio_output)
            qt_player.setSource(QUrl.fromLocalFile(str(self.clip.path)))
        except Exception:                                            # noqa: BLE001 — broad on purpose
            # Any failure inside Qt's audio init drops to fallback mode.
            return False
        self._qt_player = qt_player
        self._qt_audio_output = audio_output
        return True

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def play(self) -> None:
        if self._is_playing:
            return
        self._is_playing = True
        if self._qt_player is not None:
            self._qt_player.play()
            return
        self._fallback_play_started_at = self._now()

    def pause(self) -> None:
        if not self._is_playing:
            return
        # Capture the running fallback time BEFORE flipping the flag —
        # ``_fallback_current_time`` reads ``_is_playing`` to decide
        # whether to accumulate elapsed wall-clock time.
        new_seek = self._fallback_current_time()
        self._is_playing = False
        if self._qt_player is not None:
            self._qt_player.pause()
            return
        self._fallback_seek = new_seek
        self._fallback_play_started_at = None

    def seek(self, time_seconds: float) -> None:
        """Move the playhead to ``time_seconds``.

        Works regardless of play state — pause + seek + resume keeps the
        new position. The underlying Qt media player and the fallback
        clock are kept in sync so :meth:`current_time_seconds` reports
        the same value across the two modes.
        """
        clamped = max(0.0, float(time_seconds))
        if self._qt_player is not None:
            self._qt_player.setPosition(int(clamped * 1000))
        self._fallback_seek = clamped
        if self._is_playing:
            self._fallback_play_started_at = self._now()

    def current_time_seconds(self) -> float:
        if self._qt_player is not None:
            return float(self._qt_player.position()) / 1000.0
        return self._fallback_current_time()

    def duration_seconds(self) -> float:
        if self.clip is not None:
            return float(self.clip.duration_seconds)
        return 0.0

    # ----- internal -----------------------------------------------------
    def _fallback_current_time(self) -> float:
        if self._is_playing and self._fallback_play_started_at is not None:
            return self._fallback_seek + (self._now() - self._fallback_play_started_at)
        return self._fallback_seek

    # Test seam: lets unit tests inject a deterministic clock.
    def _set_clock_for_test(self, fn: Callable[[], float]) -> None:
        self._now = fn
