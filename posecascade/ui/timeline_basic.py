"""Basic timeline dock — play/pause/scrub/loop over a frame range.

Owns a :class:`QTimer` running at the VMD-canonical 30 Hz and emits
``current_frame_changed(int)`` whenever the playhead moves (whether via
the timer, the slider, the spin box, or :meth:`set_current_frame`).
``playback_state_changed(bool)`` fires whenever the play/pause toggle
flips state.

The widget is animation-system-agnostic: it doesn't know about VMD or
the bone player. The owning controller wires it up by connecting
``current_frame_changed`` to whatever applies the frame (typically
:meth:`VmdAnimationPlayer.apply` after a frame→seconds conversion).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from posecascade.audio.player import AudioPlayer

# VMD's frame rate is fixed at 30 fps; we drive the timeline timer at the
# same cadence to match the standard MMD playback feel.
_DEFAULT_FPS = 30
_TIMER_INTERVAL_MS = 1000 // _DEFAULT_FPS

_DEFAULT_FRAME_RANGE = 1000
_MAX_FRAME_RANGE = 1_000_000


class TimelineDock(QDockWidget):
    """Bottom dock with play / pause / scrub / loop / range controls."""

    current_frame_changed = Signal(int)
    playback_state_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Timeline", parent)
        self._fps = _DEFAULT_FPS
        self._is_playing = False
        self._loop = True
        self._timer = QTimer(self)
        self._timer.setInterval(_TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)
        # When attached, the timer pulls the playhead from the audio
        # clock instead of advancing one frame per tick. ``None`` keeps
        # the default wall-clock-style behaviour.
        self._audio: AudioPlayer | None = None
        self._build_ui()

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def fps(self) -> int:
        return self._fps

    def current_frame(self) -> int:
        return int(self._slider.value())

    def set_current_frame(self, frame: int) -> None:
        clamped = max(self._range_start.value(), min(self._range_end.value(), int(frame)))
        if clamped == self._slider.value():
            return
        self._slider.setValue(clamped)

    def set_frame_range(self, start: int, end: int) -> None:
        if start > end:
            raise ValueError(f"frame range start {start} > end {end}")
        self._range_start.setRange(0, _MAX_FRAME_RANGE)
        self._range_end.setRange(0, _MAX_FRAME_RANGE)
        self._range_start.setValue(int(start))
        self._range_end.setValue(int(end))
        self._slider.setRange(int(start), int(end))

    def play(self) -> None:
        if self._is_playing:
            return
        self._is_playing = True
        if self._audio is not None:
            self._audio.seek(self.current_frame() / float(self._fps))
            self._audio.play()
        self._timer.start()
        self._play_button.setText("Pause")
        self.playback_state_changed.emit(True)

    def pause(self) -> None:
        if not self._is_playing:
            return
        self._is_playing = False
        self._timer.stop()
        if self._audio is not None:
            self._audio.pause()
        self._play_button.setText("Play")
        self.playback_state_changed.emit(False)

    def attach_audio(self, player: AudioPlayer) -> None:
        """Use ``player.current_time_seconds`` as the master clock."""
        self._audio = player

    def detach_audio(self) -> None:
        """Stop reading from the audio player; revert to wall-clock ticking."""
        self._audio = None

    def toggle_play(self) -> None:
        if self._is_playing:
            self.pause()
        else:
            self.play()

    def set_loop(self, enabled: bool) -> None:
        self._loop_checkbox.setChecked(bool(enabled))

    # ----- internal -----------------------------------------------------
    def _build_ui(self) -> None:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 4, 8, 8)

        slider_row = QHBoxLayout()
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, _DEFAULT_FRAME_RANGE)
        self._slider.setSingleStep(1)
        self._slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self._slider, 1)

        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, _DEFAULT_FRAME_RANGE)
        self._frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        slider_row.addWidget(self._frame_spin)
        layout.addLayout(slider_row)

        controls = QHBoxLayout()
        self._play_button = QPushButton("Play")
        self._play_button.clicked.connect(self.toggle_play)
        self._play_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        controls.addWidget(self._play_button)

        controls.addWidget(QLabel("Range:"))
        self._range_start = QSpinBox()
        self._range_start.setRange(0, _MAX_FRAME_RANGE)
        self._range_start.valueChanged.connect(self._on_range_start_changed)
        controls.addWidget(self._range_start)

        controls.addWidget(QLabel("→"))
        self._range_end = QSpinBox()
        self._range_end.setRange(0, _MAX_FRAME_RANGE)
        self._range_end.setValue(_DEFAULT_FRAME_RANGE)
        self._range_end.valueChanged.connect(self._on_range_end_changed)
        controls.addWidget(self._range_end)

        controls.addStretch(1)

        self._loop_checkbox = QCheckBox("Loop")
        self._loop_checkbox.setChecked(self._loop)
        self._loop_checkbox.toggled.connect(self._on_loop_toggled)
        controls.addWidget(self._loop_checkbox)

        layout.addLayout(controls)
        self.setWidget(container)

    def _on_tick(self) -> None:
        if self._audio is not None:
            audio_frame = int(round(self._audio.current_time_seconds() * self._fps))
            next_frame = audio_frame
        else:
            next_frame = self._slider.value() + 1
        if next_frame > self._range_end.value():
            if self._loop:
                next_frame = self._range_start.value()
                if self._audio is not None:
                    self._audio.seek(next_frame / float(self._fps))
            else:
                self.pause()
                return
        if next_frame != self._slider.value():
            self._slider.setValue(next_frame)

    def _on_slider_changed(self, value: int) -> None:
        if self._frame_spin.value() != value:
            self._frame_spin.blockSignals(True)
            self._frame_spin.setValue(value)
            self._frame_spin.blockSignals(False)
        self.current_frame_changed.emit(int(value))

    def _on_frame_spin_changed(self, value: int) -> None:
        if self._slider.value() != value:
            self._slider.setValue(value)

    def _on_range_start_changed(self, value: int) -> None:
        if value > self._range_end.value():
            self._range_end.setValue(value)
        self._slider.setMinimum(value)
        self._frame_spin.setMinimum(value)

    def _on_range_end_changed(self, value: int) -> None:
        if value < self._range_start.value():
            self._range_start.setValue(value)
        self._slider.setMaximum(value)
        self._frame_spin.setMaximum(value)

    def _on_loop_toggled(self, enabled: bool) -> None:
        self._loop = bool(enabled)
