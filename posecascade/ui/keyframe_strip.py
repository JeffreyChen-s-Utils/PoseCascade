"""Per-track keyframe strip — paints diamond markers for each keyframe.

A :class:`KeyframeStripWidget` is the per-row visual that pairs with a
:class:`MultiTrackTimelineDock`'s tree row: given a list of frame
numbers + the visible frame range, it paints the strip with one
diamond per keyframe and a vertical line at the current playhead.

Selection happens by a click within a marker's hit radius; the widget
emits :attr:`keyframe_clicked(frame: int)` so the integrator can flip
the document state and refresh.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygon, QPolygonF
from PySide6.QtWidgets import QWidget

_MARKER_RADIUS_PX = 6
_PLAYHEAD_COLOR = QColor(255, 200, 80)
_TRACK_BG_COLOR = QColor(38, 38, 42)
_KEYFRAME_FILL = QColor(220, 220, 230)
_KEYFRAME_OUTLINE = QColor(0, 0, 0)
_SELECTED_FILL = QColor(120, 180, 255)


@dataclass(frozen=True)
class _Layout:
    """Cached strip-coordinate math for one paint."""

    start_frame: int
    end_frame: int
    pixel_width: int

    def frame_to_x(self, frame: float) -> float:
        span = max(1, self.end_frame - self.start_frame)
        return (frame - self.start_frame) / span * self.pixel_width

    def x_to_frame(self, x: float) -> int:
        span = max(1, self.end_frame - self.start_frame)
        return int(round(self.start_frame + x / max(1, self.pixel_width) * span))


class KeyframeStripWidget(QWidget):
    """Horizontal strip with one marker per keyframe + a playhead line."""

    keyframe_clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frames: tuple[int, ...] = ()
        self._selected_frame: int | None = None
        self._current_frame: int = 0
        self._range_start: int = 0
        self._range_end: int = 100
        self.setMinimumHeight(20)

    # ----- public API --------------------------------------------------
    def set_frames(self, frames: Sequence[int]) -> None:
        self._frames = tuple(int(f) for f in frames)
        self.update()

    def set_frame_range(self, start: int, end: int) -> None:
        if start > end:
            start, end = end, start
        self._range_start = int(start)
        self._range_end = int(end)
        self.update()

    def set_current_frame(self, frame: int) -> None:
        self._current_frame = int(frame)
        self.update()

    def set_selected_frame(self, frame: int | None) -> None:
        self._selected_frame = int(frame) if frame is not None else None
        self.update()

    @property
    def selected_frame(self) -> int | None:
        return self._selected_frame

    # ----- Qt overrides ------------------------------------------------
    def paintEvent(self, _event: object) -> None:        # noqa: N802 — Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect: QRect = self.rect()
        layout = _Layout(
            start_frame=self._range_start,
            end_frame=self._range_end,
            pixel_width=max(1, rect.width()),
        )
        painter.fillRect(rect, _TRACK_BG_COLOR)
        center_y = rect.height() // 2
        for frame in self._frames:
            self._paint_marker(painter, layout, frame, center_y)
        self._paint_playhead(painter, layout, rect, center_y)

    def mousePressEvent(self, event: object) -> None:    # noqa: N802 — Qt
        layout = _Layout(
            start_frame=self._range_start,
            end_frame=self._range_end,
            pixel_width=max(1, self.width()),
        )
        clicked_x = float(event.position().x())
        for frame in self._frames:
            marker_x = layout.frame_to_x(frame)
            if abs(clicked_x - marker_x) <= _MARKER_RADIUS_PX:
                self._selected_frame = frame
                self.update()
                self.keyframe_clicked.emit(int(frame))
                return
        self._selected_frame = None
        self.update()

    # ----- internal ----------------------------------------------------
    def _paint_marker(
        self,
        painter: QPainter,
        layout: _Layout,
        frame: int,
        center_y: int,
    ) -> None:
        x = layout.frame_to_x(frame)
        fill = _SELECTED_FILL if frame == self._selected_frame else _KEYFRAME_FILL
        painter.setPen(QPen(_KEYFRAME_OUTLINE, 1))
        painter.setBrush(fill)
        diamond = QPolygonF(
            [
                QPointF(x, center_y - _MARKER_RADIUS_PX),
                QPointF(x + _MARKER_RADIUS_PX, center_y),
                QPointF(x, center_y + _MARKER_RADIUS_PX),
                QPointF(x - _MARKER_RADIUS_PX, center_y),
            ],
        )
        painter.drawPolygon(diamond)

    def _paint_playhead(
        self,
        painter: QPainter,
        layout: _Layout,
        rect: QRect,
        center_y: int,
    ) -> None:
        del center_y    # unused — playhead spans the whole height
        x = layout.frame_to_x(self._current_frame)
        painter.setPen(QPen(_PLAYHEAD_COLOR, 1, Qt.PenStyle.SolidLine))
        painter.drawLine(int(round(x)), 0, int(round(x)), rect.height())


# Keep ``QPolygon`` reachable through the module so future variants
# (e.g. trapezoid markers for region edits) can swap shape easily.
__all__ = ["KeyframeStripWidget", "QPolygon"]
