"""Horizontal phase timeline view for the phase-blocks dock.

Renders each phase as a coloured bar whose width is proportional to
``duration_sec``, laid out left-to-right along a baseline. Designed to
sit ABOVE the list of phase cards so the author can scan the timeline
shape at a glance (relative duration, ordering, total length) without
counting cards in a vertical list.

User interactions:

* Click on a bar → emits :attr:`phase_selected`. The dock wires this
  to the list selection so clicking the timeline highlights the same
  phase in the vertical list and form below.
* Drag a bar horizontally → reorders the phases (the bar lands at the
  drop position; the dock writes the new order through
  :class:`AnimationJsonDocument.move_phase`).
* Drag the right edge of a bar → resizes ``duration_sec`` live; the
  dock writes the new value through
  :class:`AnimationJsonDocument.update_phase_field`.

Drawing is done with raw :class:`QPainter` against a :class:`QWidget`
so we don't pay the cost of a full graphics-scene scene for a feature
that's fundamentally just "a row of bars". A scene would be the right
call once we start showing per-track curves, but that's
:class:`MultiTrackTimelineDock`'s job — this widget is the
phase-level summary.
"""
from __future__ import annotations

import contextlib
from typing import Any

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from posecascade.ui.animation_json_document import AnimationJsonDocument

# Visual constants. Tuned so the timeline stays readable on dark Qt
# themes while keeping a consistent height regardless of phase count.
_BAR_HEIGHT = 36
_BASELINE_MARGIN = 10
_LABEL_FONT_PT = 9
_BACKGROUND_COLOR = QColor(34, 36, 42)
_BAR_COLOURS = (
    QColor(74, 144, 226),    # blue
    QColor(101, 188, 113),   # green
    QColor(220, 137, 90),    # orange
    QColor(170, 132, 220),   # violet
    QColor(217, 92, 96),     # red
    QColor(106, 192, 192),   # teal
    QColor(218, 188, 80),    # gold
)
_BAR_COLOUR_SELECTED_OUTLINE = QColor(255, 255, 255)
_BAR_TEXT_COLOR = QColor(20, 20, 24)
_BAR_RADIUS = 6
_RESIZE_HANDLE_WIDTH = 6
_MIN_DURATION_SEC = 0.05
_MIN_BAR_PIXEL_WIDTH = 24


class PhaseTimelineView(QWidget):
    """Bar-per-phase horizontal timeline + click / drag / resize.

    Signals:
        ``phase_selected(int)`` — emitted with the row index when the
        user clicks a bar.
        ``phase_moved(int, int)`` — ``(source_index, dest_index)`` after
        a horizontal drag that crosses another bar's centre line.
        ``phase_duration_changed(int, float)`` — emitted while the user
        drags the right edge of a bar; the dock writes through to the
        document and we redraw via the ``changed`` signal.
    """

    phase_selected = Signal(int)
    phase_moved = Signal(int, int)
    phase_duration_changed = Signal(int, float)

    def __init__(
        self,
        document: AnimationJsonDocument | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._document = document if document is not None else AnimationJsonDocument(self)
        self._selected_idx: int | None = None
        self._hover_resize_idx: int | None = None
        self._drag_kind: str | None = None     # None | "move" | "resize"
        self._drag_source_idx: int | None = None
        self._drag_offset_px = 0
        self.setMinimumHeight(_BAR_HEIGHT + 2 * _BASELINE_MARGIN)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self._document.changed.connect(self.update)

    # ----- public API ---------------------------------------------------

    def set_document(self, document: AnimationJsonDocument) -> None:
        with contextlib.suppress(TypeError, RuntimeError):
            self._document.changed.disconnect(self.update)
        self._document = document
        self._document.changed.connect(self.update)
        self._selected_idx = None
        self.update()

    def select(self, idx: int | None) -> None:
        """Highlight ``idx`` (or clear when ``None``)."""
        self._selected_idx = idx
        self.update()

    def selected(self) -> int | None:
        return self._selected_idx

    # ----- geometry helpers --------------------------------------------

    def _phase_durations(self) -> list[float]:
        return [
            max(_MIN_DURATION_SEC, float(p.get("duration_sec", 0.0)))
            for p in self._document.phases()
        ]

    def _total_duration(self) -> float:
        return sum(self._phase_durations()) or _MIN_DURATION_SEC

    def _bar_rects(self) -> list[QRect]:
        """Compute each phase's bar rectangle in widget coordinates."""
        durations = self._phase_durations()
        total = self._total_duration()
        rects: list[QRect] = []
        x = 0
        y = _BASELINE_MARGIN
        width = max(1, self.width())
        # Reserve a minimum pixel width per bar so very-short phases
        # (e.g. a 0.1-second accent) still get a clickable target.
        if durations:
            usable = width - _MIN_BAR_PIXEL_WIDTH * len(durations) // 2
            pixel_per_sec = max(1.0, usable / total)
        else:
            pixel_per_sec = 1.0
        for duration in durations:
            bar_width = max(_MIN_BAR_PIXEL_WIDTH, int(duration * pixel_per_sec))
            rects.append(QRect(x, y, bar_width, _BAR_HEIGHT))
            x += bar_width
        return rects

    def _index_at(self, point: QPoint) -> int | None:
        for idx, rect in enumerate(self._bar_rects()):
            if rect.contains(point):
                return idx
        return None

    def _is_on_resize_handle(self, idx: int, point: QPoint) -> bool:
        rects = self._bar_rects()
        if not 0 <= idx < len(rects):
            return False
        rect = rects[idx]
        handle = QRect(
            rect.right() - _RESIZE_HANDLE_WIDTH, rect.top(),
            _RESIZE_HANDLE_WIDTH * 2, rect.height(),
        )
        return handle.contains(point)

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt override
        return QSize(400, _BAR_HEIGHT + 2 * _BASELINE_MARGIN)

    # ----- paint ---------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 — Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(event.rect(), _BACKGROUND_COLOR)
        rects = self._bar_rects()
        phases = self._document.phases()
        font = QFont()
        font.setPointSize(_LABEL_FONT_PT)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        for idx, rect in enumerate(rects):
            phase = phases[idx] if idx < len(phases) else {}
            color = _BAR_COLOURS[idx % len(_BAR_COLOURS)]
            painter.setBrush(color)
            is_selected = idx == self._selected_idx
            pen = (
                QPen(_BAR_COLOUR_SELECTED_OUTLINE) if is_selected
                else QPen(color.darker(140))
            )
            pen.setWidth(2 if is_selected else 1)
            painter.setPen(pen)
            painter.drawRoundedRect(rect, _BAR_RADIUS, _BAR_RADIUS)
            text = _bar_label(phase)
            elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, rect.width() - 8)
            painter.setPen(_BAR_TEXT_COLOR)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, elided)
            # Resize handle marker — a thin tick on the right edge so
            # the user notices the affordance.
            if idx == self._hover_resize_idx:
                painter.setPen(QPen(_BAR_COLOUR_SELECTED_OUTLINE, 2))
                painter.drawLine(
                    rect.right(), rect.top() + 4,
                    rect.right(), rect.bottom() - 4,
                )
        painter.end()

    # ----- mouse handling -----------------------------------------------

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        pos = event.position().toPoint()
        if self._drag_kind == "resize" and self._drag_source_idx is not None:
            self._continue_resize(pos)
        elif self._drag_kind == "move":
            self._continue_move(pos)
        else:
            # Update hover state for resize affordance.
            new_hover = None
            for idx in range(len(self._bar_rects())):
                if self._is_on_resize_handle(idx, pos):
                    new_hover = idx
                    break
            if new_hover != self._hover_resize_idx:
                self._hover_resize_idx = new_hover
                self.setCursor(
                    Qt.CursorShape.SizeHorCursor if new_hover is not None
                    else Qt.CursorShape.ArrowCursor,
                )
                self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        idx = self._index_at(pos)
        if idx is None:
            return
        if self._is_on_resize_handle(idx, pos):
            self._drag_kind = "resize"
            self._drag_source_idx = idx
        else:
            self._drag_kind = "move"
            self._drag_source_idx = idx
            self._drag_offset_px = pos.x() - self._bar_rects()[idx].x()
        self.select(idx)
        self.phase_selected.emit(idx)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._drag_kind == "move" and self._drag_source_idx is not None:
            pos = event.position().toPoint()
            target = self._index_at(pos)
            if target is not None and target != self._drag_source_idx:
                self.phase_moved.emit(self._drag_source_idx, target)
        self._drag_kind = None
        self._drag_source_idx = None

    def _continue_resize(self, pos: QPoint) -> None:
        rects = self._bar_rects()
        if self._drag_source_idx is None or not 0 <= self._drag_source_idx < len(rects):
            return
        rect = rects[self._drag_source_idx]
        total = self._total_duration()
        # Map the new right-edge x to a fractional duration. Reuse the
        # same pixel-per-second factor the rects used to keep the math
        # consistent under varying widget widths.
        width_pixels = max(1, self.width())
        pixel_per_sec = (
            width_pixels - _MIN_BAR_PIXEL_WIDTH * len(rects) // 2
        ) / total if total > 0 else 1.0
        pixel_per_sec = max(pixel_per_sec, 1.0)
        new_width = max(_MIN_BAR_PIXEL_WIDTH, pos.x() - rect.x())
        new_duration = max(_MIN_DURATION_SEC, new_width / pixel_per_sec)
        self.phase_duration_changed.emit(self._drag_source_idx, new_duration)

    def _continue_move(self, pos: QPoint) -> None:
        # We provide live visual feedback by re-selecting whichever bar
        # the mouse is currently over; the actual move only commits on
        # mouseReleaseEvent so a stray drag doesn't reorder twice.
        target = self._index_at(pos)
        if target is not None and target != self._selected_idx:
            self.select(target)


def _bar_label(phase: dict[str, Any]) -> str:
    """Short label painted inside the bar."""
    name = str(phase.get("name", ""))
    duration = float(phase.get("duration_sec", 0.0))
    return f"{name}  {duration:.1f}s" if name else f"{duration:.1f}s"


__all__ = ["PhaseTimelineView"]
