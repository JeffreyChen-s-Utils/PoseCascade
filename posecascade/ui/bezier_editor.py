"""Bezier handle editor — 4 spin boxes + a live curve preview.

Inputs are the four ``(X1, Y1, X2, Y2)`` control values in VMD's native
``[0, 127]`` range. The widget paints a 96×96 preview pulled from
:func:`posecascade.animation.vmd_curves.evaluate_bezier` so the user
can see the curve update in real time as they drag the spin boxes.

Selection happens externally — :class:`MultiTrackTimelineDock` clicks
a :class:`KeyframeStripWidget` marker, then opens this editor to tune
that keyframe's bezier. The widget emits ``handles_changed(...)`` so
the integrator can write the new tuple back through a Command.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from posecascade.animation.vmd_curves import evaluate_bezier

_BEZIER_RANGE_MAX = 127
_PREVIEW_SIZE_PX = 96
_PREVIEW_SAMPLES = 32
_PREVIEW_BG = QColor(28, 28, 32)
_PREVIEW_GRID = QColor(60, 60, 65)
_PREVIEW_CURVE = QColor(255, 200, 80)


class BezierPreview(QWidget):
    """Read-only painter — draws the cubic curve from a handle tuple."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._handles: tuple[int, int, int, int] = (20, 20, 107, 107)
        self.setFixedSize(_PREVIEW_SIZE_PX, _PREVIEW_SIZE_PX)

    def set_handles(self, handles: tuple[int, int, int, int]) -> None:
        self._handles = handles
        self.update()

    def paintEvent(self, _event: object) -> None:    # noqa: N802 — Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = _PREVIEW_SIZE_PX
        painter.fillRect(0, 0, size, size, _PREVIEW_BG)
        painter.setPen(QPen(_PREVIEW_GRID, 1, Qt.PenStyle.DashLine))
        painter.drawLine(0, size - 1, size - 1, 0)             # diagonal reference
        painter.setPen(QPen(_PREVIEW_CURVE, 2, Qt.PenStyle.SolidLine))
        polyline = QPolygonF()
        for i in range(_PREVIEW_SAMPLES + 1):
            t = i / _PREVIEW_SAMPLES
            eased = evaluate_bezier(self._handles, t)
            x = t * (size - 1)
            y = (1.0 - eased) * (size - 1)
            polyline.append(QPointF(x, y))
        painter.drawPolyline(polyline)


class BezierHandleEditor(QWidget):
    """Inline editor with four spin boxes + a live curve preview."""

    handles_changed = Signal(int, int, int, int)

    def __init__(
        self,
        handles: tuple[int, int, int, int] = (20, 20, 107, 107),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._spins: list[QSpinBox] = []
        self._preview = BezierPreview()
        self._build_ui()
        self.set_handles(handles)

    def set_handles(self, handles: tuple[int, int, int, int]) -> None:
        for spin, value in zip(self._spins, handles, strict=True):
            spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(False)
        self._preview.set_handles(handles)

    def handles(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = (int(spin.value()) for spin in self._spins)
        return x1, y1, x2, y2

    # ----- internal ----------------------------------------------------
    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        form_host = QWidget()
        form = QFormLayout(form_host)
        # MMD-style cubic Bezier: two interior handles (X1,Y1) and
        # (X2,Y2) shape the in / out tangent. 0 / 127 = endpoints
        # (linear); raise X to delay, raise Y to overshoot.
        handle_tooltips = {
            "X1": "First handle X (0–127). Lower = faster start; higher = ease-in.",
            "Y1": "First handle Y (0–127). Higher = overshoot at the start.",
            "X2": "Second handle X (0–127). Lower = ease-out; higher = late-arriving end.",
            "Y2": "Second handle Y (0–127). Lower = undershoot near the end.",
        }
        for label in ("X1", "Y1", "X2", "Y2"):
            spin = QSpinBox()
            spin.setRange(0, _BEZIER_RANGE_MAX)
            spin.setToolTip(handle_tooltips[label])
            spin.valueChanged.connect(self._on_spin_changed)
            self._spins.append(spin)
            form.addRow(label, spin)
        layout.addWidget(form_host)
        preview_host = QWidget()
        preview_layout = QVBoxLayout(preview_host)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self._preview)
        preview_layout.addStretch(1)
        layout.addWidget(preview_host)

    def _on_spin_changed(self, _value: int) -> None:
        x1, y1, x2, y2 = self.handles()
        self._preview.set_handles((x1, y1, x2, y2))
        self.handles_changed.emit(x1, y1, x2, y2)
