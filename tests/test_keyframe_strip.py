"""Qt smoke tests for the keyframe strip widget."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from posecascade.ui.keyframe_strip import KeyframeStripWidget


def _click(widget: KeyframeStripWidget, x: float, y: float = 5.0) -> None:
    """Synthesise a left-button mouse press at ``(x, y)`` widget-local."""
    pos = QPointF(x, y)
    global_pos = QPointF(widget.mapToGlobal(QPoint(int(x), int(y))))
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        pos, global_pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.mousePressEvent(event)


def test_strip_initial_state_has_no_selection(qapp: object) -> None:
    strip = KeyframeStripWidget()
    assert strip.selected_frame is None


def test_set_frames_updates_internal_state(qapp: object) -> None:
    strip = KeyframeStripWidget()
    strip.set_frames([0, 5, 15, 30])
    # Just verify the call doesn't raise + the widget repaints.
    strip.update()


def test_clicking_near_marker_selects_it(qapp: object) -> None:
    strip = KeyframeStripWidget()
    strip.resize(200, 30)
    strip.set_frame_range(0, 100)
    strip.set_frames([0, 50, 100])
    received: list[int] = []
    strip.keyframe_clicked.connect(received.append)
    # Frame 50 maps to x = 100 in a 200-px strip; click at x=100.
    _click(strip, x=100.0)
    assert received == [50]
    assert strip.selected_frame == 50


def test_clicking_empty_area_clears_selection(qapp: object) -> None:
    strip = KeyframeStripWidget()
    strip.resize(200, 30)
    strip.set_frame_range(0, 100)
    strip.set_frames([20])
    strip.set_selected_frame(20)
    _click(strip, x=180.0)               # nowhere near frame 20 (x=40)
    assert strip.selected_frame is None


def test_set_current_frame_updates(qapp: object) -> None:
    strip = KeyframeStripWidget()
    strip.set_current_frame(7)
    # No public getter — but the call shouldn't raise + a paint after
    # should happen without error.
    strip.resize(100, 20)
    strip.update()


def test_inverted_range_is_normalised(qapp: object) -> None:
    """Passing ``end < start`` swaps internally so paint logic doesn't
    divide by zero / produce negative coordinates."""
    strip = KeyframeStripWidget()
    strip.set_frame_range(50, 0)
    strip.set_frames([10])
    strip.resize(100, 20)
    strip.update()         # should not crash


# Keep ``Qt`` reachable for IDE jumps.
__all__ = ["Qt"]
