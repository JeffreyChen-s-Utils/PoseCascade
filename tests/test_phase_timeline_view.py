"""Tests for the horizontal phase timeline view."""
from __future__ import annotations

import json

import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QMouseEvent


def _doc(*pairs: tuple[str, float]) -> dict:
    return {
        "schema_version": 1,
        "name": "test",
        "loop_sec": sum(d for _, d in pairs),
        "phases": [{"name": n, "duration_sec": d} for n, d in pairs],
    }


@pytest.fixture
def timeline(qapp: object):
    from posecascade.ui.animation_json_document import AnimationJsonDocument  # noqa: PLC0415
    from posecascade.ui.phase_timeline_view import PhaseTimelineView  # noqa: PLC0415

    model = AnimationJsonDocument()
    model.set_text(json.dumps(_doc(("a", 2.0), ("b", 1.0), ("c", 3.0))))
    view = PhaseTimelineView(document=model)
    view.resize(600, 60)
    return view, model


def test_timeline_renders_bar_per_phase(timeline) -> None:
    view, _ = timeline
    rects = view._bar_rects()  # noqa: SLF001
    assert len(rects) == 3
    # Wider phase (c, 3s) should have a wider bar than the narrower one (b, 1s).
    assert rects[2].width() > rects[1].width()


def test_timeline_click_selects_phase(timeline) -> None:
    view, _ = timeline
    received: list[int] = []
    view.phase_selected.connect(received.append)
    rects = view._bar_rects()  # noqa: SLF001
    centre = rects[1].center()
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress, centre,
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.mousePressEvent(event)
    assert view.selected() == 1
    assert received == [1]


def test_timeline_select_method_updates_state(timeline) -> None:
    view, _ = timeline
    view.select(0)
    assert view.selected() == 0
    view.select(None)
    assert view.selected() is None


def test_timeline_resize_handle_detection(timeline) -> None:
    view, _ = timeline
    rects = view._bar_rects()  # noqa: SLF001
    edge = QPoint(rects[0].right() - 1, rects[0].center().y())
    assert view._is_on_resize_handle(0, edge) is True  # noqa: SLF001
    centre = rects[0].center()
    assert view._is_on_resize_handle(0, centre) is False  # noqa: SLF001


def test_timeline_redraws_when_document_changes(timeline) -> None:
    """Adding a phase via the model triggers ``update`` so the bar count changes."""
    view, model = timeline
    model.add_phase({"name": "d", "duration_sec": 0.5})
    rects = view._bar_rects()  # noqa: SLF001
    assert len(rects) == 4
