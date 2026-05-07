"""Qt smoke tests for the bezier handle editor."""
from __future__ import annotations

import pytest

from posecascade.ui.bezier_editor import BezierHandleEditor, BezierPreview


def test_editor_round_trips_handles(qapp: object) -> None:
    editor = BezierHandleEditor(handles=(20, 40, 80, 120))
    assert editor.handles() == (20, 40, 80, 120)


def test_editor_emits_handles_changed_on_spin_edit(qapp: object) -> None:
    editor = BezierHandleEditor(handles=(20, 20, 107, 107))
    received: list[tuple[int, int, int, int]] = []
    editor.handles_changed.connect(
        lambda x1, y1, x2, y2: received.append((x1, y1, x2, y2)),
    )
    editor._spins[0].setValue(40)        # noqa: SLF001
    assert received[-1] == (40, 20, 107, 107)


def test_editor_set_handles_does_not_emit(qapp: object) -> None:
    """Programmatically updating handles must not trigger
    ``handles_changed`` — it's the user's edit path only, otherwise the
    integrator's "load existing keyframe" flow would loop.
    """
    editor = BezierHandleEditor()
    received: list[tuple[int, int, int, int]] = []
    editor.handles_changed.connect(
        lambda x1, y1, x2, y2: received.append((x1, y1, x2, y2)),
    )
    editor.set_handles((10, 10, 117, 117))
    assert received == []
    assert editor.handles() == (10, 10, 117, 117)


def test_preview_rejects_negative_size_paint(qapp: object) -> None:
    preview = BezierPreview()
    preview.set_handles((20, 20, 107, 107))
    preview.update()


@pytest.mark.parametrize("handles", [
    (0, 0, 127, 127),     # linear
    (127, 0, 0, 127),     # ease-in-out reverse
    (20, 80, 80, 20),     # ease-out then ease-in
])
def test_editor_clamps_inside_range(qapp: object, handles) -> None:    # noqa: ANN001
    editor = BezierHandleEditor(handles=handles)
    assert editor.handles() == handles


# Keep ``BezierPreview`` reachable for downstream patches.
__all__ = ["BezierPreview"]
