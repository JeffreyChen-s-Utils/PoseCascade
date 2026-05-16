"""Tests for the shared animation document model + phase-blocks dock (MVP-2)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from posecascade.ui.animation_json_document import AnimationJsonDocument


def _doc(*phase_names: str) -> dict:
    """Build a parseable minimal animation document with the given phase names."""
    return {
        "schema_version": 1,
        "name": "test",
        "loop_sec": 4.0,
        "phases": [{"name": n, "duration_sec": 1.0} for n in phase_names],
    }


# ---------------------------------------------------------------------------
# AnimationJsonDocument — pure model, no Qt UI tests needed.
# ---------------------------------------------------------------------------


def test_document_load_file_replaces_state(qapp: object, tmp_path: Path) -> None:
    """``load_file`` reads the JSON and emits ``changed`` once."""
    model = AnimationJsonDocument()
    counter: list[int] = []
    model.changed.connect(lambda: counter.append(1))
    path = tmp_path / "a.json"
    path.write_text(json.dumps(_doc("a", "b")), encoding="utf-8")
    assert model.load_file(path) is True
    assert model.path == path
    assert model.phase_count() == 2
    assert counter == [1]


def test_document_set_text_rejects_malformed_json(qapp: object) -> None:
    """A bad JSON string returns ``False`` and leaves the model untouched."""
    model = AnimationJsonDocument()
    model.set_text(json.dumps(_doc("kept")))
    received: list[int] = []
    model.changed.connect(lambda: received.append(1))
    assert model.set_text("{not json") is False
    assert model.phase_count() == 1  # unchanged
    assert received == []


def test_document_add_phase_appends_minimal_placeholder(qapp: object) -> None:
    """``add_phase(None)`` inserts a placeholder with a unique name."""
    model = AnimationJsonDocument()
    model.set_text(json.dumps(_doc("phase_1")))
    new_idx = model.add_phase()
    assert new_idx == 1
    assert model.phases()[1]["name"] == "phase_2"


def test_document_duplicate_phase_clones_with_suffix(qapp: object) -> None:
    """Duplicate inserts immediately after the source and tags ``_copy``."""
    model = AnimationJsonDocument()
    model.set_text(json.dumps(_doc("a", "b")))
    new_idx = model.duplicate_phase(0)
    assert new_idx == 1
    assert [p["name"] for p in model.phases()] == ["a", "a_copy", "b"]


def test_document_move_phase_reorders(qapp: object) -> None:
    """``move_phase(0, 2)`` lifts the first phase to the end of a 3-list."""
    model = AnimationJsonDocument()
    model.set_text(json.dumps(_doc("a", "b", "c")))
    assert model.move_phase(0, 2) is True
    assert [p["name"] for p in model.phases()] == ["b", "c", "a"]


def test_document_remove_phase(qapp: object) -> None:
    model = AnimationJsonDocument()
    model.set_text(json.dumps(_doc("a", "b", "c")))
    assert model.remove_phase(1) is True
    assert [p["name"] for p in model.phases()] == ["a", "c"]


def test_document_update_phase_field_writes_and_clears(qapp: object) -> None:
    """``update_phase_field`` writes a value; ``None`` deletes optional keys."""
    model = AnimationJsonDocument()
    model.set_text(json.dumps(_doc("a")))
    assert model.update_phase_field(0, "pose", "v_arms_up") is True
    assert model.phases()[0]["pose"] == "v_arms_up"
    # Clear by passing None — optional field disappears.
    assert model.update_phase_field(0, "pose", None) is True
    assert "pose" not in model.phases()[0]
    # Required fields refuse a None clear.
    assert model.update_phase_field(0, "name", None) is False
    assert model.phases()[0]["name"] == "a"


def test_document_text_round_trips(qapp: object) -> None:
    """``text → set_text`` round-trip preserves the phase list."""
    model = AnimationJsonDocument()
    model.set_text(json.dumps(_doc("a", "b")))
    text = model.text()
    other = AnimationJsonDocument()
    assert other.set_text(text) is True
    assert [p["name"] for p in other.phases()] == ["a", "b"]


# ---------------------------------------------------------------------------
# PhaseBlocksDock — Qt smoke tests through the dock's public surface.
# ---------------------------------------------------------------------------


@pytest.fixture
def dock(qapp: object):
    from posecascade.ui.phase_blocks_dock import PhaseBlocksDock  # noqa: PLC0415

    model = AnimationJsonDocument()
    model.set_text(json.dumps(_doc("intro", "main", "outro")))
    return PhaseBlocksDock(document=model)


def test_dock_lists_each_phase(dock) -> None:
    """The list widget renders one row per phase, summarised inline."""
    assert dock._list.count() == 3  # noqa: SLF001
    first_text = dock._list.item(0).text()  # noqa: SLF001
    assert "intro" in first_text
    assert "1.0s" in first_text


def test_dock_add_clicked_appends_and_selects(dock) -> None:
    dock._on_add_clicked()  # noqa: SLF001
    assert dock._list.count() == 4  # noqa: SLF001
    assert dock._list.currentRow() == 3  # noqa: SLF001


def test_dock_duplicate_clicked_inserts_after_source(dock) -> None:
    dock._list.setCurrentRow(0)  # noqa: SLF001
    dock._on_dup_clicked()  # noqa: SLF001
    names = [p["name"] for p in dock.document.phases()]
    assert names == ["intro", "intro_copy", "main", "outro"]


def test_dock_delete_clicked_removes_selected(dock) -> None:
    dock._list.setCurrentRow(1)  # noqa: SLF001
    dock._on_del_clicked()  # noqa: SLF001
    names = [p["name"] for p in dock.document.phases()]
    assert names == ["intro", "outro"]


def test_dock_form_writes_name_change(dock) -> None:
    """Editing the form's name field updates the underlying document."""
    dock._list.setCurrentRow(1)  # noqa: SLF001
    dock._form._name_edit.setText("renamed")  # noqa: SLF001
    dock._form._on_name_changed()  # noqa: SLF001
    assert dock.document.phases()[1]["name"] == "renamed"


def test_dock_form_writes_pose_change(dock) -> None:
    dock._list.setCurrentRow(0)  # noqa: SLF001
    dock._form._pose_combo.setCurrentText("v_arms_up")  # noqa: SLF001
    assert dock.document.phases()[0]["pose"] == "v_arms_up"


def test_dock_form_writes_body_yaw_into_nested_dict(dock) -> None:
    """Body field edits create the ``body`` sub-dict if absent."""
    dock._list.setCurrentRow(2)  # noqa: SLF001
    dock._form._yaw_spin.setValue(1.5)  # noqa: SLF001
    body = dock.document.phases()[2]["body"]
    assert body["yaw_rad"] == 1.5


def test_dock_reload_signal_emits_serialised_text(dock) -> None:
    received: list[str] = []
    dock.reload_requested.connect(received.append)
    dock._on_reload_clicked()  # noqa: SLF001
    assert len(received) == 1
    assert '"intro"' in received[0]


# ---------------------------------------------------------------------------
# MVP-3 — JSON dock + blocks dock share one document; edits on either side
# propagate to the other through the model's ``changed`` signal.
# ---------------------------------------------------------------------------


def test_dual_view_blocks_edit_updates_json_pane(qapp: object) -> None:
    """Editing a phase via blocks dock refreshes the JSON dock's text."""
    from posecascade.ui.animation_json_dock import AnimationJsonDock  # noqa: PLC0415
    from posecascade.ui.phase_blocks_dock import PhaseBlocksDock  # noqa: PLC0415

    model = AnimationJsonDocument()
    model.set_text(json.dumps(_doc("intro", "main")))
    json_dock = AnimationJsonDock(document=model)
    blocks_dock = PhaseBlocksDock(document=model)
    # Edit through the blocks dock; the JSON dock should see the new
    # text on the next ``changed`` emission.
    blocks_dock._list.setCurrentRow(0)  # noqa: SLF001
    blocks_dock._form._name_edit.setText("intro_renamed")  # noqa: SLF001
    blocks_dock._form._on_name_changed()  # noqa: SLF001
    assert "intro_renamed" in json_dock.current_text()


def test_dual_view_json_edit_updates_blocks_pane(qapp: object) -> None:
    """Editing the JSON text refreshes the blocks dock's list rows."""
    from posecascade.ui.animation_json_dock import AnimationJsonDock  # noqa: PLC0415
    from posecascade.ui.phase_blocks_dock import PhaseBlocksDock  # noqa: PLC0415

    model = AnimationJsonDocument()
    model.set_text(json.dumps(_doc("a")))
    json_dock = AnimationJsonDock(document=model)
    blocks_dock = PhaseBlocksDock(document=model)
    new_doc = _doc("alpha", "beta", "gamma")
    json_dock._editor.setPlainText(json.dumps(new_doc))  # noqa: SLF001
    # ``textChanged`` → ``set_text`` → ``changed`` → blocks dock rebuilds.
    assert blocks_dock._list.count() == 3  # noqa: SLF001
    assert "alpha" in blocks_dock._list.item(0).text()  # noqa: SLF001
