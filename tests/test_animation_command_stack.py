"""Tests for the document-snapshot undo / redo stack."""
from __future__ import annotations

import json

import pytest


def _doc(*names: str) -> dict:
    return {
        "schema_version": 1,
        "name": "test",
        "loop_sec": 1.0,
        "phases": [{"name": n, "duration_sec": 1.0} for n in names],
    }


@pytest.fixture
def stack(qapp: object):
    from posecascade.ui.animation_command_stack import AnimationCommandStack  # noqa: PLC0415
    from posecascade.ui.animation_json_document import AnimationJsonDocument  # noqa: PLC0415

    model = AnimationJsonDocument()
    model.set_text(json.dumps(_doc("a", "b")))
    return AnimationCommandStack(model), model


def test_undo_restores_pre_snapshot_state(stack) -> None:
    cmd, model = stack
    cmd.push_snapshot("first edit")
    model.add_phase()
    assert model.phase_count() == 3
    assert cmd.undo() is True
    assert model.phase_count() == 2


def test_redo_reapplies_undone_edit(stack) -> None:
    cmd, model = stack
    cmd.push_snapshot("add")
    model.add_phase()
    cmd.undo()
    assert cmd.redo() is True
    assert model.phase_count() == 3


def test_new_edit_clears_redo_stack(stack) -> None:
    cmd, model = stack
    cmd.push_snapshot("first")
    model.add_phase()
    cmd.undo()
    cmd.push_snapshot("second")
    model.duplicate_phase(0)
    assert cmd.can_redo() is False


def test_transaction_collapses_into_single_undo(stack) -> None:
    cmd, model = stack
    cmd.begin_transaction("batch")
    model.add_phase()
    model.add_phase()
    model.add_phase()
    cmd.end_transaction()
    assert cmd.can_undo() is True
    cmd.undo()
    # One undo undoes the whole transaction.
    assert model.phase_count() == 2


def test_clear_drops_both_stacks(stack) -> None:
    cmd, model = stack
    cmd.push_snapshot("a")
    model.add_phase()
    cmd.undo()
    assert cmd.can_redo() is True
    cmd.clear()
    assert not cmd.can_undo()
    assert not cmd.can_redo()


def test_duplicate_snapshots_collapse(stack) -> None:
    """Pushing the same state twice doesn't accumulate redundant undo steps."""
    cmd, model = stack
    cmd.push_snapshot("first")
    cmd.push_snapshot("second")  # document unchanged → duplicate is dropped
    model.add_phase()
    assert cmd.can_undo() is True
    cmd.undo()
    # Only one undo step covers the gap from no-change back to no-change.
    assert model.phase_count() == 2  # noqa: PLR2004 — matches fixture's seed
    assert cmd.can_undo() is False


def test_undo_through_dock_keeps_other_view_in_sync(qapp: object) -> None:
    """Cross-dock: a phase added through one dock undoes via the shared stack
    and the other dock's view refreshes."""
    from posecascade.ui.animation_command_stack import AnimationCommandStack  # noqa: PLC0415
    from posecascade.ui.animation_json_document import AnimationJsonDocument  # noqa: PLC0415
    from posecascade.ui.phase_blocks_dock import PhaseBlocksDock  # noqa: PLC0415

    model = AnimationJsonDocument()
    model.set_text(json.dumps(_doc("a", "b")))
    cmd = AnimationCommandStack(model)
    dock = PhaseBlocksDock(document=model, command_stack=cmd)
    dock._on_add_clicked()  # noqa: SLF001
    assert dock._list.count() == 3  # noqa: SLF001
    cmd.undo()
    assert dock._list.count() == 2  # noqa: SLF001
