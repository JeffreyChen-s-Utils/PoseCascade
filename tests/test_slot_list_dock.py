"""Qt smoke tests for the slot list dock."""
from __future__ import annotations

from pathlib import Path

import pytest
from pmx.importer import PmxImporter

from posecascade.scene.model_slot import ModelSlot, SceneSlots
from posecascade.ui.slot_list_dock import SlotListDock
from posecascade.utils.math3d import vec3

_TINY_PMX = Path(__file__).resolve().parent / "fixtures" / "mmd" / "tiny.pmx"


def _scene_with_two_slots() -> SceneSlots:
    slots = SceneSlots()
    slots.add(
        ModelSlot(name="character", imported=PmxImporter().load(_TINY_PMX)),
    )
    slots.add(
        ModelSlot(name="stage", imported=PmxImporter().load(_TINY_PMX), visible=False),
    )
    return slots


def test_slot_dock_lists_every_slot(qapp: object) -> None:
    dock = SlotListDock(slots=_scene_with_two_slots())
    assert dock._list.count() == 2     # noqa: SLF001 — test seam


def test_slot_dock_visibility_toggle_emits_signal(qapp: object) -> None:
    slots = _scene_with_two_slots()
    dock = SlotListDock(slots=slots)
    received: list[tuple[str, bool]] = []
    dock.slot_visibility_changed.connect(
        lambda name, visible: received.append((name, visible)),
    )
    # Toggle character (was visible) off.
    dock._rows[0].visibility.setChecked(False)     # noqa: SLF001
    assert received == [("character", False)]
    assert slots.find("character").visible is False


def test_slot_dock_translation_change_writes_to_transform(qapp: object) -> None:
    slots = _scene_with_two_slots()
    dock = SlotListDock(slots=slots)
    spins = dock._rows[0].translation_spins        # noqa: SLF001
    spins[0].setValue(5.0)
    spins[1].setValue(2.0)
    spins[2].setValue(-1.0)
    character = slots.find("character")
    assert character is not None
    assert tuple(float(v) for v in character.transform.translation) == (5.0, 2.0, -1.0)


def test_slot_dock_translation_signal_payload(qapp: object) -> None:
    slots = _scene_with_two_slots()
    dock = SlotListDock(slots=slots)
    received: list[tuple[str, float, float, float]] = []
    dock.slot_translation_changed.connect(
        lambda name, x, y, z: received.append((name, x, y, z)),
    )
    dock._rows[1].translation_spins[1].setValue(3.0)        # noqa: SLF001
    assert received[-1][0] == "stage"
    assert received[-1][2] == pytest.approx(3.0)


def test_slot_dock_refresh_rebuilds_after_slot_added(qapp: object) -> None:
    slots = _scene_with_two_slots()
    dock = SlotListDock(slots=slots)
    slots.add(ModelSlot(name="extra", imported=PmxImporter().load(_TINY_PMX)))
    dock.refresh()
    assert dock._list.count() == 3     # noqa: SLF001
    # Existing slots' transform values are pulled from the live slot data, so
    # editing dock's row before refresh stays preserved across refresh.
    assert dock._rows[0].slot.name == "character"        # noqa: SLF001


# Keep ``vec3`` reachable for IDE jumps.
__all__ = ["pytest", "vec3"]
