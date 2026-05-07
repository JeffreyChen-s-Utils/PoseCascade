"""Tests for the multi-slot scene container + cross-slot bone-follow."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pmx.importer import PmxImporter

from posecascade.scene.external_parent import (
    ExternalParentBinding,
    apply_external_parents,
)
from posecascade.scene.model_slot import ModelSlot, SceneSlots
from posecascade.utils.math3d import vec3

_TINY_PMX = Path(__file__).resolve().parent / "fixtures" / "mmd" / "tiny.pmx"


def _make_slot(name: str, *, bindings: tuple[ExternalParentBinding, ...] = ()) -> ModelSlot:
    return ModelSlot(
        name=name,
        imported=PmxImporter().load(_TINY_PMX),
        external_parents=bindings,
    )


# ----- SceneSlots container -------------------------------------------
def test_scene_slots_add_remove_find() -> None:
    slots = SceneSlots()
    slot_a = _make_slot("character")
    slot_b = _make_slot("stage")
    slots.add(slot_a)
    slots.add(slot_b)
    assert len(slots) == 2
    assert slots.find("character") is slot_a
    assert slots.find("stage") is slot_b
    assert slots.find("missing") is None
    removed = slots.remove("character")
    assert removed is slot_a
    assert slots.find("character") is None


def test_scene_slots_rejects_duplicate_name() -> None:
    slots = SceneSlots()
    slots.add(_make_slot("dup"))
    with pytest.raises(ValueError, match="already registered"):
        slots.add(_make_slot("dup"))


def test_scene_slots_iterable_in_insertion_order() -> None:
    slots = SceneSlots()
    slots.add(_make_slot("a"))
    slots.add(_make_slot("b"))
    slots.add(_make_slot("c"))
    assert [slot.name for slot in slots] == ["a", "b", "c"]


def test_scene_slots_truthy_when_non_empty() -> None:
    slots = SceneSlots()
    assert not slots
    slots.add(_make_slot("x"))
    assert slots


# ----- external parent ------------------------------------------------
def test_external_parent_copies_target_world_pose() -> None:
    """An accessory slot's bone follows the source slot's bone exactly."""
    source_slot = _make_slot("character")
    accessory_slot = _make_slot(
        "microphone",
        bindings=(
            ExternalParentBinding(
                self_bone_name="root",
                target_slot_name="character",
                target_bone_name="child",
            ),
        ),
    )
    slots = SceneSlots()
    slots.add(source_slot)
    slots.add(accessory_slot)
    # Move the source's "child" bone — the accessory's "root" should
    # snap to the same translation.
    source_slot.imported.skins[0].joints[1].transform.set_translation(vec3(2.0, 1.0, 0.0))
    apply_external_parents(slots, slots.find)
    accessory_root = accessory_slot.imported.skins[0].joints[0]
    np.testing.assert_allclose(
        accessory_root.transform.translation, [2.0, 1.0, 0.0], atol=1e-5,
    )


def test_external_parent_skips_missing_target_slot() -> None:
    """Binding to an unknown slot must not crash."""
    accessory = _make_slot(
        "ghost_holder",
        bindings=(
            ExternalParentBinding(
                self_bone_name="root",
                target_slot_name="never_loaded",
                target_bone_name="anything",
            ),
        ),
    )
    slots = SceneSlots()
    slots.add(accessory)
    # Should silently skip — no exception, no mutation.
    apply_external_parents(slots, slots.find)


def test_external_parent_skips_missing_bones() -> None:
    """Binding referencing bones that don't exist is also a no-op."""
    source = _make_slot("source")
    accessory = _make_slot(
        "follower",
        bindings=(
            ExternalParentBinding(
                self_bone_name="not_a_bone",
                target_slot_name="source",
                target_bone_name="not_a_bone_either",
            ),
        ),
    )
    slots = SceneSlots()
    slots.add(source)
    slots.add(accessory)
    apply_external_parents(slots, slots.find)
    # Accessory bones stayed at rest (no overwrite happened).
    np.testing.assert_allclose(
        accessory.imported.skins[0].joints[0].transform.translation, [0, 0, 0], atol=1e-5,
    )


def test_external_parent_skips_self_binding() -> None:
    """A binding pointing at the slot's *own* name is treated as a no-op
    (the resolver is single-pass; a self-binding would otherwise loop
    once and read garbage from the in-progress edit)."""
    slot = _make_slot(
        "loop",
        bindings=(
            ExternalParentBinding(
                self_bone_name="child",
                target_slot_name="loop",
                target_bone_name="root",
            ),
        ),
    )
    slots = SceneSlots()
    slots.add(slot)
    slot.imported.skins[0].joints[0].transform.set_translation(vec3(5, 0, 0))
    original_child = slot.imported.skins[0].joints[1].transform.translation.copy()
    apply_external_parents(slots, slots.find)
    np.testing.assert_allclose(
        slot.imported.skins[0].joints[1].transform.translation, original_child, atol=1e-5,
    )


def test_external_parent_chain_resolves_in_order() -> None:
    """A → B (A follows B), B → C (B follows C). After one pass, both A
    and B should reflect C's state."""
    slot_a = _make_slot(
        "a",
        bindings=(
            ExternalParentBinding(
                self_bone_name="root",
                target_slot_name="b",
                target_bone_name="root",
            ),
        ),
    )
    slot_b = _make_slot(
        "b",
        bindings=(
            ExternalParentBinding(
                self_bone_name="root",
                target_slot_name="c",
                target_bone_name="root",
            ),
        ),
    )
    slot_c = _make_slot("c")
    slot_c.imported.skins[0].joints[0].transform.set_translation(vec3(7, 0, 0))
    slots = SceneSlots()
    # Order: c first, then b, then a — so the resolver visits each
    # binding's target after that target's own bindings have already
    # snapped into place.
    slots.add(slot_c)
    slots.add(slot_b)
    slots.add(slot_a)
    apply_external_parents(slots, slots.find)
    np.testing.assert_allclose(
        slot_b.imported.skins[0].joints[0].transform.translation, [7, 0, 0], atol=1e-5,
    )
    np.testing.assert_allclose(
        slot_a.imported.skins[0].joints[0].transform.translation, [7, 0, 0], atol=1e-5,
    )
