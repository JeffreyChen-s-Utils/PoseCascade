"""Tests for :class:`SlotsPlayer` — multi-slot animation driver."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pmx.importer import PmxImporter
from vmd.importer import VmdImporter

from posecascade.animation.slots_player import SlotsPlayer
from posecascade.animation.vmd_track import VMD_FRAMES_PER_SECOND
from posecascade.scene.external_parent import ExternalParentBinding
from posecascade.scene.model_slot import ModelSlot, SceneSlots

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mmd"
_TINY_PMX = _FIXTURES / "tiny.pmx"
_WAVE_VMD = _FIXTURES / "wave.vmd"


def _slot_with_motion(name: str) -> ModelSlot:
    return ModelSlot(
        name=name,
        imported=PmxImporter().load(_TINY_PMX),
        motion=VmdImporter().load(_WAVE_VMD),
    )


def _slot_no_motion(
    name: str, *, bindings: tuple[ExternalParentBinding, ...] = (),
) -> ModelSlot:
    return ModelSlot(
        name=name,
        imported=PmxImporter().load(_TINY_PMX),
        external_parents=bindings,
    )


def test_slots_player_drives_each_slot_independently() -> None:
    """Two slots with their own motion should each see their bones move."""
    slots = SceneSlots()
    slot_a = _slot_with_motion("alice")
    slot_b = _slot_with_motion("bob")
    slots.add(slot_a)
    slots.add(slot_b)
    player = SlotsPlayer(slots=slots)
    rest_a = slot_a.imported.skins[0].joints[1].transform.rotation.copy()
    rest_b = slot_b.imported.skins[0].joints[1].transform.rotation.copy()
    player.apply(5.0 / VMD_FRAMES_PER_SECOND)
    moved_a = slot_a.imported.skins[0].joints[1].transform.rotation
    moved_b = slot_b.imported.skins[0].joints[1].transform.rotation
    assert not np.allclose(moved_a, rest_a)
    assert not np.allclose(moved_b, rest_b)


def test_slots_player_skips_slot_without_motion() -> None:
    """A slot with no motion stays at rest even when others advance."""
    slots = SceneSlots()
    moved_slot = _slot_with_motion("animated")
    static_slot = _slot_no_motion("static")
    slots.add(moved_slot)
    slots.add(static_slot)
    player = SlotsPlayer(slots=slots)
    rest = static_slot.imported.skins[0].joints[1].transform.rotation.copy()
    player.apply(5.0 / VMD_FRAMES_PER_SECOND)
    np.testing.assert_allclose(
        static_slot.imported.skins[0].joints[1].transform.rotation, rest, atol=1e-6,
    )


def test_slots_player_resolves_external_parent_after_per_slot_animation() -> None:
    """Animation drives slot A; slot B's bone follows slot A's bone after."""
    slots = SceneSlots()
    source = _slot_with_motion("source")
    follower = _slot_no_motion(
        "follower",
        bindings=(
            ExternalParentBinding(
                self_bone_name="root",
                target_slot_name="source",
                target_bone_name="child",
            ),
        ),
    )
    slots.add(source)
    slots.add(follower)
    player = SlotsPlayer(slots=slots)
    player.apply(5.0 / VMD_FRAMES_PER_SECOND)
    target_rotation = source.imported.skins[0].joints[1].transform.rotation
    follower_rotation = follower.imported.skins[0].joints[0].transform.rotation
    np.testing.assert_allclose(follower_rotation, target_rotation, atol=1e-5)


def test_slots_player_player_for_returns_underlying_player() -> None:
    """Debugging seam — the integrator can grab the per-slot player to
    inspect VmdMotionAsset state without re-creating it."""
    slots = SceneSlots()
    slots.add(_slot_with_motion("alice"))
    slots.add(_slot_no_motion("static"))
    player = SlotsPlayer(slots=slots)
    assert player.player_for("alice") is not None
    assert player.player_for("static") is None
    assert player.player_for("unknown") is None
