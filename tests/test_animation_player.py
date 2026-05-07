"""Tests for the VMD animation player.

Exercises the bone-track sampler, name-mapping, rest-pose snapshot, and
the player's interaction with the scene graph (writing TRS onto Node
transforms so the renderer picks up the change next frame).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pmx.importer import PmxImporter
from vmd.importer import VmdImporter

from posecascade.animation.player import (
    VmdAnimationPlayer,
    build_skin_lookup,
)
from posecascade.animation.vmd_track import VMD_FRAMES_PER_SECOND, vmd_bone_key

_TINY_PMX = Path(__file__).resolve().parent / "fixtures" / "mmd" / "tiny.pmx"
_WAVE_VMD = Path(__file__).resolve().parent / "fixtures" / "mmd" / "wave.vmd"


# ----- name mapping -----------------------------------------------------
def test_vmd_bone_key_round_trips_japanese() -> None:
    """A 4-character Japanese bone name fits in 15 SJIS bytes intact."""
    assert vmd_bone_key("左腕") == "左腕"
    # Re-encoding confirms the truncation is byte-stable.
    assert vmd_bone_key("左腕").encode("cp932") == "左腕".encode("cp932")


def test_vmd_bone_key_truncates_overlong_ascii() -> None:
    long = "a" * 20
    assert vmd_bone_key(long) == "a" * 15


def test_vmd_bone_key_strips_after_nul() -> None:
    """Some fields embed a stray ``\x00`` mid-name; everything after it is gone."""
    raw = b"a\x00b".decode("cp932")
    assert vmd_bone_key(raw) == "a"


# ----- player core ------------------------------------------------------
def _load_player_for_tiny() -> tuple[VmdAnimationPlayer, object]:
    scene = PmxImporter().load(_TINY_PMX)
    motion = VmdImporter().load(_WAVE_VMD)
    player = VmdAnimationPlayer.for_skin(motion, scene.skins[0])
    return player, scene


def test_player_lookup_covers_skin_joints() -> None:
    scene = PmxImporter().load(_TINY_PMX)
    lookup = build_skin_lookup(scene.skins[0])
    assert set(lookup.keys()) == {"root", "child"}


def test_player_at_keyframe_gives_keyframe_value() -> None:
    player, scene = _load_player_for_tiny()
    child = scene.skins[0].joints[1]
    player.apply(5.0 / VMD_FRAMES_PER_SECOND)
    np.testing.assert_allclose(
        child.transform.rotation,
        [0.3826834, 0.0, 0.0, 0.9238795],
        atol=1e-5,
    )


def test_player_eases_between_keyframes() -> None:
    """At frame 2.5 (linear bezier between identity and quarter-X-turn) the
    rotation must be the eighth turn — i.e., halfway through the slerp."""
    player, scene = _load_player_for_tiny()
    child = scene.skins[0].joints[1]
    player.apply(2.5 / VMD_FRAMES_PER_SECOND)
    expected_x = float(np.sin(np.deg2rad(11.25)))
    expected_w = float(np.cos(np.deg2rad(11.25)))
    np.testing.assert_allclose(
        child.transform.rotation,
        [expected_x, 0.0, 0.0, expected_w],
        atol=1e-4,
    )


def test_player_clamps_below_first_keyframe() -> None:
    player, scene = _load_player_for_tiny()
    child = scene.skins[0].joints[1]
    player.apply(-5.0)
    # First keyframe is identity — rest-translation + zero offset.
    np.testing.assert_allclose(child.transform.rotation, [0.0, 0.0, 0.0, 1.0])


def test_player_clamps_above_last_keyframe() -> None:
    player, scene = _load_player_for_tiny()
    child = scene.skins[0].joints[1]
    # Frame 20 is identity — same value clamps for any time past it.
    player.apply(60.0)
    np.testing.assert_allclose(child.transform.rotation, [0.0, 0.0, 0.0, 1.0])


def test_reset_to_rest_undoes_apply() -> None:
    player, scene = _load_player_for_tiny()
    child = scene.skins[0].joints[1]
    rest_translation = child.transform.translation.copy()
    player.apply(5.0 / VMD_FRAMES_PER_SECOND)
    assert child.transform.rotation[0] != 0.0
    player.reset_to_rest()
    np.testing.assert_allclose(child.transform.rotation, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(child.transform.translation, rest_translation)


def test_player_does_not_disturb_unmapped_bones() -> None:
    """The wave VMD only animates ``child``; ``root`` must keep its rest TRS."""
    player, scene = _load_player_for_tiny()
    root = scene.skins[0].joints[0]
    rest_translation = root.transform.translation.copy()
    rest_rotation = root.transform.rotation.copy()
    player.apply(5.0 / VMD_FRAMES_PER_SECOND)
    np.testing.assert_allclose(root.transform.translation, rest_translation)
    np.testing.assert_allclose(root.transform.rotation, rest_rotation)


def test_duration_seconds_matches_last_frame() -> None:
    player, _ = _load_player_for_tiny()
    assert player.duration_seconds == pytest.approx(20.0 / VMD_FRAMES_PER_SECOND)
