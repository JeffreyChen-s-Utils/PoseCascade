"""Tests for the humanoid bone-alias auto-detector.

Covers:

- Pattern matching across VRoid, March 7th FBX, and MMD Japanese
  naming conventions (and a generic FBX / Mixamo case).
- :class:`Scene` ``find`` precedence — aliases take priority but
  literal name lookup still works as a fallback.
- ``ImporterManager`` populates ``scene.bone_aliases`` so script
  authors don't have to call the detector themselves.
- No false positives: a rig that doesn't match any canonical pattern
  produces an empty alias dict rather than wrong matches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from posecascade.animation.bone_aliasing import (
    canonical_keys,
    detect_humanoid_aliases,
)
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene


def _nodes(*names: str) -> list[Node]:
    """Build a flat list of nodes from joint names — order matches the importer."""
    return [Node(name=name) for name in names]


def test_canonical_keys_includes_core_humanoid_landmarks() -> None:
    """Every script in ``examples/scripts/`` keys off this canonical list."""
    keys = canonical_keys()
    for required in (
        "head", "chest", "hip",
        "upper_arm_L", "upper_arm_R",
        "lower_arm_L", "lower_arm_R",
        "upper_leg_L", "upper_leg_R",
        "lower_leg_L", "lower_leg_R",
        "foot_L", "foot_R",
    ):
        assert required in keys, f"canonical key {required!r} missing"


def test_detect_vroid_rig() -> None:
    """VRoid / VRM J_Bip_* names resolve to canonical keys."""
    joints = _nodes(
        "J_Bip_C_Hips", "J_Bip_C_Spine", "J_Bip_C_Chest", "J_Bip_C_UpperChest",
        "J_Bip_C_Neck", "J_Bip_C_Head",
        "J_Bip_L_Shoulder", "J_Bip_L_UpperArm", "J_Bip_L_LowerArm", "J_Bip_L_Hand",
        "J_Bip_R_Shoulder", "J_Bip_R_UpperArm", "J_Bip_R_LowerArm", "J_Bip_R_Hand",
        "J_Bip_L_UpperLeg", "J_Bip_L_LowerLeg", "J_Bip_L_Foot", "J_Bip_L_ToeBase",
        "J_Bip_R_UpperLeg", "J_Bip_R_LowerLeg", "J_Bip_R_Foot", "J_Bip_R_ToeBase",
    )
    aliases = detect_humanoid_aliases(joints)
    assert aliases["head"].name == "J_Bip_C_Head"
    # UpperChest is preferred over Chest because patterns are ordered
    # by specificity (UpperChest matches before Chest in the chest list).
    assert aliases["chest"].name == "J_Bip_C_UpperChest"
    assert aliases["hip"].name == "J_Bip_C_Hips"
    assert aliases["upper_arm_L"].name == "J_Bip_L_UpperArm"
    assert aliases["upper_arm_R"].name == "J_Bip_R_UpperArm"
    assert aliases["foot_L"].name == "J_Bip_L_Foot"
    assert aliases["foot_R"].name == "J_Bip_R_Foot"
    assert aliases["toe_L"].name == "J_Bip_L_ToeBase"


def test_detect_march7th_fbx_rig() -> None:
    """HoYoverse / Aplaybox FBX names (Head_M_055, Shoulder_L_0183) resolve."""
    joints = _nodes(
        "_rootJoint", "Main_00", "Root_M_01",
        "Hip_L_02", "Knee_L_04", "Ankle_L_05", "Toes_L_06",
        "Hip_R_07", "Knee_R_09", "Ankle_R_010", "Toes_R_011",
        "Spine1_M_037", "Spine2_M_044", "Chest_M_049", "Neck_M_054", "Head_M_055",
        "Shoulder_L_0183", "Elbow_L_0184", "Wrist_L_0177",
        "Shoulder_R_0233", "Elbow_R_0234", "Wrist_R_0247",
    )
    aliases = detect_humanoid_aliases(joints)
    assert aliases["head"].name == "Head_M_055"
    assert aliases["chest"].name == "Chest_M_049"
    # March 7th has no central hip; Root_M_01 acts as the equivalent.
    assert aliases["hip"].name == "Root_M_01"
    assert aliases["upper_arm_L"].name == "Shoulder_L_0183"
    assert aliases["upper_arm_R"].name == "Shoulder_R_0233"
    assert aliases["lower_arm_L"].name == "Elbow_L_0184"
    assert aliases["upper_leg_L"].name == "Hip_L_02"
    assert aliases["lower_leg_L"].name == "Knee_L_04"
    assert aliases["foot_L"].name == "Ankle_L_05"
    assert aliases["toe_L"].name == "Toes_L_06"


def test_detect_mmd_japanese_rig() -> None:
    """PMX / PMD Japanese bone names resolve to canonical keys."""
    joints = _nodes(
        "センター", "上半身", "上半身2", "首", "頭",
        "左腕", "左ひじ", "左手首",
        "右腕", "右ひじ", "右手首",
        "左足", "左ひざ", "左足首", "左つま先",
        "右足", "右ひざ", "右足首", "右つま先",
    )
    aliases = detect_humanoid_aliases(joints)
    assert aliases["head"].name == "頭"
    assert aliases["neck"].name == "首"
    # 上半身2 is the upper chest; pattern ordering ensures it's picked
    # before 上半身 as the canonical chest.
    assert aliases["chest"].name == "上半身2"
    assert aliases["hip"].name == "センター"
    assert aliases["upper_arm_L"].name == "左腕"
    assert aliases["lower_arm_R"].name == "右ひじ"
    assert aliases["foot_L"].name == "左足首"
    assert aliases["toe_R"].name == "右つま先"


def test_detect_mixamo_rig() -> None:
    """Mixamo's ``mixamorig:Head`` style names resolve."""
    joints = _nodes(
        "mixamorigHips", "mixamorigSpine", "mixamorigSpine1", "mixamorigSpine2",
        "mixamorigNeck", "mixamorigHead",
        "mixamorigLeftArm", "mixamorigLeftForeArm", "mixamorigLeftHand",
        "mixamorigRightArm", "mixamorigRightForeArm", "mixamorigRightHand",
        "mixamorigLeftUpLeg", "mixamorigLeftLeg", "mixamorigLeftFoot", "mixamorigLeftToeBase",
        "mixamorigRightUpLeg", "mixamorigRightLeg", "mixamorigRightFoot", "mixamorigRightToeBase",
    )
    aliases = detect_humanoid_aliases(joints)
    assert aliases["head"].name == "mixamorigHead"
    assert aliases["upper_arm_L"].name == "mixamorigLeftArm"
    assert aliases["lower_arm_L"].name == "mixamorigLeftForeArm"
    assert aliases["upper_leg_R"].name == "mixamorigRightUpLeg"
    assert aliases["foot_R"].name == "mixamorigRightFoot"


def test_unknown_rig_returns_empty_dict() -> None:
    """Rigs with no matching pattern get an empty alias map, no false positives."""
    joints = _nodes(
        "weirdBone1", "weirdBone2", "weirdBone3",
        "ribbon_chain_0", "decoration_a",
    )
    aliases = detect_humanoid_aliases(joints)
    assert aliases == {}


def test_partial_rig_still_returns_what_matches() -> None:
    """A rig missing some canonical bones still gets aliases for what it has."""
    joints = _nodes("J_Bip_C_Head", "J_Bip_C_Hips")  # only head + hip
    aliases = detect_humanoid_aliases(joints)
    assert set(aliases.keys()) == {"head", "hip"}
    assert aliases["head"].name == "J_Bip_C_Head"
    assert aliases["hip"].name == "J_Bip_C_Hips"


def test_scene_find_prefers_alias_over_literal_match() -> None:
    """``scene.find("head")`` returns the canonical bone, not a literal lookup."""
    head_node = Node(name="Head_M_055")
    scene = Scene()
    scene.root.add_child(head_node)
    scene.bone_aliases = {"head": head_node}
    found = scene.find("head")
    assert found is head_node


def test_scene_find_falls_back_to_literal_lookup_for_non_aliases() -> None:
    """Asking for a specific rig bone name still works even with aliases set."""
    head_node = Node(name="Head_M_055")
    scene = Scene()
    scene.root.add_child(head_node)
    scene.bone_aliases = {"head": head_node}
    found = scene.find("Head_M_055")
    assert found is head_node


def test_scene_find_returns_none_for_missing_node() -> None:
    """``scene.find`` falls through aliases AND literal search before returning None."""
    scene = Scene()
    scene.bone_aliases = {"head": Node(name="Head_M_055")}
    assert scene.find("nonexistent_bone") is None


def test_scene_bone_aliases_default_to_empty_dict() -> None:
    """Fresh scenes have an empty alias map (no surprise mutations)."""
    scene = Scene()
    assert scene.bone_aliases == {}
    # find() falls back to literal lookup with no aliases present.
    test_node = Node(name="foo")
    scene.root.add_child(test_node)
    assert scene.find("foo") is test_node
    assert scene.find("missing") is None


def test_importer_manager_populates_aliases_on_bundled_herta() -> None:
    """``ImporterManager.load`` runs auto-rig → aliases populated on the bundled model."""
    repo_root = Path(__file__).resolve().parent.parent
    absolute = repo_root / "examples" / "assets" / "herta" / "herta.glb"
    if not absolute.is_file():
        pytest.skip("herta.glb not bundled in this checkout")
    sys.path.insert(0, str(repo_root / "importers"))
    from posecascade.assets.importer_manager import ImporterManager  # noqa: PLC0415

    manager = ImporterManager(importers_root=repo_root / "importers")
    manager.discover()
    scene = manager.load(absolute)
    assert "head" in scene.scene.bone_aliases
    # The bundled binary is The Herta (uses ``Head_<digits>`` naming);
    # the alias detector resolves the canonical ``head`` slot via the
    # ``^Head_\d+$`` pattern regardless of which HSR rig is bundled.
    head_name = scene.scene.bone_aliases["head"].name
    assert head_name.startswith("Head_"), f"Expected Head_<digits>, got {head_name!r}"
    # The same scene.find call user scripts make returns the canonical bone.
    assert scene.scene.find("head") is scene.scene.bone_aliases["head"]
