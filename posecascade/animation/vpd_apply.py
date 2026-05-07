"""Apply a :class:`~vpd.types.VpdPose` to an :class:`ImportedScene`.

VPD is a single-frame snapshot: walk every override and write directly
onto the scene's bone Nodes / morph state. Bone translations are added
to the rest pose (matches VMD's offset semantics so a VPD applied at
``t = 0`` and a VMD bone keyframe at frame 0 don't conflict). Missing
bones / morphs are silently skipped — a VPD authored for a different
model than the scene is a routine retargeting case, not an error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from vpd.types import VpdPose

from posecascade.animation.morph_accumulator import (
    LeafWeights,
    accumulate_weights,
)
from posecascade.animation.morph_apply import MorphApplier
from posecascade.animation.vmd_track import vmd_bone_key
from posecascade.assets.types import ImportedScene
from posecascade.scene.node import Node
from posecascade.utils.math3d import quat_normalize


@dataclass(frozen=True)
class VpdApplyResult:
    """Diagnostics from :func:`apply_pose` — useful for tests + the UI."""

    bones_applied: int
    bones_skipped: tuple[str, ...]
    morphs_applied: int
    morphs_skipped: tuple[str, ...]


def apply_pose(pose: VpdPose, scene: ImportedScene) -> VpdApplyResult:
    """Write ``pose``'s overrides onto ``scene``'s bones + morphs.

    Returns a tally of which entries hit and which were skipped — same
    information the pose-library UI shows next to "loaded N/M bones".
    """
    bones_applied, bones_skipped = _apply_bones(pose, scene)
    morphs_applied, morphs_skipped = _apply_morphs(pose, scene)
    return VpdApplyResult(
        bones_applied=bones_applied,
        bones_skipped=bones_skipped,
        morphs_applied=morphs_applied,
        morphs_skipped=morphs_skipped,
    )


def _apply_bones(pose: VpdPose, scene: ImportedScene) -> tuple[int, tuple[str, ...]]:
    if not pose.bones or not scene.skins:
        return 0, tuple(bone.name for bone in pose.bones)
    bone_lookup: dict[str, Node] = {}
    for joint in scene.skins[0].joints:
        if isinstance(joint, Node):
            bone_lookup.setdefault(vmd_bone_key(joint.name), joint)
    rest_pose = {
        key: node.transform.translation.copy() for key, node in bone_lookup.items()
    }
    applied = 0
    skipped: list[str] = []
    for override in pose.bones:
        key = vmd_bone_key(override.name)
        node = bone_lookup.get(key)
        if node is None:
            skipped.append(override.name)
            continue
        rest_translation = rest_pose[key]
        node.transform.set_translation(
            (
                rest_translation
                + np.asarray(override.translation, dtype=np.float32)
            ).astype(np.float32, copy=False)
        )
        node.transform.set_rotation(
            quat_normalize(
                np.asarray(override.rotation, dtype=np.float32),
            ).astype(np.float32, copy=False)
        )
        applied += 1
    return applied, tuple(skipped)


def _apply_morphs(pose: VpdPose, scene: ImportedScene) -> tuple[int, tuple[str, ...]]:
    if not pose.morphs or not scene.morphs.by_index:
        return 0, tuple(morph.name for morph in pose.morphs)
    name_lookup: dict[str, int] = {
        vmd_bone_key(morph.name): index
        for index, morph in enumerate(scene.morphs.by_index)
    }
    weights: dict[str, float] = {}
    skipped: list[str] = []
    for override in pose.morphs:
        if vmd_bone_key(override.name) not in name_lookup:
            skipped.append(override.name)
            continue
        weights[override.name] = float(override.weight)
    if not weights:
        return 0, tuple(skipped)
    applier = MorphApplier(scene)
    indexed = {
        name_lookup[vmd_bone_key(name)]: weight
        for name, weight in weights.items()
    }
    leaf = LeafWeights(weights=dict(indexed))
    # Run the accumulator manually so group / flip morphs pulled in by
    # name still expand to their leaf set.
    name_keyed = {
        scene.morphs.by_index[i].name: w
        for i, w in indexed.items()
    }
    leaf = accumulate_weights(scene.morphs, name_keyed)
    applier.apply(leaf)
    return len(weights), tuple(skipped)
