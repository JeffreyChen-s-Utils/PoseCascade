"""Heuristic mapper from rig-specific bone names → canonical humanoid keys.

Different humanoid asset pipelines bake different bone-name conventions:

- **VRoid / VRM**: ``J_Bip_C_Head``, ``J_Bip_L_UpperArm``, ``J_Bip_L_UpperLeg``
- **HoYoverse FBX (Aplaybox / Sketchfab rips)**: ``Head_M_055``, ``Shoulder_L_0183``,
  ``Hip_L_02`` — Maya HIK-style names with numeric suffixes.
- **MMD PMX / PMD**: Japanese names — ``頭``, ``左腕``, ``左足``.
- **Mixamo**: ``mixamorig:Head``, ``mixamorig:LeftArm``.

User scripts (``examples/scripts/walk.py``, ``showcase.json``, …) reference
canonical names (``head``, ``upper_arm_L``, ``upper_leg_L``) because they
encode anatomical intent rather than format trivia. Without an
alias layer, ``scene.find("head")`` returns ``None`` on every real rig
because no rig literally names a bone ``head`` — the scripts silently
no-op.

This module fixes that. :func:`detect_humanoid_aliases` walks a rig's
joints, pattern-matches each canonical name against an ordered list of
regexes (most-specific first), and returns ``{canonical → Node}`` for
every match it finds. The dict is attached to the :class:`Scene` by
:func:`posecascade.assets.auto_rig.apply_post_import_rig` so script
authors can keep writing in canonical vocabulary.

Patterns are conservative — when the rig is ambiguous (multiple
plausible matches), pick the first hit in the priority order rather
than guessing. A rig that doesn't match any pattern simply gets an
empty alias dict; scripts then degrade to no-op exactly as they did
before (no behaviour regression).
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from posecascade.scene.node import Node

# Canonical humanoid keys → ordered list of regex patterns to match
# joint names against. Patterns are tried in declaration order; the
# first match wins. Anchored with ^/$ so partial matches in long
# decorative bone names (e.g. ``HeadPart1`` shouldn't shadow ``Head``)
# don't accidentally claim a slot.
#
# The order within each list goes most-specific → most-generic:
#   1. Format-specific full names (J_Bip_C_Head, Head_M_055, 頭)
#   2. Mixamo / generic FBX names (mixamorig:Head, Head)
#   3. Catch-all loose matches kept disabled for now to avoid false
#      positives on decorative bones.
_DETECTORS: dict[str, list[str]] = {
    # ----- spine + head -----
    "head": [
        r"^J_Bip_C_Head$",
        r"^Head_M_\d+$",
        r"^Head_\d+$",
        r"^mixamorig:?Head$",
        r"^Head$",
        r"^頭$",
    ],
    "neck": [
        r"^J_Bip_C_Neck$",
        r"^Neck_M_\d+$",
        r"^Neck_\d+$",
        r"^mixamorig:?Neck$",
        r"^Neck$",
        r"^首$",
    ],
    "chest": [
        r"^J_Bip_C_UpperChest$",
        r"^J_Bip_C_Chest$",
        r"^Chest_M_\d+$",
        r"^Chest_\d+$",
        r"^Spine2_M_\d+$",
        r"^mixamorig:?Spine2$",
        r"^Chest$",
        r"^上半身2$",
        r"^上半身$",
    ],
    "spine": [
        r"^J_Bip_C_Spine$",
        r"^Spine1_M_\d+$",
        r"^Spine_\d+$",
        r"^mixamorig:?Spine1$",
        r"^mixamorig:?Spine$",
        r"^Spine$",
    ],
    "hip": [
        r"^J_Bip_C_Hips$",
        r"^Root_M_\d+$",
        r"^Hips_\d+$",
        r"^mixamorig:?Hips$",
        r"^Hips$",
        r"^センター$",
        r"^下半身$",
    ],
    # ----- arms (L / R mirrored) -----
    "upper_arm_L": [
        r"^J_Bip_L_UpperArm$",
        r"^Shoulder_L_\d+$",
        # HSR-split rigs distinguish ``Left shoulder`` (clavicle, used for
        # shrug) from ``Left arm`` (actual upper arm above the elbow).
        # ``Left arm`` is what scripts targeting ``upper_arm_L`` expect —
        # rotating the clavicle just shifts the whole arm sideways and
        # leaves the arm at T-pose. List ``Left arm`` BEFORE the bare
        # ``shoulder`` fallback so the proper bone wins.
        r"^Left arm_\d+$",
        r"^mixamorig:?LeftArm$",
        r"^UpperArm_L$",
        r"^左腕$",
    ],
    "upper_arm_R": [
        r"^J_Bip_R_UpperArm$",
        r"^Shoulder_R_\d+$",
        r"^Right arm_\d+$",
        r"^mixamorig:?RightArm$",
        r"^UpperArm_R$",
        r"^右腕$",
    ],
    "lower_arm_L": [
        r"^J_Bip_L_LowerArm$",
        r"^Elbow_L_\d+$",
        r"^Left elbow_\d+$",
        r"^mixamorig:?LeftForeArm$",
        r"^LowerArm_L$",
        r"^左ひじ$",
    ],
    "lower_arm_R": [
        r"^J_Bip_R_LowerArm$",
        r"^Elbow_R_\d+$",
        r"^Right elbow_\d+$",
        r"^mixamorig:?RightForeArm$",
        r"^LowerArm_R$",
        r"^右ひじ$",
    ],
    "hand_L": [
        r"^J_Bip_L_Hand$",
        r"^Wrist_L_\d+$",
        r"^Left wrist_\d+$",
        r"^mixamorig:?LeftHand$",
        r"^Hand_L$",
        r"^左手首$",
    ],
    "hand_R": [
        r"^J_Bip_R_Hand$",
        r"^Wrist_R_\d+$",
        r"^Right wrist_\d+$",
        r"^mixamorig:?RightHand$",
        r"^Hand_R$",
        r"^右手首$",
    ],
    # ----- legs (L / R mirrored) -----
    "upper_leg_L": [
        r"^J_Bip_L_UpperLeg$",
        r"^Hip_L_\d+$",
        r"^Left leg_\d+$",
        r"^mixamorig:?LeftUpLeg$",
        r"^UpperLeg_L$",
        r"^左足$",
    ],
    "upper_leg_R": [
        r"^J_Bip_R_UpperLeg$",
        r"^Hip_R_\d+$",
        r"^Right leg_\d+$",
        r"^mixamorig:?RightUpLeg$",
        r"^UpperLeg_R$",
        r"^右足$",
    ],
    "lower_leg_L": [
        r"^J_Bip_L_LowerLeg$",
        r"^Knee_L_\d+$",
        r"^Left knee_\d+$",
        r"^mixamorig:?LeftLeg$",
        r"^LowerLeg_L$",
        r"^左ひざ$",
    ],
    "lower_leg_R": [
        r"^J_Bip_R_LowerLeg$",
        r"^Knee_R_\d+$",
        r"^Right knee_\d+$",
        r"^mixamorig:?RightLeg$",
        r"^LowerLeg_R$",
        r"^右ひざ$",
    ],
    "foot_L": [
        r"^J_Bip_L_Foot$",
        r"^Ankle_L_\d+$",
        r"^Left ankle_\d+$",
        r"^mixamorig:?LeftFoot$",
        r"^Foot_L$",
        r"^左足首$",
    ],
    "foot_R": [
        r"^J_Bip_R_Foot$",
        r"^Ankle_R_\d+$",
        r"^Right ankle_\d+$",
        r"^mixamorig:?RightFoot$",
        r"^Foot_R$",
        r"^右足首$",
    ],
    "toe_L": [
        r"^J_Bip_L_ToeBase$",
        r"^Toes_L_\d+$",
        r"^mixamorig:?LeftToeBase$",
        r"^ToeBase_L$",
        r"^左つま先$",
    ],
    "toe_R": [
        r"^J_Bip_R_ToeBase$",
        r"^Toes_R_\d+$",
        r"^mixamorig:?RightToeBase$",
        r"^ToeBase_R$",
        r"^右つま先$",
    ],
    # ----- shoulder clavicle (separate from upper_arm where rigs split them) -----
    # Skipped intentionally — rigs that don't split clavicle/upper-arm
    # (March 7th FBX, MMD) would alias both shoulder_L and upper_arm_L
    # to the same node, which is misleading. Scripts that need the
    # clavicle should target ``upper_arm_L`` and accept that on merged
    # rigs the rotation includes the collar bone as well.
}

# Pre-compile every regex once so :func:`detect_humanoid_aliases` doesn't
# pay the compile cost on every import. Module load happens once per
# process, so the up-front cost is amortised across every scene load.
_COMPILED: dict[str, list[re.Pattern[str]]] = {
    canonical: [re.compile(pat) for pat in patterns]
    for canonical, patterns in _DETECTORS.items()
}


def canonical_keys() -> tuple[str, ...]:
    """Return every canonical humanoid key the detector knows about.

    Stable ordering by declaration order in :data:`_DETECTORS` — useful
    for tests, debug dumps, and the editor's bone-mapping inspector.
    """
    return tuple(_DETECTORS.keys())


def detect_humanoid_aliases(joints: Iterable[Node]) -> dict[str, Node]:
    """Return ``{canonical → Node}`` for every canonical key the rig matches.

    Walks ``joints`` once to build a name → node lookup, then for each
    canonical key tries every pattern in priority order. The first
    matching joint claims that canonical slot; later patterns for the
    same key are skipped. A canonical key with no matching joint is
    simply absent from the returned dict (no None entries).

    The same joint can appear under multiple canonical keys when a rig
    merges anatomical roles (e.g. March 7th FBX has no separate
    clavicle / upper-arm bones, so future shoulder_L / upper_arm_L
    aliases would both point at ``Shoulder_L_0183``). The current
    pattern set avoids that overlap; if it ever happens, callers
    should treat the duplicate references as intentional.
    """
    by_name: dict[str, Node] = {}
    for joint in joints:
        if joint.name not in by_name:
            by_name[joint.name] = joint

    aliases: dict[str, Node] = {}
    for canonical, regexes in _COMPILED.items():
        for regex in regexes:
            match = _first_matching_node(regex, by_name)
            if match is not None:
                aliases[canonical] = match
                break
    return aliases


def _first_matching_node(
    regex: re.Pattern[str], by_name: dict[str, Node],
) -> Node | None:
    """Return the first node whose name fully matches ``regex``, or ``None``.

    Iteration order over ``by_name`` follows insertion order (dict spec
    since 3.7), which is the joint order the importer produced. That's
    the rig's natural traversal — top-down for hierarchies — so when
    multiple joints could legitimately match (e.g. ``Spine1_M_037`` vs
    ``Spine2_M_044`` for the ``spine`` key), the higher-up bone wins
    only because its pattern appears earlier in the list. Within a
    single pattern, the first joint seen by the importer is taken.
    """
    for name, node in by_name.items():
        if regex.match(name):
            return node
    return None


__all__ = ["canonical_keys", "detect_humanoid_aliases"]
