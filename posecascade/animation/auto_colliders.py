"""Emit a standard humanoid body collider set from canonical bone aliases.

When a declarative animation attaches cloth pieces to a humanoid rig, the
common case is "I want the skirt / cape / hair to collide with the body
underneath." The cloth solver already supports sphere + capsule colliders,
but wiring them per-asset is tedious — the user has to know each rig's
exact bone names (``Hip_L_02`` on a Sketchfab FBX rip, ``J_Bip_L_UpperLeg``
on a VRoid rig, ``左足`` on an MMD PMX, …).

:func:`emit_humanoid_body_colliders` reads the canonical bone aliases the
importer's auto-rig produces (``hip``, ``upper_leg_L``, ``foot_R`` etc.)
and emits a standard set of capsule + sphere colliders covering the hips,
thighs, shins, and a hip sphere — enough geometry to keep a skirt off
the body without per-character wiring.

Users who want different geometry can still add explicit colliders in the
JSON (auto-emit MERGES with them) or set ``auto_body_colliders: false``
on the document to disable entirely.

Pure helper — no Qt, no GL, deterministic from the alias map alone.
"""
from __future__ import annotations

from typing import Any, Protocol

# Body collider standard set, in (start_alias, end_alias, kind, radius_m).
# ``end_alias=None`` makes it a sphere; otherwise a capsule from start→end.
# Radii are metres on a metre-scale humanoid (head crown ≈ 1.6 m).
# Tuned so a typical mini-skirt drapes off the body during a 45° hip
# rotation without the inner thigh or knee poking through the cloth.
# The hip sphere + thigh capsules together form a continuous shell
# around the pelvis-to-knee region that the cloth solver pushes against.
#
# Torso + arm + head set added so spring-chain hair pushed by wind / pose
# changes doesn't pass through the chest, back, shoulders, or skull when
# the head tilts. Without it, a forward-bent crawl pose drops the hair into
# the chest mesh because the spring solver only sees leg geometry.
_HUMANOID_BODY_COLLIDERS: tuple[tuple[str, str | None, str, float], ...] = (
    # HSR/Genshin-style fine-grained capsule set — 18 colliders covering
    # every body region hair could swing into. Each radius oversized
    # (+2-4 cm over anatomical) so the cloth/spring solver pushes BONES
    # to a position whose VISIBLE MESH still leaves margin. Industry
    # convention: animators hand-add 15-30 of these per character.
    #
    # Pelvis + legs (skirt drape):
    ("hip",         None,          "sphere",  0.16),
    ("upper_leg_L", "lower_leg_L", "capsule", 0.14),
    ("upper_leg_R", "lower_leg_R", "capsule", 0.14),
    ("lower_leg_L", "foot_L",      "capsule", 0.08),
    ("lower_leg_R", "foot_R",      "capsule", 0.08),
    # Torso (back-hair drape + cape):
    ("spine",       "chest",       "capsule", 0.16),
    ("hip",         "spine",       "capsule", 0.16),
    # Chest → neck bridge — closes the gap at the collar where back-hair
    # swinging forward in bent-over poses would otherwise sneak between
    # the chest cap and the throat cylinder.
    ("chest",       "neck",        "capsule", 0.13),
    # Neck → head — 0.13 is the safe radius: thicker (0.18) pushes
    # back-hair anchors UP because they sit within the cylinder when
    # the head tilts, then the per-frame push-out fights the chain's
    # dynamics. 0.13 hugs the actual throat surface while leaving room
    # for the chain's first joint above the back of the skull.
    ("neck",        "head",        "capsule", 0.13),
    # Head sphere — covers skull mass (not the hat which has its own
    # Hat_Root collider attached separately in scripts). Kept tight
    # (~7 cm) so spring-chain hair anchored at the back of the skull
    # (BackHairUpper, ponytails) isn't trapped INSIDE the sphere at
    # construction time — that triggers per-frame collision push-out
    # that fights against the chain's own dynamics.
    ("head",        None,          "sphere",  0.07),
    # Shoulder spheres — close the gap at the upper-arm root where
    # plain chest→upper-arm transitions miss the deltoid bulge.
    ("upper_arm_L", None,          "sphere",  0.09),
    ("upper_arm_R", None,          "sphere",  0.09),
    # Upper + lower arm chain — radii inflated to catch hair brushing
    # past extended arms in crawl / forward-reach poses.
    ("upper_arm_L", "lower_arm_L", "capsule", 0.08),
    ("upper_arm_R", "lower_arm_R", "capsule", 0.08),
    ("lower_arm_L", "hand_L",      "capsule", 0.08),
    ("lower_arm_R", "hand_R",      "capsule", 0.08),
    # Hand spheres — back-of-hand bulge that thin lower-arm caps miss.
    ("hand_L",      None,          "sphere",  0.07),
    ("hand_R",      None,          "sphere",  0.07),
)


class _ColliderSpec(Protocol):
    """Minimal structural interface expected from the caller's collider spec.

    The caller (:mod:`posecascade.scripting.declarative`) defines its own
    ``ColliderSpec`` dataclass; we don't import it here to keep this module
    dependency-free. ``emit_humanoid_body_colliders`` is parameterised on
    the constructor so the caller stays in control of the concrete type.
    """

    kind: str
    follow_bone: str
    radius: float
    end_bone: str | None
    skin_offset: float


def emit_humanoid_body_colliders(
    bone_aliases: dict[str, Any],
    *,
    spec_factory: Any,
    skin_offset: float = 0.005,
) -> list[_ColliderSpec]:
    """Build the standard humanoid body collider set from canonical aliases.

    Parameters
    ----------
    bone_aliases
        ``{canonical_name → Node}`` produced by
        :func:`posecascade.animation.bone_aliasing.detect_humanoid_aliases`
        and stored on :attr:`posecascade.scene.scene.Scene.bone_aliases`.
        Bones missing from the alias map are silently skipped — a rig
        without distinct foot bones still gets the hip + thigh colliders.
    spec_factory
        Callable matching ``ColliderSpec``'s constructor (kw-only:
        ``kind``, ``follow_bone``, ``radius``, ``end_bone``,
        ``skin_offset``). Passed in by the caller so this module stays
        free of a runtime import cycle.
    skin_offset
        Extra clearance pushed into every emitted collider so cloth verts
        don't graze the underlying body geometry. Default 5 mm.

    Returns
    -------
    list[ColliderSpec]
        Ordered list of collider specs ready to feed into the cloth host.
        Empty if ``bone_aliases`` doesn't contain any of the canonical
        body keys.
    """
    out: list[_ColliderSpec] = []
    for start_alias, end_alias, kind, radius in _HUMANOID_BODY_COLLIDERS:
        start_node = bone_aliases.get(start_alias)
        if start_node is None:
            continue
        if kind == "sphere":
            out.append(
                spec_factory(
                    kind="sphere",
                    follow_bone=start_node.name,
                    radius=radius,
                    end_bone=None,
                    skin_offset=skin_offset,
                ),
            )
        elif kind == "capsule":
            end_node = bone_aliases.get(end_alias) if end_alias else None
            if end_node is None:
                continue
            out.append(
                spec_factory(
                    kind="capsule",
                    follow_bone=start_node.name,
                    radius=radius,
                    end_bone=end_node.name,
                    skin_offset=skin_offset,
                ),
            )
    return out
