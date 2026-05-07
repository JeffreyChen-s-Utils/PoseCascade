"""Bone post-IK resolver: append (付与) inheritance + fixed-axis projection.

PMX bones can carry two non-VMD-driven rules that fire after the IK
pass:

1. **Append (付与)**: a bone "inherits" rotation and/or translation
   *delta* from another bone, scaled by a per-bone weight. Used for
   things like an upper-body twist bone that picks up a percentage of
   the spine's rotation, or a follow bone that copies a hand's
   translation at half-weight.

2. **Fixed axis**: the bone's rotation is filtered to keep only the
   component spinning around a single unit axis. Hair / tail joints
   typically use this so they can twist along their length but never
   bend sideways.

PMX evaluates these in *deformation order* — an explicit per-bone
``deformation_depth`` integer that the importer sorts by. We mirror
that exactly here, because a bone can append from another bone whose
own rotation is itself the product of an earlier append.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from posecascade.scene.node import Node
from posecascade.utils.math3d import (
    Quat,
    Vec3,
    normalize,
    quat_conjugate,
    quat_identity,
    quat_mul,
    quat_normalize,
    quat_slerp,
)

_EPSILON = 1.0e-6


@dataclass(frozen=True)
class BoneAppendRule:
    """One PMX append (``付与``) declaration.

    A bone may inherit rotation, translation, or both from
    ``parent_index``. The weight scales the inherited delta; negative
    weights flip its sense (a common authoring trick to mirror the
    parent's rotation onto a bone on the other side of the body).
    """

    bone_index: int
    parent_index: int
    weight: float = 1.0
    inherit_rotation: bool = False
    inherit_translation: bool = False


@dataclass(frozen=True)
class FixedAxisRule:
    """A bone whose rotation is filtered to a single unit axis."""

    bone_index: int
    axis: tuple[float, float, float]   # unit vector in bone-local space


@dataclass
class BoneResolver:
    """Applies append + fixed-axis rules to a model's bone Nodes.

    Construction snapshots each bone's *rest* TRS so the per-frame
    "delta from rest" used by append is computable without re-walking
    the rest pose every call.
    """

    deformation_order: tuple[int, ...]
    appends: tuple[BoneAppendRule, ...] = ()
    fixed_axes: tuple[FixedAxisRule, ...] = ()
    bone_index_to_node: dict[int, Node] = field(default_factory=dict)
    rest_pose: dict[int, tuple[Vec3, Quat]] = field(default_factory=dict)

    @classmethod
    def from_rules(
        cls,
        rules: BoneResolverRules,
        bone_index_to_node: dict[int, Node],
    ) -> BoneResolver:
        """Build a resolver, snapshotting rest TRS for every mapped bone."""
        rest: dict[int, tuple[Vec3, Quat]] = {}
        for index, node in bone_index_to_node.items():
            rest[index] = (
                node.transform.translation.copy(),
                node.transform.rotation.copy(),
            )
        return cls(
            deformation_order=rules.deformation_order,
            appends=rules.appends,
            fixed_axes=rules.fixed_axes,
            bone_index_to_node=bone_index_to_node,
            rest_pose=rest,
        )

    def resolve(self) -> None:
        """Apply append + fixed-axis to every bone in deformation order.

        Run AFTER VMD bone keyframes, bone morphs, and IK have written
        their results onto the bone Nodes. Mutates Node TRS in place.
        """
        appends_by_bone = {rule.bone_index: rule for rule in self.appends}
        fixed_by_bone = {rule.bone_index: rule for rule in self.fixed_axes}
        for bone_index in self.deformation_order:
            node = self.bone_index_to_node.get(bone_index)
            if node is None:
                continue
            append = appends_by_bone.get(bone_index)
            if append is not None:
                self._apply_append(node, append)
            fixed = fixed_by_bone.get(bone_index)
            if fixed is not None:
                self._apply_fixed_axis(node, fixed)

    # ----- append --------------------------------------------------------
    def _apply_append(self, node: Node, rule: BoneAppendRule) -> None:
        parent_node = self.bone_index_to_node.get(rule.parent_index)
        parent_rest = self.rest_pose.get(rule.parent_index)
        if parent_node is None or parent_rest is None:
            return
        if rule.inherit_translation:
            node.transform.set_translation(
                _compose_translation_delta(node, parent_node, parent_rest, rule.weight)
            )
        if rule.inherit_rotation:
            node.transform.set_rotation(
                _compose_rotation_delta(node, parent_node, parent_rest, rule.weight)
            )

    # ----- fixed axis ----------------------------------------------------
    def _apply_fixed_axis(self, node: Node, rule: FixedAxisRule) -> None:
        axis = np.asarray(rule.axis, dtype=np.float32)
        norm = float(np.linalg.norm(axis))
        if norm < _EPSILON:
            return
        unit_axis = axis / norm
        node.transform.set_rotation(
            quat_project_to_axis(node.transform.rotation, unit_axis)
        )


@dataclass(frozen=True)
class BoneResolverRules:
    """Importer-side bundle of every append / fixed-axis rule + the order
    in which to run them. The importer constructs this once; the player
    builds a per-instance :class:`BoneResolver` against the live skin's
    bone Nodes."""

    deformation_order: tuple[int, ...] = ()
    appends: tuple[BoneAppendRule, ...] = ()
    fixed_axes: tuple[FixedAxisRule, ...] = ()


def _compose_translation_delta(
    node: Node,
    parent_node: Node,
    parent_rest: tuple[Vec3, Quat],
    weight: float,
) -> Vec3:
    """Add ``weight × (parent_current - parent_rest)`` to the bone's translation."""
    rest_t, _rest_r = parent_rest
    parent_delta = (parent_node.transform.translation - rest_t).astype(np.float32, copy=False)
    return (node.transform.translation + parent_delta * weight).astype(np.float32, copy=False)


def _compose_rotation_delta(
    node: Node,
    parent_node: Node,
    parent_rest: tuple[Vec3, Quat],
    weight: float,
) -> Quat:
    """Multiply the bone's rotation by ``slerp(identity, parent_delta, weight)``.

    A negative ``weight`` flips the inherited delta's sense (right-side
    twin bones in mirrored rigs); we handle it by conjugating the delta
    and using the absolute weight for the slerp.
    """
    _rest_t, rest_r = parent_rest
    parent_delta = quat_mul(parent_node.transform.rotation, quat_conjugate(rest_r))
    if weight < 0.0:
        parent_delta = quat_conjugate(parent_delta)
        weight = -weight
    contribution = quat_slerp(quat_identity(), parent_delta, float(min(weight, 1.0)))
    if weight > 1.0:
        # Outside the slerp's [0, 1] range — fall back to chaining the delta
        # ``floor(weight)`` whole turns plus the fractional slerp on top.
        whole_turns = int(weight)
        for _ in range(whole_turns):
            contribution = quat_mul(parent_delta, contribution)
    return quat_normalize(quat_mul(node.transform.rotation, contribution))


def quat_project_to_axis(q: Quat, axis: Vec3) -> Quat:
    """Swing-twist decomposition: keep only the rotation around ``axis``.

    Implementation follows the standard formula:

    - vec part of ``q`` projected onto ``axis`` becomes the twist's vec part
    - the scalar (``w``) is preserved
    - the result is renormalised to unit length

    The remaining "swing" component (rotation perpendicular to the axis)
    is dropped — exactly what PMX's ``fixed_axis`` flag specifies.
    """
    qx, qy, qz, qw = (float(v) for v in q)
    ax, ay, az = (float(v) for v in axis)
    dot_axis = qx * ax + qy * ay + qz * az
    twisted_vec = np.array(
        [ax * dot_axis, ay * dot_axis, az * dot_axis, qw],
        dtype=np.float32,
    )
    return quat_normalize(twisted_vec)


# Keep ``normalize`` importable through this module — the importer uses it
# to ensure axis vectors are unit-length before stashing them on a rule.
__all__ = [
    "BoneAppendRule",
    "BoneResolver",
    "BoneResolverRules",
    "FixedAxisRule",
    "normalize",
    "quat_project_to_axis",
]
