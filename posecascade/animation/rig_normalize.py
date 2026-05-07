"""Post-import normalization for nested-EMPTY rig wrappers.

VRoid / Sketchfab / FBX-from-Blender exports often wrap the actual
character in a chain of pure-transform EMPTYs:

    Sketchfab_model (R=-90°X)         ← Y-up→Z-up convention
      └─ <fbx-id>.fbx (R=+90°X)       ← FBX coordinate flip
        └─ RootNode
          └─ Armature (R=-90°X)       ← second axis-fix
            └─ <body content...>

The cumulative rotation across that chain is the right "natural facing"
fix, but a user script that wants to drive ``Sketchfab_model.rotation``
ends up composing its own yaw on top of an already-rotated stack and
the character ends up lying flat or upside-down. The same script,
written against a clean rig (one EMPTY at the scene root with no
rotation), works as expected.

This module collapses those wrapper chains at import time. Wrapper
EMPTYs (no mesh, no joint, single child) get their TRS folded into
their direct child, and they are removed from the tree. World
positions and bone bind matrices are preserved exactly, since the
fold is just shifting where the same combined transform is stored.

After collapse:
    Sketchfab_model (R=-90°X * +90°X * -90°X = -90°X all stacked)
      └─ <body content>

If the engine sees a single "wrapper" with the cumulative rotation, a
script targeting that node can reason about a single rest_rot — no
double-rotation fights.
"""
from __future__ import annotations

import numpy as np

from posecascade.assets.types import Skin
from posecascade.scene.component import MeshRefComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.utils.math3d import (
    compose_trs,
    decompose_trs,
    quat_from_axis_angle,
    quat_identity,
    quat_mul,
)

_AXIS_TOLERANCE = 1.0e-4
_UPRIGHT_TOLERANCE = 1.0e-3
# Threshold for picking a perpendicular axis when computing a 180° quaternion:
# below this the seed (X axis) is too aligned with ``a`` and we use Y instead.
_AXIS_PARALLEL_THRESHOLD = 0.9
# Body-up vectors below this norm are degenerate (head/feet at the same world
# point) — bail out rather than divide by ~0 and produce a wild rotation.
_DEGENERATE_BODY_UP_NORM = 1.0e-4
# Bone rest rotations smaller than this are considered "natural pose tilt"
# (collarbone droop, slight knee bend) — they shouldn't trigger orientation
# correction. The threshold catches the 90° / 180° axis flips that VRoid /
# Mixamo / FBX rigs bake in for coordinate-system reasons.
_SIGNIFICANT_ROTATION_RADIANS = 0.785398163397  # π/4 = 45°


def collapse_wrapper_empties(scene: Scene, skins: tuple[Skin, ...]) -> int:
    """Fold pure-transform wrapper EMPTYs into their child. Returns count removed.

    A node is treated as a "wrapper EMPTY" when it has:
      - No :class:`MeshRefComponent` attached (it's not a renderable).
      - It is not a joint of any imported skin.
      - It has exactly one child (so the fold is unambiguous).

    The wrapper's TRS is composed into the child's TRS so the child's
    world-space transform is preserved. The wrapper is removed from
    the tree by reparenting its child onto its parent in place.

    Direct children of ``scene.root`` are never folded — they remain
    the script-addressable "character root" handles (``scene.find(...)``
    on names like ``Sketchfab_model`` keeps working).
    """
    joint_ids = {id(j) for skin in skins for j in skin.joints}
    removed = 0
    for top_child in scene.root.children:
        # Recurse into each scene-root child, but never fold the
        # top-level handle itself.
        removed += _collapse_recursive(top_child, joint_ids)
    return removed


def _collapse_recursive(parent: Node, joint_ids: set[int]) -> int:
    """Walk ``parent``'s children, collapsing any wrapper EMPTYs in place.

    Returns the number of wrappers removed. The traversal is post-order
    so deeply-nested wrappers collapse before their (now-direct) parent.
    """
    removed = 0
    i = 0
    while i < len(parent.children):
        child = parent.children[i]
        # Recurse first so any nested wrappers below `child` are
        # collapsed before we decide whether `child` itself is one.
        removed += _collapse_recursive(child, joint_ids)
        if _is_wrapper(child, joint_ids):
            grandchild = child.children[0]
            _fold_transform_into(child, grandchild)
            # Replace `child` with `grandchild` in `parent`'s child list.
            parent.children[i] = grandchild
            grandchild.parent = parent
            removed += 1
            # Don't increment i — re-evaluate the new occupant of slot i
            # in case the grandchild is also a wrapper.
            continue
        i += 1
    return removed


def _is_wrapper(node: Node, joint_ids: set[int]) -> bool:
    """A wrapper is a single-child EMPTY that is not a joint and has no mesh."""
    if len(node.children) != 1:
        return False
    if id(node) in joint_ids:
        return False
    return all(not isinstance(c, MeshRefComponent) for c in node.components)


def _fold_transform_into(parent: Node, child: Node) -> None:
    """Compose ``parent``'s TRS into ``child``'s TRS so the child's world
    transform is preserved when ``parent`` is removed from the chain.

    Composition is done via 4x4 matrices to handle rotation, scale, and
    translation correctly in one step (a common pitfall is to compose
    rotations and translations separately and miss the parent-rotates-
    child-translation interaction).
    """
    parent_matrix = parent.transform.to_matrix()
    child_matrix = child.transform.to_matrix()
    composed = parent_matrix @ child_matrix
    translation, rotation, scale = decompose_trs(composed)
    child.transform.set_translation(translation)
    child.transform.set_rotation(rotation)
    child.transform.set_scale(scale)


def normalize_character_orientation(
    scene: Scene, skins: tuple[Skin, ...], *, lift_to_floor: bool = True
) -> int:
    """Rotate top-level wrappers so each rigged character stands upright at rest.

    Uses a SPATIAL approach: compute current world positions of head /
    feet joints, derive body-up direction from those, then rotate so
    body-up aligns with world +Y. This works regardless of the rig's
    bone-rest-rotation conventions (VRoid hip 180°-Z, etc.) because
    we look at where the joints physically end up, not at the local
    rotation chain.

    Falls back to a no-op if head/feet joints can't be identified by
    name (no 'head' substring or no 'foot' joints in the skin).

    After rotation, optionally lifts the rig along +Y so the lowest
    joint sits at world Y=0 — keeps the character on the floor instead
    of letting the rotation swing it below.

    Returns the number of skins for which a correction was applied.
    """
    count = 0
    for skin in skins:
        head, feet = _find_orientation_landmarks(skin.joints)
        if head is None or not feet:
            continue
        top = _find_top_wrapper(head, scene)
        if top is None:
            continue
        head_pos = _world_position(head)
        feet_avg = np.mean([_world_position(f) for f in feet], axis=0)
        body_up = head_pos - feet_avg
        norm = float(np.linalg.norm(body_up))
        if norm < _DEGENERATE_BODY_UP_NORM:
            continue
        body_up = body_up / norm
        # If already aligned with world +Y (within tolerance), no rotation needed.
        if abs(body_up[1] - 1.0) < _UPRIGHT_TOLERANCE:
            continue
        correction = _quat_align_vectors(body_up, np.array([0.0, 1.0, 0.0]))
        new_rotation = quat_mul(correction, top.transform.rotation)
        top.transform.set_rotation(new_rotation.astype(np.float32, copy=False))
        if lift_to_floor:
            _lift_skin_to_floor(top, skin)
        count += 1
    return count


def _find_orientation_landmarks(joints) -> tuple[Node | None, list[Node]]:
    """Pick the head joint and the foot joints by name match.

    ``head`` matches a single joint whose name contains ``head`` and
    isn't a face sub-joint (eye/jaw). ``feet`` collects every joint
    whose name contains ``foot`` (typically ``foot_L`` and ``foot_R``).
    """
    head: Node | None = None
    feet: list[Node] = []
    for j in joints:
        lower = j.name.lower()
        if "head" in lower and "eye" not in lower and "jaw" not in lower and head is None:
            head = j
        if "foot" in lower:
            feet.append(j)
    return head, feet


def _lift_skin_to_floor(top: Node, skin: Skin) -> None:
    """After a rotation correction, the rig may have flipped below Y=0.
    Add a +Y translation to ``top`` so the lowest joint sits at world Y=0."""
    min_y = float("inf")
    for joint in skin.joints:
        pos = _world_position(joint)
        if pos[1] < min_y:
            min_y = float(pos[1])
    if min_y == float("inf") or min_y >= -_UPRIGHT_TOLERANCE:
        return
    # Lift everything by -min_y; preserves rotation.
    old_t = np.asarray(top.transform.translation, dtype=np.float64)
    new_t = old_t.copy()
    new_t[1] -= min_y
    top.transform.set_translation(new_t.astype(np.float32, copy=False))


def _world_position(node: Node) -> np.ndarray:
    """Walk parent chain and compose 4x4 matrices to get the node's world translation."""
    matrix = node.transform.to_matrix()
    parent = node.parent
    while parent is not None:
        matrix = parent.transform.to_matrix() @ matrix
        parent = parent.parent
    return np.array(
        [float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])],
        dtype=np.float64,
    )


def _world_rotation_below(node: Node, top: Node):
    """Compose rotations from ``top``'s child down to and including ``node``.

    Excludes ``top`` itself — used by the orientation pass to compute
    "chain that survives if top's rotation is taken out", so we know
    what to set top.rotation to.
    """
    chain: list = []
    cur = node
    while cur is not None and cur is not top:
        chain.append(cur.transform.rotation)
        cur = cur.parent
    rot = quat_identity()
    for r in reversed(chain):
        rot = quat_mul(rot, r)
    return rot


def _quat_inverse(q):
    """Conjugate of a unit quaternion ``[x, y, z, w]`` → ``[-x, -y, -z, w]``."""
    return np.array(
        [-float(q[0]), -float(q[1]), -float(q[2]), float(q[3])], dtype=np.float32
    )


def _is_near_identity(quat) -> bool:
    """A unit quaternion close to ``[0, 0, 0, ±1]`` is identity for rotation."""
    return abs(abs(float(quat[3])) - 1.0) < _UPRIGHT_TOLERANCE


def _find_first_nontrivial_joint(joints) -> Node | None:
    """First joint with a SIGNIFICANT rest rotation (>= π/4).

    Small rotations on joints (5-15° for collar bones, slight knee
    bends, etc.) are part of the rig's natural pose, not orientation
    bugs. We only normalize when we see a clear axis flip — typically
    the 180° around -Z that VRoid bakes onto its hip.
    """
    for joint in joints:
        rot = np.asarray(joint.transform.rotation, dtype=np.float64)
        # Quaternion → angle: angle = 2 * acos(|qw|).
        qw_clamped = max(-1.0, min(1.0, abs(float(rot[3]))))
        angle = 2.0 * float(np.arccos(qw_clamped))
        if angle >= _SIGNIFICANT_ROTATION_RADIANS:
            return joint
    return None


def _find_top_wrapper(node: Node, scene: Scene) -> Node | None:
    """Walk up from ``node`` to the topmost ancestor that's a direct
    child of ``scene.root``. That node is the script-addressable
    handle (``Sketchfab_model`` or similar)."""
    cur = node
    while cur is not None:
        parent = cur.parent
        if parent is None:
            return None
        if parent is scene.root:
            return cur
        cur = parent
    return None


def _world_rotation(node: Node):
    """Compose rotations along ``node``'s parent chain (root → leaf)."""
    chain: list = []
    cur = node
    while cur is not None and cur.parent is not None:
        chain.append(cur.transform.rotation)
        cur = cur.parent
    rot = quat_identity()
    for r in reversed(chain):
        rot = quat_mul(rot, r)
    return rot


def _rotate_vector(quat, vec):
    """Apply a unit quaternion rotation to a 3-vector."""
    qx, qy, qz, qw = (float(c) for c in quat)
    vx, vy, vz = (float(c) for c in vec)
    # v' = q * v * conj(q), expanded for vec3.
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return np.array([rx, ry, rz], dtype=np.float64)


def _quat_align_vectors(from_vec, to_vec):
    """Quaternion that rotates ``from_vec`` to ``to_vec`` (both unit)."""
    a = from_vec / max(np.linalg.norm(from_vec), 1.0e-9)
    b = to_vec / max(np.linalg.norm(to_vec), 1.0e-9)
    dot = float(np.dot(a, b))
    if dot > 1.0 - _AXIS_TOLERANCE:
        return quat_identity()
    if dot < -1.0 + _AXIS_TOLERANCE:
        # 180° rotation; pick any axis perpendicular to ``a``.
        # Use a unit vector that's not parallel to ``a``.
        seed = (
            np.array([1.0, 0.0, 0.0])
            if abs(a[0]) < _AXIS_PARALLEL_THRESHOLD
            else np.array([0.0, 1.0, 0.0])
        )
        axis = np.cross(a, seed)
        axis = axis / max(np.linalg.norm(axis), 1.0e-9)
        return quat_from_axis_angle(axis.astype(np.float32, copy=False), float(np.pi))
    axis = np.cross(a, b)
    axis = axis / max(np.linalg.norm(axis), 1.0e-9)
    angle = float(np.arccos(max(-1.0, min(1.0, dot))))
    return quat_from_axis_angle(axis.astype(np.float32, copy=False), angle)


# Re-export for tests.
__all__ = [
    "collapse_wrapper_empties",
    "compose_trs",
    "normalize_character_orientation",
]
