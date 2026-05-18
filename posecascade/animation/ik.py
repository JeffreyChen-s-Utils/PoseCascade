"""MMD-flavoured CCD inverse-kinematics solver.

Each :class:`IkChain` describes one PMX IK definition:

- ``driver_bone_index`` is the "IK bone" — the visible target the user
  can drag in MMD. Its world position is what the chain tries to reach.
- ``effector_bone_index`` is the chain's tip — the bone whose world
  position should match the driver after solving.
- ``links`` are the rotatable joints between effector and root, ordered
  effector-to-root. Each :class:`IkLink` may carry per-axis Euler
  ``limit_min`` / ``limit_max`` (in radians, XYZ order) that clamp the
  link's local rotation after each CCD step — this is how knees stay on
  their bend axis.

The solver matches the MMD spec: axis-angle is computed in the link's
*local* frame, the per-step angle is capped at ``limit_radian`` per
iteration, and quaternion limits go through Euler-XYZ
decomposition / clamp / recomposition. Chains can be turned on / off
per-frame via :attr:`IkChain.enabled` so the player can drive them
straight from VMD's ``IK`` keyframe stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from posecascade.scene.node import Node
from posecascade.utils.math3d import (
    Mat4,
    Quat,
    Vec3,
    compose_trs,
    quat_from_axis_angle,
    quat_from_euler_xyz,
    quat_mul,
    quat_to_euler_xyz,
)

_EPSILON = 1.0e-6
_DOT_CLAMP = 1.0 - _EPSILON
# When picking a default fallback axis perpendicular to target_dir, treat
# target_dir as "almost vertical" (worth swapping the cross-product partner
# axis for) once its world-Y component crosses this threshold.
_PARALLEL_AXIS_THRESHOLD = 0.9


@dataclass(frozen=True)
class IkLink:
    """One rotatable joint in an IK chain."""

    bone_index: int
    has_limit: bool = False
    limit_min: tuple[float, float, float] = (0.0, 0.0, 0.0)   # XYZ radians
    limit_max: tuple[float, float, float] = (0.0, 0.0, 0.0)   # XYZ radians


@dataclass
class IkChain:
    """One IK chain — a CCD-driven rotation chain matching ``driver`` to ``effector``.

    ``enabled`` is mutable so VMD's per-frame IK on/off keyframes can flip
    it without rebuilding the chain. The frozen-vs-mutable choice mirrors
    PMX's: the chain definition itself is immutable per import, only the
    enable flag changes per frame.
    """

    driver_bone_index: int
    effector_bone_index: int
    iterations: int = 8
    limit_radian: float = 0.0          # per-step rotation cap; 0 means unlimited
    links: tuple[IkLink, ...] = field(default_factory=tuple)
    enabled: bool = True


def solve_chain(chain: IkChain, lookup: dict[int, Node]) -> None:
    """Run CCD on ``chain`` for at most ``iterations`` passes.

    Mutates the rotation of each link Node in place; bails early when the
    effector lands within :data:`_EPSILON` of the driver's world position.
    """
    if not chain.enabled or not chain.links:
        return
    driver = lookup.get(chain.driver_bone_index)
    effector = lookup.get(chain.effector_bone_index)
    if driver is None or effector is None:
        return
    target_world = _world_position(driver)
    for _ in range(max(chain.iterations, 1)):
        if _ccd_iteration(chain, lookup, effector, target_world):
            return


def _ccd_iteration(
    chain: IkChain,
    lookup: dict[int, Node],
    effector: Node,
    target_world: Vec3,
) -> bool:
    """One CCD pass; returns ``True`` when the effector hits the target."""
    for link in chain.links:
        link_node = lookup.get(link.bone_index)
        if link_node is None:
            continue
        effector_world = _world_position(effector)
        if _within_target(effector_world, target_world):
            return True
        delta = _compute_link_delta(link_node, link, chain, effector_world, target_world)
        if delta is None:
            continue
        new_rotation = quat_mul(link_node.transform.rotation, delta)
        if link.has_limit:
            new_rotation = _clamp_euler(new_rotation, link.limit_min, link.limit_max)
        link_node.transform.set_rotation(new_rotation.astype(np.float32, copy=False))
    effector_world = _world_position(effector)
    return _within_target(effector_world, target_world)


def _compute_link_delta(
    link_node: Node,
    link: IkLink,
    chain: IkChain,
    effector_world: Vec3,
    target_world: Vec3,
) -> Quat | None:
    """Return the local-frame delta quaternion for one CCD step (or ``None``)."""
    link_world_inv = _inverse_world_matrix(link_node)
    local_eff = _transform_point(link_world_inv, effector_world)
    local_tgt = _transform_point(link_world_inv, target_world)
    norm_eff = float(np.linalg.norm(local_eff))
    norm_tgt = float(np.linalg.norm(local_tgt))
    if norm_eff < _EPSILON or norm_tgt < _EPSILON:
        return None
    cos_angle = float(np.dot(local_eff, local_tgt) / (norm_eff * norm_tgt))
    cos_angle = max(-_DOT_CLAMP, min(_DOT_CLAMP, cos_angle))
    angle = float(np.arccos(cos_angle))
    if angle < _EPSILON:
        return None
    if chain.limit_radian > 0.0:
        angle = min(angle, chain.limit_radian)
    cross_v = np.cross(local_eff, local_tgt)
    axis_norm = float(np.linalg.norm(cross_v))
    if axis_norm < _EPSILON:
        return None
    axis = (cross_v / axis_norm).astype(np.float32, copy=False)
    return quat_from_axis_angle(axis, angle)


def _within_target(effector_world: Vec3, target_world: Vec3) -> bool:
    return float(np.linalg.norm(effector_world - target_world)) < _EPSILON


def _clamp_euler(
    rotation: Quat,
    limit_min: tuple[float, float, float],
    limit_max: tuple[float, float, float],
) -> Quat:
    """Quaternion → Euler XYZ → clamp per axis → quaternion."""
    rx, ry, rz = quat_to_euler_xyz(rotation)
    rx = max(limit_min[0], min(limit_max[0], rx))
    ry = max(limit_min[1], min(limit_max[1], ry))
    rz = max(limit_min[2], min(limit_max[2], rz))
    return quat_from_euler_xyz(rx, ry, rz)


def _world_matrix(node: Node) -> Mat4:
    """Compose ``node``'s world matrix by walking the parent chain.

    Mirrors :func:`posecascade.render.renderer._world_matrix`; we duplicate
    the helper here so the IK pass doesn't take a render-layer dependency.
    """
    matrix = node.transform.to_matrix()
    parent = node.parent
    while parent is not None:
        matrix = parent.transform.to_matrix() @ matrix
        parent = parent.parent
    return matrix.astype(np.float32, copy=False)


def _world_position(node: Node) -> Vec3:
    """The translation column of the world matrix (i.e., the bone's world origin)."""
    world = _world_matrix(node)
    return np.array([world[0, 3], world[1, 3], world[2, 3]], dtype=np.float32)


def _inverse_world_matrix(node: Node) -> Mat4:
    """Inverse of the world matrix; falls back to identity when singular."""
    world = _world_matrix(node)
    try:
        return np.linalg.inv(world).astype(np.float32, copy=False)
    except np.linalg.LinAlgError:
        return np.eye(4, dtype=np.float32)


def _transform_point(matrix: Mat4, point: Vec3) -> Vec3:
    """Apply a 4×4 matrix to a 3-vector treated as a point (w = 1)."""
    homogeneous = np.array(
        [float(point[0]), float(point[1]), float(point[2]), 1.0],
        dtype=np.float32,
    )
    transformed = matrix @ homogeneous
    return transformed[:3].astype(np.float32, copy=False)


# --- Script-friendly two-bone wrapper -----------------------------------
#
# The CCD core above takes bone *indices* and a lookup dict — it was built
# for PMX-imported skeletons where the bone hierarchy is index-addressable.
# User scripts do not have indices; they hold direct :class:`Node`
# references for the upper-leg / lower-leg / foot they want to drive.
#
# ``solve_two_bone`` adapts the CCD core to that usage: synthesise a
# disposable virtual driver Node placed at the world target, build an
# ad-hoc :class:`IkChain`, and run the same solver. Knee-style joints can
# be hinge-locked via ``mid_limit_min`` / ``mid_limit_max`` (Euler XYZ
# clamp inside the CCD step).


def solve_two_bone_analytic(
    root: Node,
    mid: Node,
    end: Node,
    target_world: Vec3,
    bend_hint: Vec3 | None = None,
) -> None:
    """Closed-form (law-of-cosines) 2-bone IK — guaranteed to converge.

    Given a chain root → mid → end and a world-space target, this places
    ``end`` exactly at ``target_world`` if the target is within the
    chain's reach. Outside the reachable annulus the leg either fully
    extends toward the target (for d > L1+L2) or folds as tight as it
    can (for d < |L1-L2|), so feet over a stair the leg can't reach
    still land at the closest possible point along the hip→target
    line — they don't snap, jitter, or detach from the chain.

    The knee bends in the same direction it currently points (the
    component of K-H perpendicular to H→T). When the current knee is
    colinear with the hip-target line — degenerate, no preferred
    side — ``bend_hint`` chooses the bend plane. Pass the body's
    forward direction (e.g. world ``vec3(0, 0, -1)`` for a character
    facing -Z) so the knee bends forward instead of locking in some
    arbitrary perpendicular axis.

    This is preferred over CCD for foot planting: every solve takes
    constant time (no iterations), and the result is deterministic.
    """
    hip = _world_position(root).astype(np.float64)
    knee = _world_position(mid).astype(np.float64)
    tip = _world_position(end).astype(np.float64)

    len_upper = float(np.linalg.norm(knee - hip))
    len_lower = float(np.linalg.norm(tip - knee))
    if len_upper < _EPSILON or len_lower < _EPSILON:
        return

    target = np.asarray(target_world, dtype=np.float64).reshape(3)
    to_target = target - hip
    distance = float(np.linalg.norm(to_target))
    if distance < _EPSILON:
        return

    # Clamp ``distance`` into the reachable annulus
    # [|len_upper-len_lower|, len_upper+len_lower]. Step a tiny epsilon
    # inside the bounds so the law-of-cosines argument stays away from
    # ±1 (where acos's derivative blows up).
    margin = _EPSILON
    inner = abs(len_upper - len_lower)
    outer = len_upper + len_lower
    d_clamped = max(inner + margin, min(outer - margin, distance))
    target_dir = to_target / distance

    # Hip-to-knee angle from the hip-target line. cos β follows from
    # the law of cosines applied to the (hip, knee, target) triangle
    # with sides (len_upper, d_clamped, len_lower).
    cos_beta = (
        len_upper * len_upper + d_clamped * d_clamped - len_lower * len_lower
    ) / (2.0 * len_upper * d_clamped)
    cos_beta = max(-1.0, min(1.0, cos_beta))
    beta = float(np.arccos(cos_beta))

    hip_to_knee = knee - hip
    bend_dir = _resolve_bend_direction(hip_to_knee, target_dir, bend_hint)
    if bend_dir is None:
        return

    new_knee = hip + len_upper * (
        np.cos(beta) * target_dir + np.sin(beta) * bend_dir
    )

    # Rotate root so its tail (the knee) lands at new_knee.
    knee_dir_old = hip_to_knee / len_upper
    knee_dir_new = (new_knee - hip) / len_upper
    delta_root_world = _quat_from_to(knee_dir_old, knee_dir_new)
    _add_world_delta(root, delta_root_world)

    # After root rotation, the knee is at new_knee. The mid bone has
    # rotated along with root, so the tip has moved too — recompute
    # where it is now before solving for mid's correction.
    tip_offset_after_root = _rotate_vec_by_quat(delta_root_world, tip - knee)
    tip_dir_old = tip_offset_after_root / len_lower
    knee_to_target = target - new_knee
    knee_to_target_norm = float(np.linalg.norm(knee_to_target))
    if knee_to_target_norm < _EPSILON:
        return
    tip_dir_new = knee_to_target / knee_to_target_norm
    delta_mid_world = _quat_from_to(tip_dir_old, tip_dir_new)
    _add_world_delta(mid, delta_mid_world)


def align_bone_axis_to_world(node: Node, local_axis: Vec3, world_target: Vec3) -> None:
    """Rotate ``node`` so its local-frame ``local_axis`` points along ``world_target``.

    Used as the post-step after a hand- or foot-IK solve to flatten the
    palm / sole onto the ground plane: the IK puts the end joint AT
    the target position, this rotates the end bone around itself so
    the contact surface lies parallel to the world XZ plane (with the
    contact normal along world ``-Y``). Pass ``world_target = (0, -1, 0)``
    for sole-down or palm-down.

    Reads the node's current world rotation, computes where its local
    axis currently points in world space, then composes the shortest-arc
    quaternion that maps THAT direction onto ``world_target`` and applies
    it via :func:`_add_world_delta` — preserves the bone's parent-relative
    twist around the axis (only the axis ORIENTATION is corrected).
    """
    cur_world = _world_rotation(node)
    cur_axis_world = _rotate_vec_by_quat(cur_world, np.asarray(local_axis, dtype=np.float64))
    target = np.asarray(world_target, dtype=np.float64)
    target_norm = float(np.linalg.norm(target))
    if target_norm < _EPSILON:
        return
    target = target / target_norm
    cur_norm = float(np.linalg.norm(cur_axis_world))
    if cur_norm < _EPSILON:
        return
    cur_axis_world = cur_axis_world / cur_norm
    correction = _quat_from_to(cur_axis_world, target)
    _add_world_delta(node, correction)


def _resolve_bend_direction(
    hip_to_knee: np.ndarray,
    target_dir: np.ndarray,
    bend_hint: Vec3 | None,
) -> np.ndarray | None:
    """Pick a unit vector for the knee bend plane perpendicular to ``target_dir``.

    Priority: (1) ``bend_hint`` if provided — caller knows which way
    the joint should fold (knees forward, elbows back, etc.); preferring
    it over the current knee position fixes hyperextended poses where
    a previous IK pass converged with the knee on the wrong side.
    (2) The current knee-side direction as a fallback when no hint is
    given. (3) Any perpendicular axis as last resort.

    Returns ``None`` only on the pathological case where all three
    candidate vectors are degenerate.
    """
    if bend_hint is not None:
        hint = np.asarray(bend_hint, dtype=np.float64).reshape(3)
        hint_perp = hint - float(np.dot(hint, target_dir)) * target_dir
        hp_norm = float(np.linalg.norm(hint_perp))
        if hp_norm >= _EPSILON:
            return hint_perp / hp_norm
    side = hip_to_knee - float(np.dot(hip_to_knee, target_dir)) * target_dir
    side_norm = float(np.linalg.norm(side))
    if side_norm >= _EPSILON:
        return side / side_norm
    ortho = (
        np.array([0.0, 1.0, 0.0])
        if abs(target_dir[1]) < _PARALLEL_AXIS_THRESHOLD
        else np.array([1.0, 0.0, 0.0])
    )
    side = np.cross(target_dir, ortho)
    side_norm = float(np.linalg.norm(side))
    if side_norm < _EPSILON:
        return None
    return side / side_norm


def _quat_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion that rotates unit vector ``a`` to unit ``b``.

    Direct half-angle construction: ``q = (a × b, 1 + a·b)`` then
    normalised. Returns identity when ``a`` and ``b`` are aligned, and
    a 180° rotation around an arbitrary perpendicular when opposed.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cos_a = float(np.dot(a, b))
    if cos_a > 1.0 - _EPSILON:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if cos_a < -1.0 + _EPSILON:
        # Opposed vectors: pick any perpendicular axis for the 180° spin.
        ortho = (
            np.array([1.0, 0.0, 0.0])
            if abs(a[0]) < _PARALLEL_AXIS_THRESHOLD
            else np.array([0.0, 1.0, 0.0])
        )
        axis = np.cross(a, ortho)
        axis = axis / float(np.linalg.norm(axis))
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float32)
    cross = np.cross(a, b)
    w = 1.0 + cos_a
    q = np.array([cross[0], cross[1], cross[2], w], dtype=np.float64)
    q = q / float(np.linalg.norm(q))
    return q.astype(np.float32, copy=False)


def _rotate_vec_by_quat(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply unit quaternion ``q`` to vector ``v`` via ``v' = q v q⁻¹``."""
    qx, qy, qz, qw = (float(c) for c in q)
    qv = np.array([qx, qy, qz], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def _add_world_delta(node: Node, delta_world: np.ndarray) -> None:
    """Compose ``delta_world`` ON TOP OF ``node``'s current local rotation.

    Differs from ``_set_bone``-style overwrite (which sets local =
    delta * rest) — analytical IK accumulates rotations across the
    two bones, so we keep whatever the gait already authored on this
    bone and stack the corrective rotation on top of it.
    """
    parent = node.parent
    if parent is None:
        delta_local = delta_world
    else:
        parent_rot = _world_rotation(parent)
        parent_inv = _quat_inverse(parent_rot)
        delta_local = _quat_mul(_quat_mul(parent_inv, delta_world), parent_rot)
    new_local = _quat_mul(delta_local, node.transform.rotation)
    node.transform.set_rotation(new_local.astype(np.float32, copy=False))


def _world_rotation(node: Node) -> np.ndarray:
    """Compose parent-chain quaternions to get ``node``'s world rotation."""
    chain = []
    cur: Node | None = node
    while cur is not None:
        chain.append(cur.transform.rotation)
        cur = cur.parent
    pw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    for r in reversed(chain):
        pw = _quat_mul(pw, r)
    return pw


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product (XYZW). quat_mul(a, b) means ``a`` applied AFTER ``b``."""
    ax, ay, az, aw = (float(c) for c in a)
    bx, by, bz, bw = (float(c) for c in b)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def solve_two_bone(
    root: Node,
    mid: Node,
    end: Node,
    target_world: Vec3,
    iterations: int = 8,
    step_radian: float = 0.6,
    mid_limit_min: tuple[float, float, float] | None = None,
    mid_limit_max: tuple[float, float, float] | None = None,
) -> None:
    """Place ``end``'s world origin at ``target_world`` by rotating ``root`` and ``mid``.

    Drives the existing CCD core. ``mid_limit_*`` (XYZ Euler radians)
    clamp ``mid``'s local rotation each step — set them to constrain a
    knee or elbow to a single hinge axis. Pass ``None`` for no clamp.

    Prefer :func:`solve_two_bone_analytic` for foot planting and
    other "match this exact target" scenarios — it converges in
    constant time and always returns the closest reachable pose. CCD
    is kept here for backwards compat and for chains where the joint
    has a hard hinge-axis constraint that an analytic solve can't
    express.
    """
    target_arr = np.asarray(target_world, dtype=np.float32).reshape(3)
    driver = Node(name="_ik_target_virtual")
    driver.transform.set_translation(target_arr)

    has_limit = mid_limit_min is not None or mid_limit_max is not None
    mid_link = IkLink(
        bone_index=1,
        has_limit=has_limit,
        limit_min=tuple(mid_limit_min or (0.0, 0.0, 0.0)),
        limit_max=tuple(mid_limit_max or (0.0, 0.0, 0.0)),
    )
    root_link = IkLink(bone_index=0)
    chain = IkChain(
        driver_bone_index=3,
        effector_bone_index=2,
        iterations=int(iterations),
        limit_radian=float(step_radian),
        links=(mid_link, root_link),
    )
    lookup = {0: root, 1: mid, 2: end, 3: driver}
    solve_chain(chain, lookup)


# Keep ``compose_trs`` importable via this module — IK tests sometimes
# build synthetic scenes without going through the importer.
__all__ = [
    "IkChain",
    "IkLink",
    "align_bone_axis_to_world",
    "compose_trs",
    "solve_chain",
    "solve_two_bone",
    "solve_two_bone_analytic",
]
