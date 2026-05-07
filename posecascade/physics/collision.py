"""Rigid-vs-rigid collision detection + impulse resolution.

The broadphase is an O(N²) AABB sweep — perfectly adequate for the
20–80 bodies a typical PMX model carries, and trivially deterministic.
PMX models rarely exceed that count; if they ever do, the same
``find_contacts`` API can be re-implemented behind a sweep-and-prune
without touching :mod:`posecascade.physics.world`.

The narrowphase covers every shape pair PMX models actually ship:

- sphere ↔ sphere
- sphere ↔ capsule
- capsule ↔ capsule
- sphere ↔ box (OBB)
- capsule ↔ box (OBB)
- box ↔ box (OBB / SAT)

Each case computes a contact (point, normal, penetration depth) which
is then handed to :func:`resolve_contacts`. The resolver applies a
linear restitution-modulated impulse and a Baumgarte split positional
correction. Kinematic bodies have effectively infinite inertia
(``inv_mass = 0``) so they push but are not pushed; pairs of kinematic
bodies are skipped entirely.

Pair filtering honours PMX's ``group`` / ``non_collision_mask``: a body
in group ``g`` whose ``non_collision_mask`` has bit ``g'`` set never
collides with bodies in group ``g'``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from posecascade.physics.types import (
    PhysicsMode,
    RigidBody,
    RigidShape,
)

if TYPE_CHECKING:
    from posecascade.physics.world import _BodyState

_NarrowphaseFn = Callable[
    [int, int, "_BodyState", "_BodyState"], "Contact | None",
]

_EPSILON = 1.0e-6
# Penetration correction parameters — tuned so resting stacks settle
# without visible jitter and don't over-correct on a single tick.
_POS_CORRECTION_PERCENT = 0.4
_POS_CORRECTION_SLOP = 0.001
_MAX_GROUPS = 16


@dataclass(frozen=True)
class Contact:
    """One rigid-rigid contact resolved per integration tick.

    ``normal`` points from ``a`` to ``b`` (i.e. pushing ``b`` along
    ``+normal`` and ``a`` along ``-normal`` separates them).
    ``depth`` is the (positive) penetration along ``normal``.
    ``point`` is in world space and used by torque-aware solvers (the
    current resolver is linear-only).
    """

    a_index: int
    b_index: int
    normal: NDArray[np.float32]
    depth: float
    point: NDArray[np.float32]


def find_contacts(states: list[_BodyState]) -> list[Contact]:
    """Run broadphase + narrowphase over every unique unordered pair.

    Builds world AABBs once per body, then iterates over the upper
    triangle. Pairs filtered out by PMX masks or non-overlapping AABBs
    are dropped before the per-pair narrowphase dispatch.
    """
    contacts: list[Contact] = []
    aabbs = [_world_aabb(state) for state in states]
    body_count = len(states)
    for i in range(body_count):
        for j in range(i + 1, body_count):
            if not _pair_eligible(states[i], states[j]):
                continue
            if not _aabbs_overlap(aabbs[i], aabbs[j]):
                continue
            contact = _narrowphase(i, j, states[i], states[j])
            if contact is not None:
                contacts.append(contact)
    return contacts


def resolve_contacts(states: list[_BodyState], contacts: list[Contact]) -> None:
    """Apply velocity impulses + positional correction for each contact."""
    for contact in contacts:
        a = states[contact.a_index]
        b = states[contact.b_index]
        inv_a = _inverse_mass(a)
        inv_b = _inverse_mass(b)
        inv_sum = inv_a + inv_b
        if inv_sum < _EPSILON:
            continue
        relative_velocity = b.linear_velocity - a.linear_velocity
        velocity_along_normal = float(np.dot(relative_velocity, contact.normal))
        # Only resolve when the two bodies are approaching.
        if velocity_along_normal < 0.0:
            restitution = max(a.body.restitution, b.body.restitution)
            impulse_magnitude = -(1.0 + restitution) * velocity_along_normal / inv_sum
            impulse = impulse_magnitude * contact.normal
            a.linear_velocity = (
                a.linear_velocity - impulse * inv_a
            ).astype(np.float32, copy=False)
            b.linear_velocity = (
                b.linear_velocity + impulse * inv_b
            ).astype(np.float32, copy=False)
        # Baumgarte split positional correction — keeps bodies from
        # accumulating drift through stacks of contacts.
        penetration = contact.depth - _POS_CORRECTION_SLOP
        if penetration <= 0.0:
            continue
        correction = penetration * _POS_CORRECTION_PERCENT / inv_sum
        delta = correction * contact.normal
        a.position = (a.position - delta * inv_a).astype(np.float32, copy=False)
        b.position = (b.position + delta * inv_b).astype(np.float32, copy=False)


# ----- broadphase -------------------------------------------------------
def _aabbs_overlap(
    a: tuple[NDArray[np.float32], NDArray[np.float32]],
    b: tuple[NDArray[np.float32], NDArray[np.float32]],
) -> bool:
    a_min, a_max = a
    b_min, b_max = b
    return bool(
        a_min[0] <= b_max[0] and a_max[0] >= b_min[0]
        and a_min[1] <= b_max[1] and a_max[1] >= b_min[1]
        and a_min[2] <= b_max[2] and a_max[2] >= b_min[2]
    )


def _world_aabb(
    state: _BodyState,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return a conservative world-space AABB for the body's current pose."""
    body = state.body
    if body.shape == RigidShape.SPHERE:
        radius = float(body.size[0])
        offset = np.array([radius, radius, radius], dtype=np.float32)
        return state.position - offset, state.position + offset
    rotation = _quat_to_matrix3(state.orientation)
    if body.shape == RigidShape.CAPSULE:
        radius = float(body.size[0])
        half_height = float(body.size[1]) * 0.5
        local_endpoints = np.array(
            [[0.0, half_height, 0.0], [0.0, -half_height, 0.0]], dtype=np.float32,
        )
        world_endpoints = (rotation @ local_endpoints.T).T + state.position
        offset = np.array([radius, radius, radius], dtype=np.float32)
        ep_min = np.minimum(world_endpoints[0], world_endpoints[1])
        ep_max = np.maximum(world_endpoints[0], world_endpoints[1])
        return (ep_min - offset).astype(np.float32), (ep_max + offset).astype(np.float32)
    # Box: project the OBB half-extents onto each world axis.
    half = np.asarray(body.size, dtype=np.float32)
    abs_rotation = np.abs(rotation)
    extents = abs_rotation @ half
    return (
        (state.position - extents).astype(np.float32),
        (state.position + extents).astype(np.float32),
    )


# ----- pair filtering ---------------------------------------------------
def _pair_eligible(a: _BodyState, b: _BodyState) -> bool:
    """``True`` when the two bodies are allowed to interact this tick."""
    if (
        a.body.physics_mode == PhysicsMode.KINEMATIC
        and b.body.physics_mode == PhysicsMode.KINEMATIC
    ):
        return False
    if _filtered_by_group(a.body, b.body):
        return False
    return not _filtered_by_group(b.body, a.body)


def _filtered_by_group(self_body: RigidBody, other_body: RigidBody) -> bool:
    """``True`` when ``self_body`` opted out of colliding with ``other_body``'s group."""
    if not 0 <= other_body.group < _MAX_GROUPS:
        return False
    return bool((self_body.non_collision_mask >> other_body.group) & 1)


# ----- narrowphase ------------------------------------------------------
def _narrowphase(
    i: int, j: int, state_a: _BodyState, state_b: _BodyState,
) -> Contact | None:
    """Dispatch to the right shape-pair handler.

    Symmetric pairs (B×A) reuse the A×B handler and flip the resulting
    contact's normal, which keeps the per-pair functions readable
    without doubling the table size.
    """
    direct = _DIRECT_NARROWPHASE.get((state_a.body.shape, state_b.body.shape))
    if direct is not None:
        return direct(i, j, state_a, state_b)
    flipped_handler = _DIRECT_NARROWPHASE.get(
        (state_b.body.shape, state_a.body.shape),
    )
    if flipped_handler is not None:
        contact = flipped_handler(j, i, state_b, state_a)
        return _flip(contact) if contact is not None else None
    return None


def _flip(contact: Contact) -> Contact:
    """Swap which body each side of ``contact`` refers to + invert the normal."""
    return Contact(
        a_index=contact.b_index,
        b_index=contact.a_index,
        normal=(-contact.normal).astype(np.float32, copy=False),
        depth=contact.depth,
        point=contact.point,
    )


def _sphere_sphere(
    i: int, j: int, state_a: _BodyState, state_b: _BodyState,
) -> Contact | None:
    radius_a = float(state_a.body.size[0])
    radius_b = float(state_b.body.size[0])
    delta = state_b.position - state_a.position
    distance = float(np.linalg.norm(delta))
    sum_radii = radius_a + radius_b
    if distance >= sum_radii:
        return None
    if distance < _EPSILON:
        # Coincident centres — push along +Y to break the symmetry.
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        depth = sum_radii
        point = state_a.position
    else:
        normal = (delta / distance).astype(np.float32, copy=False)
        depth = sum_radii - distance
        point = state_a.position + normal * radius_a
    return Contact(i, j, normal, float(depth), point.astype(np.float32, copy=False))


def _sphere_capsule(
    i: int, j: int, state_sphere: _BodyState, state_capsule: _BodyState,
) -> Contact | None:
    sphere_radius = float(state_sphere.body.size[0])
    cap_radius = float(state_capsule.body.size[0])
    seg_a, seg_b = _capsule_segment(state_capsule)
    closest = _closest_point_on_segment(seg_a, seg_b, state_sphere.position)
    delta = closest - state_sphere.position
    distance = float(np.linalg.norm(delta))
    sum_radii = sphere_radius + cap_radius
    if distance >= sum_radii:
        return None
    if distance < _EPSILON:
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        depth = sum_radii
    else:
        normal = (delta / distance).astype(np.float32, copy=False)
        depth = sum_radii - distance
    point = state_sphere.position + normal * sphere_radius
    return Contact(i, j, normal, float(depth), point.astype(np.float32, copy=False))


def _capsule_capsule(
    i: int, j: int, state_a: _BodyState, state_b: _BodyState,
) -> Contact | None:
    radius_a = float(state_a.body.size[0])
    radius_b = float(state_b.body.size[0])
    a0, a1 = _capsule_segment(state_a)
    b0, b1 = _capsule_segment(state_b)
    closest_a, closest_b = _closest_points_on_segments(a0, a1, b0, b1)
    delta = closest_b - closest_a
    distance = float(np.linalg.norm(delta))
    sum_radii = radius_a + radius_b
    if distance >= sum_radii:
        return None
    if distance < _EPSILON:
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        depth = sum_radii
        point = closest_a
    else:
        normal = (delta / distance).astype(np.float32, copy=False)
        depth = sum_radii - distance
        point = closest_a + normal * radius_a
    return Contact(i, j, normal, float(depth), point.astype(np.float32, copy=False))


def _sphere_box(
    i: int, j: int, state_sphere: _BodyState, state_box: _BodyState,
) -> Contact | None:
    radius = float(state_sphere.body.size[0])
    half_extents = np.asarray(state_box.body.size, dtype=np.float32)
    rotation = _quat_to_matrix3(state_box.orientation)
    local_centre = rotation.T @ (state_sphere.position - state_box.position)
    clamped = np.clip(local_centre, -half_extents, half_extents)
    inside = bool(np.all(np.abs(local_centre) <= half_extents + _EPSILON))
    if inside:
        # Sphere centre is inside the box — pick the face it's closest to.
        distances = half_extents - np.abs(local_centre)
        axis = int(np.argmin(distances))
        face_normal = np.zeros(3, dtype=np.float32)
        face_normal[axis] = 1.0 if local_centre[axis] >= 0.0 else -1.0
        depth = float(distances[axis] + radius)
        normal_world = rotation @ face_normal           # box → sphere is opposite
        normal = (-normal_world).astype(np.float32, copy=False)
        contact_point = state_sphere.position + normal * radius
        return Contact(i, j, normal, depth, contact_point.astype(np.float32, copy=False))
    delta_local = local_centre - clamped
    distance = float(np.linalg.norm(delta_local))
    if distance >= radius:
        return None
    normal_local = (delta_local / max(distance, _EPSILON)).astype(np.float32, copy=False)
    # Closest point on the box surface, expressed in world space, is the
    # centre of the contact patch — used as the impulse application point.
    closest_world = state_box.position + rotation @ clamped
    # Sphere → box is +normal_local in box local frame; box → sphere flips it.
    normal_box_to_sphere = (rotation @ normal_local).astype(np.float32, copy=False)
    normal = (-normal_box_to_sphere).astype(np.float32, copy=False)
    depth = radius - distance
    return Contact(i, j, normal, float(depth), closest_world.astype(np.float32, copy=False))


def _capsule_box(
    i: int, j: int, state_capsule: _BodyState, state_box: _BodyState,
) -> Contact | None:
    """Treat the capsule as a swept sphere along its segment.

    We sample the segment's closest approach to the OBB by clamping
    each endpoint into box-local space; if neither endpoint is inside,
    we fall back to the midpoint. This is approximate but matches what
    the deformable PMX accessories actually need (contact between a
    skirt capsule and a torso box).
    """
    radius = float(state_capsule.body.size[0])
    a0, a1 = _capsule_segment(state_capsule)
    midpoint = 0.5 * (a0 + a1)
    rotation = _quat_to_matrix3(state_box.orientation)
    half_extents = np.asarray(state_box.body.size, dtype=np.float32)

    def closest_world_to_box(point: NDArray[np.float32]) -> NDArray[np.float32]:
        local = rotation.T @ (point - state_box.position)
        clamped = np.clip(local, -half_extents, half_extents)
        return state_box.position + rotation @ clamped

    candidates = [a0, midpoint, a1]
    best_index = 0
    best_distance = float("inf")
    best_close: NDArray[np.float32] | None = None
    for cand_index, candidate in enumerate(candidates):
        close = closest_world_to_box(candidate)
        distance = float(np.linalg.norm(candidate - close))
        if distance < best_distance:
            best_distance = distance
            best_index = cand_index
            best_close = close
    if best_close is None:
        return None
    if best_distance >= radius:
        return None
    delta = candidates[best_index] - best_close
    if best_distance < _EPSILON:
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        depth = radius
    else:
        normal = (delta / best_distance).astype(np.float32, copy=False)
        depth = radius - best_distance
    return Contact(i, j, normal, float(depth), best_close.astype(np.float32, copy=False))


def _box_box(
    i: int, j: int, state_a: _BodyState, state_b: _BodyState,
) -> Contact | None:
    """Separating-axis test between two oriented boxes.

    Tests the 15 candidate axes (3 face normals from each box + 9 edge
    cross products) and reports the smallest-overlap axis as the
    contact normal. When two axes are parallel (cross product is zero)
    that axis is skipped.
    """
    half_a = np.asarray(state_a.body.size, dtype=np.float32)
    half_b = np.asarray(state_b.body.size, dtype=np.float32)
    rot_a = _quat_to_matrix3(state_a.orientation)
    rot_b = _quat_to_matrix3(state_b.orientation)
    delta = state_b.position - state_a.position
    axes_a = [rot_a[:, k] for k in range(3)]
    axes_b = [rot_b[:, k] for k in range(3)]
    candidate_axes = list(axes_a) + list(axes_b)
    for axis_a in axes_a:
        for axis_b in axes_b:
            cross = np.cross(axis_a, axis_b)
            if float(np.dot(cross, cross)) < _EPSILON:
                continue
            candidate_axes.append(cross / float(np.linalg.norm(cross)))
    smallest_overlap = float("inf")
    smallest_axis: NDArray[np.float32] | None = None
    for raw_axis in candidate_axes:
        axis = np.asarray(raw_axis, dtype=np.float32)
        norm = float(np.linalg.norm(axis))
        if norm < _EPSILON:
            continue
        axis = axis / norm
        radius_a = sum(
            abs(float(np.dot(ax, axis))) * half_a[k] for k, ax in enumerate(axes_a)
        )
        radius_b = sum(
            abs(float(np.dot(ax, axis))) * half_b[k] for k, ax in enumerate(axes_b)
        )
        distance = abs(float(np.dot(delta, axis)))
        overlap = radius_a + radius_b - distance
        if overlap <= 0.0:
            return None
        if overlap < smallest_overlap:
            smallest_overlap = overlap
            smallest_axis = axis if float(np.dot(delta, axis)) >= 0.0 else -axis
    if smallest_axis is None:
        return None
    contact_point = state_a.position + smallest_axis * (
        float(np.linalg.norm(delta)) * 0.5
    )
    return Contact(
        i, j,
        smallest_axis.astype(np.float32, copy=False),
        float(smallest_overlap),
        contact_point.astype(np.float32, copy=False),
    )


# ----- geometry helpers -------------------------------------------------
def _capsule_segment(
    state: _BodyState,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Capsule segment in world space.

    PMX capsules are oriented along their local Y axis with
    ``size[1]`` total cylinder height; the cap hemispheres sit on top
    of those endpoints, contributing ``size[0]`` (radius) on each end.
    """
    half_height = float(state.body.size[1]) * 0.5
    rotation = _quat_to_matrix3(state.orientation)
    direction = rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return (
        (state.position + direction * half_height).astype(np.float32, copy=False),
        (state.position - direction * half_height).astype(np.float32, copy=False),
    )


def _closest_point_on_segment(
    p0: NDArray[np.float32], p1: NDArray[np.float32], target: NDArray[np.float32],
) -> NDArray[np.float32]:
    direction = p1 - p0
    length_squared = float(np.dot(direction, direction))
    if length_squared < _EPSILON:
        return p0.copy()
    t = float(np.dot(target - p0, direction)) / length_squared
    t = max(0.0, min(1.0, t))
    return (p0 + direction * t).astype(np.float32, copy=False)


def _closest_points_on_segments(
    a0: NDArray[np.float32], a1: NDArray[np.float32],
    b0: NDArray[np.float32], b1: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Closest pair of points on two finite line segments (Eberly 2001)."""
    d1 = a1 - a0
    d2 = b1 - b0
    r = a0 - b0
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    if a < _EPSILON and e < _EPSILON:
        return a0.copy(), b0.copy()
    if a < _EPSILON:
        s = 0.0
        t = max(0.0, min(1.0, f / e))
    else:
        c = float(np.dot(d1, r))
        if e < _EPSILON:
            t = 0.0
            s = max(0.0, min(1.0, -c / a))
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = (
                max(0.0, min(1.0, (b * f - c * e) / denom))
                if denom > _EPSILON
                else 0.0
            )
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))
    return (
        (a0 + d1 * s).astype(np.float32, copy=False),
        (b0 + d2 * t).astype(np.float32, copy=False),
    )


def _quat_to_matrix3(quat: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert a unit quaternion ``[x, y, z, w]`` to a 3×3 rotation matrix."""
    qx, qy, qz, qw = (float(v) for v in quat)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _inverse_mass(state: _BodyState) -> float:
    if state.body.physics_mode == PhysicsMode.KINEMATIC:
        return 0.0
    mass = float(state.body.mass)
    if mass < _EPSILON:
        return 0.0
    return 1.0 / mass


# Dispatch table for the narrowphase. Symmetric pairs (B×A) are *not*
# listed — :func:`_narrowphase` looks up the (A, B) variant, swaps the
# arguments, and flips the contact normal afterwards.
_DIRECT_NARROWPHASE: dict[tuple[RigidShape, RigidShape], _NarrowphaseFn] = {
    (RigidShape.SPHERE, RigidShape.SPHERE): _sphere_sphere,
    (RigidShape.SPHERE, RigidShape.CAPSULE): _sphere_capsule,
    (RigidShape.CAPSULE, RigidShape.CAPSULE): _capsule_capsule,
    (RigidShape.SPHERE, RigidShape.BOX): _sphere_box,
    (RigidShape.CAPSULE, RigidShape.BOX): _capsule_box,
    (RigidShape.BOX, RigidShape.BOX): _box_box,
}
