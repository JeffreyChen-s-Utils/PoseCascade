"""Semi-implicit Euler rigid-body simulator with 6DOF spring joints.

The world advances in fixed sub-steps so a frame's ``apply(dt)`` call
that's longer than the configured tick splits into multiple integration
passes — keeps numeric drift bounded and makes joint stiffness work
predictably across frame rates.

Per body the integrator stores ``position`` / ``orientation`` /
``linear_velocity`` / ``angular_velocity``. Each step:

1. Snapshot kinematic / hybrid bodies from their bone Nodes' world
   matrices.
2. For every dynamic / dynamic-bone body, apply gravity + damping.
3. For every joint, walk the 6DOF axes and add a Hookean spring force
   pushing the linear / angular violation back inside its range.
4. Integrate position / orientation.
5. Write each body's pose back to the corresponding bone (rotation only
   for dynamic-bone, full TRS for dynamic).

Frame-jump support: :meth:`PhysicsWorld.reset` snaps every dynamic body
to its kinematic-pose baseline and runs an optional warm-up sequence so
the user doesn't see a discrete jump after scrubbing the timeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from posecascade.physics.collision import (
    Contact,
    find_contacts,
    resolve_contacts,
)
from posecascade.physics.types import (
    Joint6DofSpring,
    PhysicsMode,
    PhysicsScene,
    RigidBody,
)
from posecascade.utils.math3d import (
    Quat,
    Vec3,
    quat_conjugate,
    quat_exp_map,
    quat_from_euler_xyz,
    quat_identity,
    quat_mul,
    quat_normalize,
    quat_rotate_vec,
    quat_to_euler_xyz,
    vec3,
)

if TYPE_CHECKING:
    from posecascade.scene.node import Node

# Fixed integration tick — 60 Hz matches the standard MMD physics rate
# and keeps the spring stiffness numbers in PMX models in their familiar
# "feels right at this dt" range.
DEFAULT_TICK_SECONDS = 1.0 / 60.0
# Beyond this the apply() loop assumes the user scrubbed the timeline
# and resets dynamic bodies to their kinematic baseline before stepping.
SCRUB_DT_CAP_SECONDS = 0.5
# Default gentle MMD-flavour gravity. Engine users tweak this through
# :meth:`PhysicsWorld.set_gravity`.
DEFAULT_GRAVITY = (0.0, -9.8, 0.0)
_EPSILON = 1.0e-6


@dataclass
class _BodyState:
    """Live integration state for one rigid body."""

    body: RigidBody
    position: NDArray[np.float32]
    orientation: NDArray[np.float32]   # quaternion (x, y, z, w)
    linear_velocity: NDArray[np.float32]
    angular_velocity: NDArray[np.float32]
    rest_position: NDArray[np.float32]
    rest_orientation: NDArray[np.float32]
    bone_node: Node | None = None


@dataclass
class PhysicsWorld:
    """The integrator. One per :class:`PhysicsScene`."""

    scene: PhysicsScene
    bone_index_to_node: dict[int, Node] = field(default_factory=dict)
    tick_seconds: float = DEFAULT_TICK_SECONDS
    gravity: tuple[float, float, float] = DEFAULT_GRAVITY
    _states: list[_BodyState] = field(default_factory=list, init=False)
    _accumulator: float = field(default=0.0, init=False)
    # Last-tick contact list, exposed for tests + debug overlays. Populated
    # at the end of every ``_step_once`` so a caller can introspect what
    # the broadphase + narrowphase produced for the most recent frame.
    _last_contacts: list[Contact] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._build_states()

    def set_gravity(self, gravity: tuple[float, float, float]) -> None:
        self.gravity = gravity

    def reset(self, *, warmup_steps: int = 0) -> None:
        """Snap every body back to its kinematic baseline + step ``warmup_steps`` times."""
        for state in self._states:
            np.copyto(state.position, state.rest_position)
            np.copyto(state.orientation, state.rest_orientation)
            state.linear_velocity[:] = 0.0
            state.angular_velocity[:] = 0.0
        for _ in range(max(warmup_steps, 0)):
            self._step_once(self.tick_seconds)
        self._accumulator = 0.0

    def step(self, dt: float) -> None:
        """Advance the simulation by ``dt`` seconds (sub-stepped at ``tick_seconds``)."""
        if dt <= 0.0:
            return
        if dt > SCRUB_DT_CAP_SECONDS:
            self.reset(warmup_steps=0)
            return
        self._accumulator += dt
        while self._accumulator >= self.tick_seconds:
            self._step_once(self.tick_seconds)
            self._accumulator -= self.tick_seconds

    # ----- public state inspection (used by tests / debug overlays) -----
    def body_count(self) -> int:
        return len(self._states)

    def body_position(self, index: int) -> Vec3:
        return self._states[index].position.copy()

    def body_orientation(self, index: int) -> Quat:
        return self._states[index].orientation.copy()

    def body_linear_velocity(self, index: int) -> Vec3:
        return self._states[index].linear_velocity.copy()

    def last_contacts(self) -> tuple[Contact, ...]:
        """Return the contacts produced by the most recent integration tick."""
        return tuple(self._last_contacts)

    # ----- internal -----------------------------------------------------
    def _build_states(self) -> None:
        for body in self.scene.bodies:
            position = vec3(*body.position)
            orientation = quat_from_euler_xyz(*body.rotation)
            self._states.append(
                _BodyState(
                    body=body,
                    position=position.copy(),
                    orientation=orientation.copy(),
                    linear_velocity=vec3(0.0, 0.0, 0.0),
                    angular_velocity=vec3(0.0, 0.0, 0.0),
                    rest_position=position.copy(),
                    rest_orientation=orientation.copy(),
                    bone_node=self.bone_index_to_node.get(body.bone_index),
                )
            )

    def _step_once(self, dt: float) -> None:
        self._sync_kinematic_from_bones(dt)
        self._apply_gravity_and_damping(dt)
        self._apply_joint_forces(dt)
        self._integrate(dt)
        self._resolve_collisions()
        self._write_back_to_bones()

    def _resolve_collisions(self) -> None:
        """Detect + resolve rigid-rigid contacts after integration."""
        contacts = find_contacts(self._states)
        resolve_contacts(self._states, contacts)
        self._last_contacts = contacts

    def _sync_kinematic_from_bones(self, dt: float) -> None:
        """Pull bone-driven pose into kinematic / dynamic-bone bodies.

        For kinematic bodies, the body simply *is* the bone's pose; the
        integrator's velocities are derived from finite-differencing the
        previous step so the joint solver can compute realistic relative
        velocities to other (dynamic) bodies.
        """
        for state in self._states:
            if state.bone_node is None:
                continue
            mode = state.body.physics_mode
            if mode == PhysicsMode.DYNAMIC:
                continue
            new_position, new_orientation = _bone_world_pose(state)
            if dt > _EPSILON:
                state.linear_velocity = (
                    (new_position - state.position) / dt
                ).astype(np.float32, copy=False)
                state.angular_velocity = _angular_velocity_from_quats(
                    state.orientation, new_orientation, dt,
                )
            np.copyto(state.position, new_position)
            np.copyto(state.orientation, new_orientation)

    def _apply_gravity_and_damping(self, dt: float) -> None:
        gravity = np.asarray(self.gravity, dtype=np.float32)
        for state in self._states:
            if state.body.physics_mode == PhysicsMode.KINEMATIC:
                continue
            state.linear_velocity += gravity * dt
            state.linear_velocity *= max(0.0, 1.0 - state.body.linear_damping * dt)
            state.angular_velocity *= max(0.0, 1.0 - state.body.angular_damping * dt)

    def _apply_joint_forces(self, dt: float) -> None:
        for joint in self.scene.joints:
            if not (
                0 <= joint.rigid_a_index < len(self._states)
                and 0 <= joint.rigid_b_index < len(self._states)
            ):
                continue
            self._apply_one_joint(joint, dt)

    def _apply_one_joint(self, joint: Joint6DofSpring, dt: float) -> None:
        a = self._states[joint.rigid_a_index]
        b = self._states[joint.rigid_b_index]
        # Linear spring along the joint anchor direction. We pin to the
        # joint's rest world position rather than computing a full
        # reciprocal-mass impulse; for the chain-of-bodies use case this
        # is what the user perceives as "the body's anchored to the
        # bone above it".
        anchor = np.asarray(joint.position, dtype=np.float32)
        for state, _partner in ((a, b), (b, a)):
            if state.body.physics_mode == PhysicsMode.KINEMATIC:
                continue
            displacement = state.position - anchor
            for axis in range(3):
                lower = joint.linear_lower[axis]
                upper = joint.linear_upper[axis]
                stiffness = joint.spring_linear[axis]
                if stiffness <= 0.0:
                    continue
                violation = _range_violation(displacement[axis], lower, upper)
                if violation == 0.0:
                    continue
                state.linear_velocity[axis] -= stiffness * violation * dt
        # Angular spring: constrain B's relative rotation to A's frame.
        relative_rot = quat_mul(quat_conjugate(a.orientation), b.orientation)
        rx, ry, rz = quat_to_euler_xyz(relative_rot)
        for axis, current in enumerate((rx, ry, rz)):
            lower = joint.angular_lower[axis]
            upper = joint.angular_upper[axis]
            stiffness = joint.spring_angular[axis]
            if stiffness <= 0.0:
                continue
            violation = _range_violation(current, lower, upper)
            if violation == 0.0:
                continue
            torque_axis = np.zeros(3, dtype=np.float32)
            torque_axis[axis] = -stiffness * violation * dt
            if b.body.physics_mode != PhysicsMode.KINEMATIC:
                b.angular_velocity += torque_axis
            if a.body.physics_mode != PhysicsMode.KINEMATIC:
                a.angular_velocity -= torque_axis

    def _integrate(self, dt: float) -> None:
        for state in self._states:
            if state.body.physics_mode == PhysicsMode.KINEMATIC:
                continue
            state.position = (
                state.position + state.linear_velocity * dt
            ).astype(np.float32, copy=False)
            omega_dt = state.angular_velocity * dt
            state.orientation = quat_normalize(
                quat_mul(quat_exp_map(omega_dt), state.orientation),
            ).astype(np.float32, copy=False)
            if (
                state.body.physics_mode == PhysicsMode.DYNAMIC_BONE
                and state.bone_node is not None
            ):
                # Hybrid mode: bone owns the position; physics owns the
                # rotation. Snap the body back to the bone's translation
                # so accumulated drift doesn't pile up.
                bone_position, _ = _bone_world_pose(state)
                state.position = bone_position.astype(np.float32, copy=False)
                state.linear_velocity[:] = 0.0

    def _write_back_to_bones(self) -> None:
        for state in self._states:
            node = state.bone_node
            if node is None:
                continue
            mode = state.body.physics_mode
            if mode == PhysicsMode.DYNAMIC:
                # Free body — owns both translation + rotation.
                _write_world_to_local(node, state.position, state.orientation)
            elif mode == PhysicsMode.DYNAMIC_BONE:
                # PMX hybrid: physics drives ROTATION only; the bone keeps
                # whatever translation came in from VMD / morph / IK /
                # bone resolver up the pipeline.
                _write_world_rotation_to_local(node, state.orientation)


def _range_violation(value: float, lower: float, upper: float) -> float:
    """Return how far ``value`` is outside ``[lower, upper]`` (signed)."""
    if value < lower:
        return float(value - lower)
    if value > upper:
        return float(value - upper)
    return 0.0


def _bone_world_pose(state: _BodyState) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compose the bone Node's world matrix and pull TRS out as (pos, quat)."""
    node = state.bone_node
    if node is None:
        return state.position.copy(), state.orientation.copy()
    matrix = node.transform.to_matrix()
    parent = node.parent
    while parent is not None:
        matrix = parent.transform.to_matrix() @ matrix
        parent = parent.parent
    position = np.array(
        [matrix[0, 3], matrix[1, 3], matrix[2, 3]], dtype=np.float32,
    )
    orientation = _matrix_to_quat(matrix[:3, :3].astype(np.float32, copy=False))
    return position, orientation


def _matrix_to_quat(rot: NDArray[np.float32]) -> Quat:
    """Convert a 3×3 rotation block to a unit quaternion (Shepperd's method)."""
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        s = 0.5 / float(np.sqrt(trace + 1.0))
        return np.array(
            [
                (rot[2, 1] - rot[1, 2]) * s,
                (rot[0, 2] - rot[2, 0]) * s,
                (rot[1, 0] - rot[0, 1]) * s,
                0.25 / s,
            ],
            dtype=np.float32,
        )
    return _matrix_to_quat_off_diagonal(rot)


def _matrix_to_quat_off_diagonal(rot: NDArray[np.float32]) -> Quat:
    if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]))
        return np.array(
            [
                0.25 * s,
                (rot[0, 1] + rot[1, 0]) / s,
                (rot[0, 2] + rot[2, 0]) / s,
                (rot[2, 1] - rot[1, 2]) / s,
            ],
            dtype=np.float32,
        )
    if rot[1, 1] > rot[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]))
        return np.array(
            [
                (rot[0, 1] + rot[1, 0]) / s,
                0.25 * s,
                (rot[1, 2] + rot[2, 1]) / s,
                (rot[0, 2] - rot[2, 0]) / s,
            ],
            dtype=np.float32,
        )
    s = 2.0 * float(np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]))
    return np.array(
        [
            (rot[0, 2] + rot[2, 0]) / s,
            (rot[1, 2] + rot[2, 1]) / s,
            0.25 * s,
            (rot[1, 0] - rot[0, 1]) / s,
        ],
        dtype=np.float32,
    )


def _angular_velocity_from_quats(
    previous: Quat, current: Quat, dt: float,
) -> NDArray[np.float32]:
    """Approximate the angular velocity that takes ``previous`` to ``current`` in ``dt``."""
    delta = quat_mul(current, quat_conjugate(previous))
    qx, qy, qz, qw = (float(v) for v in delta)
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    sin_half_sq = qx * qx + qy * qy + qz * qz
    if sin_half_sq < _EPSILON:
        return vec3(0.0, 0.0, 0.0)
    sin_half = float(np.sqrt(sin_half_sq))
    angle = 2.0 * float(np.arctan2(sin_half, qw))
    inv = angle / max(sin_half, _EPSILON) / max(dt, _EPSILON)
    return np.array([qx * inv, qy * inv, qz * inv], dtype=np.float32)


def _write_world_to_local(node: Node, world_position: Vec3, world_orientation: Quat) -> None:
    """Express ``(world_position, world_orientation)`` in the node's parent frame."""
    parent_position, parent_orientation = _parent_world_pose(node)
    inv_parent = quat_conjugate(parent_orientation)
    local_translation = quat_rotate_vec(inv_parent, world_position - parent_position)
    local_rotation = quat_normalize(quat_mul(inv_parent, world_orientation))
    node.transform.set_translation(local_translation.astype(np.float32, copy=False))
    node.transform.set_rotation(local_rotation.astype(np.float32, copy=False))


def _write_world_rotation_to_local(node: Node, world_orientation: Quat) -> None:
    """Hybrid mode: only rotation flows from physics back to the bone."""
    _, parent_orientation = _parent_world_pose(node)
    inv_parent = quat_conjugate(parent_orientation)
    local_rotation = quat_normalize(quat_mul(inv_parent, world_orientation))
    node.transform.set_rotation(local_rotation.astype(np.float32, copy=False))


def _parent_world_pose(node: Node) -> tuple[NDArray[np.float32], Quat]:
    """Walk up the parent chain and return its accumulated (position, quat)."""
    parent = node.parent
    if parent is None:
        return vec3(0.0, 0.0, 0.0), quat_identity()
    matrix = parent.transform.to_matrix()
    grandparent = parent.parent
    while grandparent is not None:
        matrix = grandparent.transform.to_matrix() @ matrix
        grandparent = grandparent.parent
    position = np.array(
        [matrix[0, 3], matrix[1, 3], matrix[2, 3]], dtype=np.float32,
    )
    orientation = _matrix_to_quat(matrix[:3, :3].astype(np.float32, copy=False))
    return position, orientation
