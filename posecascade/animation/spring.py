"""Angular-spring chain physics for hair, ribbons, and other secondary motion.

Includes a chain-detection helper that scans a flat list of bone Nodes for
``<prefix>_<index>`` naming patterns (e.g. ``hair_C_0``..``hair_C_3``) and
returns ready-to-rig :class:`DetectedChain` records — used by the glTF
importer to auto-attach :class:`~posecascade.scene.component.SpringChainComponent`
on load.


Each :class:`SpringJoint` is integrated as an independent angular spring with
inertia: a restoring torque pulls the joint's world rotation toward
``parent_world_rotation ⊗ rest_local_rotation``, damping bleeds angular
velocity, and external forces (gravity, wind, point forces) apply torque about
the joint pivot proportional to bone length. The decoupled per-joint model is
the standard hair/cloth approximation used in real-time engines — cheaper than
full multibody dynamics and stable under semi-implicit Euler with substepping.

Bone length is preserved structurally: each joint stores a constant
``rest_local_position`` (its pivot in parent-local space), so child joints
ride rigidly on their parents without distance-constraint passes.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from posecascade.scene.node import Node
from posecascade.utils.math3d import (
    Quat,
    Vec3,
    decompose_trs,
    quat_exp_map,
    quat_identity,
    quat_inverse,
    quat_mul,
    quat_normalize,
    quat_rotate_vec,
    quat_to_axis_angle,
    vec3,
)

# Stable default substep at typical frame rates (60–120 Hz inputs ⇒ 1–2 substeps).
_DEFAULT_FIXED_DT = 1.0 / 120.0
_DEFAULT_GRAVITY = (0.0, -1.0, 0.0)
_DEFAULT_STIFFNESS = 8.0
_DEFAULT_DAMPING = 1.5
_DEFAULT_INERTIA = 1.0
_NUMERIC_EPSILON = 1.0e-6
# Cap per-substep angular speed so a single huge step (e.g. anchor teleport) cannot
# blow the integration up. 8π rad/s ≈ 4 full turns / second — well past anything
# real hair sees in practice but generous enough not to clamp normal motion.
_MAX_ANGULAR_SPEED = 8.0 * math.pi


class ExternalForce(Protocol):
    """A force evaluated at a world-space point at a given simulation time."""

    def force_at(self, world_position: Vec3, time: float) -> Vec3:  # pragma: no cover - protocol
        ...


@dataclass
class Gravity:
    """Constant force vector. Treat the value as ``mass × g`` — sim never scales by mass."""

    force: Vec3 = field(default_factory=lambda: vec3(*_DEFAULT_GRAVITY))

    def force_at(self, world_position: Vec3, time: float) -> Vec3:
        # Position and time are part of the protocol but unused for a uniform field.
        del world_position, time
        return self.force


@dataclass
class Wind:
    """Directional wind with optional sinusoidal cross-direction turbulence."""

    direction: Vec3 = field(default_factory=lambda: vec3(1.0, 0.0, 0.0))
    speed: float = 1.0
    turbulence_amplitude: float = 0.0
    turbulence_frequency_hz: float = 2.0

    def force_at(self, world_position: Vec3, time: float) -> Vec3:
        del world_position
        base = self.direction * self.speed
        if self.turbulence_amplitude == 0.0:
            return base.astype(np.float32)
        cross = np.cross(self.direction, vec3(0.0, 1.0, 0.0)).astype(np.float32)
        cross_len = float(np.linalg.norm(cross))
        cross = vec3(1.0, 0.0, 0.0) if cross_len < _NUMERIC_EPSILON else cross / cross_len
        jitter = math.sin(time * self.turbulence_frequency_hz * math.tau)
        return (base + cross * self.turbulence_amplitude * jitter).astype(np.float32)


@dataclass
class PointForce:
    """Radial force from ``source`` with linear falloff over ``falloff_distance``."""

    source: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    magnitude: float = 1.0
    falloff_distance: float = 1.0

    def force_at(self, world_position: Vec3, time: float) -> Vec3:
        del time
        delta = world_position - self.source
        dist = float(np.linalg.norm(delta))
        if dist < _NUMERIC_EPSILON or dist >= self.falloff_distance:
            return vec3(0.0, 0.0, 0.0)
        attenuation = 1.0 - dist / self.falloff_distance
        return (delta / dist).astype(np.float32) * self.magnitude * attenuation


@dataclass
class SpringParams:
    """Tunable per-chain parameters. Defaults tuned for short hair-like sway."""

    stiffness: float = _DEFAULT_STIFFNESS
    damping: float = _DEFAULT_DAMPING
    inertia: float = _DEFAULT_INERTIA


@dataclass
class SpringJoint:
    """One joint in a chain. Snapshot rest pose at construction; integrator mutates state.

    ``world_rotation`` and ``world_position`` are integrator scratch space. They are
    set on the first tick from the parent's world transform and the rest pose.
    """

    node: Node
    rest_local_position: Vec3
    rest_local_rotation: Quat
    bone_vector_local: Vec3
    inertia: float = _DEFAULT_INERTIA
    angular_velocity: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    world_rotation: Quat = field(default_factory=quat_identity)
    world_position: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    initialized: bool = False


@dataclass(frozen=True)
class _ParentFrame:
    """World position + rotation of a joint's parent — passed to the per-joint integrator."""

    position: Vec3
    rotation: Quat


@dataclass
class SpringChain:
    """A chain of joints anchored to ``anchor`` and animated by spring physics.

    ``anchor`` is a regular scene Node (e.g. a ``head_anchor`` bone) read each
    frame to derive the chain's base frame. The simulator never writes to it.
    """

    name: str
    anchor: Node
    joints: tuple[SpringJoint, ...]
    stiffness: float = _DEFAULT_STIFFNESS
    damping: float = _DEFAULT_DAMPING
    forces: list[ExternalForce] = field(default_factory=list)
    enabled: bool = True

    @classmethod
    def from_node_chain(
        cls,
        name: str,
        anchor: Node,
        joint_nodes: Sequence[Node],
        *,
        params: SpringParams | None = None,
        forces: Sequence[ExternalForce] = (),
    ) -> SpringChain:
        """Snapshot rest pose from ``joint_nodes`` and build a ready-to-simulate chain."""
        if not joint_nodes:
            raise ValueError(f"chain {name!r} requires at least one joint")
        params = params or SpringParams()
        joints = tuple(
            _build_joint(node, joint_nodes, idx, params.inertia)
            for idx, node in enumerate(joint_nodes)
        )
        return cls(
            name=name,
            anchor=anchor,
            joints=joints,
            stiffness=params.stiffness,
            damping=params.damping,
            forces=list(forces),
        )

    def reset(self) -> None:
        """Forget integrator state. Next ``step`` reinitialises from current rest pose."""
        for joint in self.joints:
            joint.angular_velocity = vec3(0.0, 0.0, 0.0)
            joint.initialized = False
            joint.node.transform.set_rotation(joint.rest_local_rotation.copy())


@dataclass
class SpringSimulator:
    """Drives all :class:`SpringChain`\\s with substepped semi-implicit Euler integration."""

    chains: list[SpringChain] = field(default_factory=list)
    global_forces: list[ExternalForce] = field(default_factory=list)
    fixed_dt: float = _DEFAULT_FIXED_DT
    time: float = 0.0

    def add_chain(self, chain: SpringChain) -> None:
        self.chains.append(chain)

    def add_force(self, force: ExternalForce) -> None:
        self.global_forces.append(force)

    def find_chain(self, name: str) -> SpringChain | None:
        """Return the first chain matching ``name`` (or ``None``)."""
        for chain in self.chains:
            if chain.name == name:
                return chain
        return None

    def step(self, dt: float) -> None:
        """Advance simulation by ``dt`` seconds (substepped to ``fixed_dt``)."""
        if dt <= 0.0:
            return
        remaining = float(dt)
        while remaining > _NUMERIC_EPSILON:
            sub = min(self.fixed_dt, remaining)
            for chain in self.chains:
                if chain.enabled:
                    _integrate_chain(chain, sub, self.time, self.global_forces)
            self.time += sub
            remaining -= sub


def _build_joint(
    node: Node,
    joint_nodes: Sequence[Node],
    index: int,
    inertia: float,
) -> SpringJoint:
    """Snapshot one joint's rest pose and derive its bone vector from neighbours."""
    rest_pos = vec3(*node.transform.translation)
    rest_rot = node.transform.rotation.copy()
    bone_vec = _derive_bone_vector(joint_nodes, index)
    return SpringJoint(
        node=node,
        rest_local_position=rest_pos,
        rest_local_rotation=rest_rot,
        bone_vector_local=bone_vec,
        inertia=inertia,
    )


def _derive_bone_vector(joint_nodes: Sequence[Node], index: int) -> Vec3:
    """Bone direction in the joint's own local frame.

    Internal joints take the next joint's local position as the bone vector.
    The tip joint extends along the previous segment's direction so the
    integrator has a non-zero lever arm for force application.
    """
    if index + 1 < len(joint_nodes):
        return vec3(*joint_nodes[index + 1].transform.translation)
    if index > 0:
        prev_offset = (
            joint_nodes[index].transform.translation
            - joint_nodes[index - 1].transform.translation
        )
        return vec3(*prev_offset)
    # Single-joint chain: pick a small default along +Y so torque math has a lever arm.
    return vec3(0.0, 0.1, 0.0)


def _integrate_chain(
    chain: SpringChain,
    dt: float,
    time: float,
    global_forces: Sequence[ExternalForce],
) -> None:
    """Step every joint in ``chain`` once. Walks root → tip so each joint sees its parent's
    updated world transform."""
    anchor_pos, anchor_rot = node_world_pose(chain.anchor)
    parent = _ParentFrame(position=anchor_pos, rotation=anchor_rot)
    for joint in chain.joints:
        if not joint.initialized:
            _initialize_joint_world(joint, parent)
        _step_joint(joint, parent, chain, time, global_forces, dt)
        parent = _ParentFrame(position=joint.world_position, rotation=joint.world_rotation)


def _initialize_joint_world(joint: SpringJoint, parent: _ParentFrame) -> None:
    """First-tick: place the joint's tracked world rotation/position at its rest pose."""
    joint.world_rotation = quat_normalize(quat_mul(parent.rotation, joint.rest_local_rotation))
    joint.world_position = parent.position + quat_rotate_vec(
        parent.rotation, joint.rest_local_position
    )
    joint.initialized = True


def _step_joint(
    joint: SpringJoint,
    parent: _ParentFrame,
    chain: SpringChain,
    time: float,
    global_forces: Sequence[ExternalForce],
    dt: float,
) -> None:
    """Apply spring + damping + external torques and integrate one joint by ``dt``."""
    pivot = parent.position + quat_rotate_vec(parent.rotation, joint.rest_local_position)
    rest_world = quat_normalize(quat_mul(parent.rotation, joint.rest_local_rotation))

    spring_torque = _spring_restoring_torque(joint.world_rotation, rest_world, chain.stiffness)
    force_torque = _force_torque(joint, pivot, chain.forces, global_forces, time)

    # Implicit damping integration: (I + c·dt)·ω_new = I·ω_old + τ_non_damping·dt.
    # Treating damping implicitly is unconditionally stable for any positive c, so
    # an over-damped chain cannot reverse-amplify when a large external displacement
    # (parent teleport, scripted pose) injects a big spring torque.
    inertia = max(joint.inertia, _NUMERIC_EPSILON)
    non_damping_torque = spring_torque + force_torque
    joint.angular_velocity = (
        inertia * joint.angular_velocity + non_damping_torque * dt
    ) / (inertia + chain.damping * dt)
    joint.angular_velocity = _clamp_angular_speed(joint.angular_velocity, _MAX_ANGULAR_SPEED)

    rotation_step = quat_exp_map(joint.angular_velocity * dt)
    joint.world_rotation = quat_normalize(quat_mul(rotation_step, joint.world_rotation))
    joint.world_position = pivot

    new_local = quat_normalize(quat_mul(quat_inverse(parent.rotation), joint.world_rotation))
    joint.node.transform.set_rotation(new_local)


def _clamp_angular_speed(omega: Vec3, max_speed: float) -> Vec3:
    """Cap the magnitude of an angular velocity vector to ``max_speed``."""
    speed = float(np.linalg.norm(omega))
    if speed <= max_speed:
        return omega
    return (omega / speed * max_speed).astype(np.float32)


def _spring_restoring_torque(current_world: Quat, rest_world: Quat, stiffness: float) -> Vec3:
    """Restoring torque that pulls ``current_world`` rotation back toward ``rest_world``."""
    deviation = quat_mul(current_world, quat_inverse(rest_world))
    axis, angle = quat_to_axis_angle(deviation)
    return -stiffness * angle * axis


def _force_torque(
    joint: SpringJoint,
    pivot: Vec3,
    chain_forces: Sequence[ExternalForce],
    global_forces: Sequence[ExternalForce],
    time: float,
) -> Vec3:
    """Sum chain + global forces at the bone's midpoint and convert to a pivot torque."""
    bone_world = quat_rotate_vec(joint.world_rotation, joint.bone_vector_local)
    bone_center = pivot + bone_world * 0.5
    total_force = vec3(0.0, 0.0, 0.0)
    for force in chain_forces:
        total_force = total_force + force.force_at(bone_center, time)
    for force in global_forces:
        total_force = total_force + force.force_at(bone_center, time)
    return np.cross(bone_world * 0.5, total_force).astype(np.float32)


def node_world_pose(node: Node) -> tuple[Vec3, Quat]:
    """Compose ``node``'s world transform by walking parent chain.

    Returns ``(translation, rotation)``. Scale is dropped because spring physics
    operates on rigid joints — only translation and rotation contribute.
    """
    matrix = node.transform.to_matrix()
    parent = node.parent
    while parent is not None:
        matrix = parent.transform.to_matrix() @ matrix
        parent = parent.parent
    translation, rotation, _scale = decompose_trs(matrix)
    return translation, rotation


# A bone name like ``hair_C_0`` splits into prefix ``hair_C`` and index ``0``.
# The pattern requires at least one non-digit character before the trailing digits
# so anchor names like ``head_anchor`` (no integer suffix) do not match.
_DEFAULT_CHAIN_PATTERN = re.compile(r"^(?P<prefix>.+?)_(?P<index>\d+)$")
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectedChain:
    """A bone chain identified by name pattern, ready for rigging.

    ``anchor`` is the parent of ``joints[0]`` in the scene graph — the static
    bone the chain hangs from. The simulator never animates the anchor.
    """

    name: str
    anchor: Node
    joints: tuple[Node, ...]


# Sensible per-prefix defaults the importer (or any caller) can fall back on. Hair
# uses moderate stiffness; ornaments use stiffer + heavier damping so accessories
# don't flop loosely.
DEFAULT_CHAIN_PROFILES: Mapping[str, SpringParams] = {
    "hair": SpringParams(stiffness=12.0, damping=2.5, inertia=0.05),
    "orn": SpringParams(stiffness=24.0, damping=4.0, inertia=0.05),
}
_FALLBACK_PROFILE = SpringParams()


def detect_chains(
    bones: Sequence[Node],
    *,
    pattern: re.Pattern[str] = _DEFAULT_CHAIN_PATTERN,
) -> list[DetectedChain]:
    """Group bones by ``<prefix>_<index>`` pattern and return validated chains.

    Skips groups whose joints are non-consecutive, lack a shared anchor, or whose
    parent chain in the scene graph does not match the index order. Logs a warning
    for each rejected group so glTF authors can debug their bone naming.
    """
    grouped = _group_bones_by_prefix(bones, pattern)
    chains: list[DetectedChain] = []
    for prefix, indexed in grouped.items():
        chain = _validate_group(prefix, indexed)
        if chain is not None:
            chains.append(chain)
    return chains


def resolve_chain_params(
    chain_name: str,
    profiles: Mapping[str, SpringParams] = DEFAULT_CHAIN_PROFILES,
) -> SpringParams:
    """Pick a :class:`SpringParams` preset for ``chain_name`` by leading-token match.

    Tries the longest matching profile key first (so ``"hair_C"`` picks ``"hair"``
    over an empty fallback). Falls back to default :class:`SpringParams` if nothing
    matches — useful for custom prefixes the engine doesn't know about.
    """
    candidates = sorted(profiles.keys(), key=len, reverse=True)
    for key in candidates:
        if chain_name == key or chain_name.startswith(f"{key}_"):
            return profiles[key]
    return _FALLBACK_PROFILE


def _group_bones_by_prefix(
    bones: Sequence[Node],
    pattern: re.Pattern[str],
) -> dict[str, list[tuple[int, Node]]]:
    """Bucket ``bones`` by name prefix; values are sorted ``(index, node)`` pairs."""
    grouped: dict[str, list[tuple[int, Node]]] = {}
    for bone in bones:
        match = pattern.match(bone.name)
        if match is None:
            continue
        prefix = match.group("prefix")
        index = int(match.group("index"))
        grouped.setdefault(prefix, []).append((index, bone))
    for entries in grouped.values():
        entries.sort(key=lambda pair: pair[0])
    return grouped


def _validate_group(
    prefix: str,
    indexed: Sequence[tuple[int, Node]],
) -> DetectedChain | None:
    """Verify a candidate chain is consecutive and parent-linked. Returns ``None`` on rejection."""
    indices = [pair[0] for pair in indexed]
    if indices != list(range(len(indices))):
        _log.warning("chain %r rejected: indices not 0..N: %s", prefix, indices)
        return None
    joints = tuple(pair[1] for pair in indexed)
    anchor = joints[0].parent
    if anchor is None:
        _log.warning("chain %r rejected: joint 0 (%s) has no parent", prefix, joints[0].name)
        return None
    for idx in range(1, len(joints)):
        if joints[idx].parent is not joints[idx - 1]:
            _log.warning(
                "chain %r rejected: %s.parent is not %s",
                prefix,
                joints[idx].name,
                joints[idx - 1].name,
            )
            return None
    return DetectedChain(name=prefix, anchor=anchor, joints=joints)
