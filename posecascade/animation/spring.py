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

from posecascade.animation.cloth import CapsuleCollider, SphereCollider
from posecascade.scene.node import Node
from posecascade.utils.math3d import (
    Quat,
    Vec3,
    decompose_trs,
    quat_exp_map,
    quat_from_axis_angle,
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
# Passes of collider push-out per substep. 2 is the sweet spot — handles a hair
# tip sandwiched between two adjacent colliders (chest + shoulder, say) with
# one corrective pass per collider direction. Past 2 we hit diminishing returns
# since the swing rotation is exact for one collider per pass. Dropping from 3
# to 2 cut the mesh-collider per-frame cost ~33% on a 30 k-triangle body
# without visible clip-through regression (3-collider sandwich cases are
# uncommon outside of stacked-sphere torsos that we don't ship by default).
_COLLIDER_PROJECTION_PASSES = 2

# Fractions along the bone (from pivot toward tip) sampled against
# colliders each push-out pass. Tip-only sampling misses the common
# "long bone passes THROUGH a body capsule but ends outside it" case
# — e.g. a back-hair strand anchored above the chest with its tip past
# the chest's lower edge. Mid + 3/4 + tip catches that with one extra
# sample test per substep without changing rotation math (swing still
# pivots around the bone's hip, computed so the sample-point lands on
# the collider surface). Order matters: the TIP is tested FIRST so the
# common "tip inside collider" case (e.g. short hair joints sitting in
# the head capsule) gets the original direct-projection behaviour that
# always converges in one pass. Intermediate samples are then tested
# from tip toward pivot so a bone passing THROUGH a capsule still gets
# rotated out when its tip already happens to lie outside.
_BONE_SAMPLE_FRACTIONS: tuple[tuple[float, int], ...] = (
    (1.0, 100),
    (0.75, 75),
    (0.5, 50),
    (0.25, 25),
)


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
    """Tunable per-chain parameters. Defaults tuned for short hair-like sway.

    ``max_swing_rad`` is the angular-cone constraint — every joint's
    rotation is hard-clamped so it can never deviate from its rest
    pose by more than this many radians. This is the same primitive
    that HSR / Genshin / Magica Cloth call ``Angle Limit``: it stops
    hair from physically passing through the head / body in extreme
    poses by making certain joint orientations geometrically
    unreachable, no matter how strong the wind or gravity. Defaults
    to ``π`` (no limit); typical hair values are ``0.8 – 1.3 rad``
    (45 – 75°).
    """

    stiffness: float = _DEFAULT_STIFFNESS
    damping: float = _DEFAULT_DAMPING
    inertia: float = _DEFAULT_INERTIA
    max_swing_rad: float = math.pi


@dataclass
class SpringJoint:
    """One joint in a chain. Snapshot rest pose at construction; integrator mutates state.

    ``world_rotation`` and ``world_position`` are integrator scratch space. They are
    set on the first tick from the parent's world transform and the rest pose.

    ``rest_in_anchor_frame`` is the joint's rest orientation relative to
    the chain anchor, computed at construction as the cumulative product
    of all parent joints' rest local rotations × this joint's rest
    local rotation. The angular-cone constraint clamps the joint's
    world rotation against ``anchor.rotation × rest_in_anchor_frame`` so
    the limit follows the head as the head rotates BUT never compounds
    across joints — every joint must stay within ``max_swing_rad`` of
    its initial rest orientation in the anchor's frame, not of its
    live parent's frame. This matches the HSR / Magica Cloth angle-limit
    semantics where a 55° cone really means "55° from rest", not
    "55° × N joints stacked".
    """

    node: Node
    rest_local_position: Vec3
    rest_local_rotation: Quat
    bone_vector_local: Vec3
    rest_in_anchor_frame: Quat = field(default_factory=quat_identity)
    inertia: float = _DEFAULT_INERTIA
    angular_velocity: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    world_rotation: Quat = field(default_factory=quat_identity)
    world_position: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    initialized: bool = False
    # Per-joint mass. Forces (gravity, wind) divide by mass when
    # computing torque, so heavier joints accelerate less. Magica's
    # "Mass" per-particle. Default 1.0 = uniform, the historical
    # behaviour. Mass <= 0 is treated as 1 (avoid divide-by-zero).
    mass: float = 1.0


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
    max_swing_rad: float = math.pi
    # Per-chain gravity override. When non-None, replaces the global
    # gravity force ONLY for this chain — used by pose-specific
    # animations where world-down gravity would drape the hair into
    # the body (dog crawl, lying poses, hand-stand, …). HSR / Genshin
    # animations carry a "gravity_dir" tag for exactly this purpose.
    # World-space vector; the spring solver applies this directly as
    # a uniform force on every joint, the same way global gravity
    # would be applied.
    gravity_override: Vec3 | None = None
    # Static-drape mode: when ``True`` (and ``gravity_override`` is set),
    # the chain skips spring integration entirely and just sets every
    # joint's world rotation to align its bone vector with the override
    # direction. This is the same "scripted hair pose" mode HSR /
    # Genshin / Magica Cloth use for cutscenes and extreme stances —
    # physics simulation can't reliably wrap a multi-joint chain
    # around a complex body shape (dog-crawl, prone, hand-stand), so
    # the author provides the drape direction and the engine just
    # places joints along it. Reliable, deterministic, no oscillation,
    # no through-body clipping.
    static_drape: bool = False
    # Self-collision radius. When > 0 the chain participates in the
    # post-integrator self-collision pass: every joint world position is
    # treated as a sphere of this radius and projected apart from every
    # other self-colliding joint (within this chain and across chains).
    # Magica Cloth equivalent: "Particle Radius" + "Self Collision". Use
    # ~0.5× the bone length for typical hair strands. Default 0 = off
    # (cheaper, matches earlier behaviour for chains that don't visibly
    # cross).
    self_collision_radius: float = 0.0
    # Tether constraint: cap the parent→joint world distance at
    # ``tether_max_stretch × rest_bone_length``. Applied after the spring
    # step so an overstretched chain (e.g. when the anchor accelerates
    # hard) snaps back to a bounded reach instead of trailing behind
    # forever. ``1.0`` = rigid (no stretch), ``1.05`` = up to 5%
    # over rest (Magica default). ``0.0`` disables the constraint.
    tether_max_stretch: float = 0.0
    # Per-chain stiffness gradient. When ``stiffness_tip`` is non-None,
    # each joint's effective stiffness is ``lerp(stiffness, stiffness_tip,
    # joint_index / (n_joints - 1))``. Magica style: stiff near the
    # root (follows anchor tightly), soft at the tip (free swing). When
    # None the chain uses ``stiffness`` uniformly across joints — earlier
    # behaviour.
    stiffness_tip: float | None = None
    # Per-chain air drag. Quadratic drag on the joint's tip-side angular
    # velocity: torque -= air_drag × ω × |ω|. Distinct from ``damping``,
    # which is a linear coefficient; air drag dominates at high angular
    # speeds (whip motion) the way real air resistance behaves. 0 = off.
    air_drag: float = 0.0
    # Anchor angular-momentum carry. When > 0 the chain inherits a
    # scaled portion of the anchor's per-step angular velocity, so a
    # head turn imparts a swing to the chain instead of dragging it
    # rigidly. 0 = off (default — backwards-compatible), 1 = full
    # inheritance (chain swings as if anchor had momentum).
    inertia_carry: float = 0.0
    # Local-space simulation. When True, the spring step subtracts the
    # anchor's frame-to-frame translation from each joint before
    # integrating, so character motion does not jitter the chain. The
    # translation is re-applied at the end of the step. Matches Magica's
    # "Animation Space = Local" option.
    local_space: bool = False
    # Attract-to-surface: list of collider ``bone_tag`` strings whose
    # surfaces this chain's joint tips should be pulled ONTO (not pushed
    # AWAY from). Used for "lay-on-body" effects — hair drapes onto the
    # back of a bent character, cape rests on shoulders, etc. The attract
    # pass runs AFTER the standard push-out, so a chain still can't get
    # stuck inside the body; it can only land on the named surfaces.
    # Each joint's tip projects to the closest point on any qualifying
    # collider within :attr:`attract_max_distance`. Empty tuple = no
    # attract (default — push-out only).
    attract_to_bones: tuple[str, ...] = ()
    # Maximum distance from joint tip to collider surface beyond which
    # the attract pass does NOT pull. Prevents a chain from being yanked
    # toward a distant collider when the user only wants it to drape
    # onto a nearby surface. 0.30 m = 30 cm — typical reach for a hair-
    # on-back scenario.
    attract_max_distance: float = 0.30
    forces: list[ExternalForce] = field(default_factory=list)
    enabled: bool = True
    # When ``True`` the CPU :func:`_integrate_chain` skips this chain —
    # the renderer's GPU compute dispatcher owns its integration + collider
    # push-out. Set via :meth:`SpringSimulator.mark_gpu_managed`. The
    # GPU path writes the new local rotations back to ``joint.node.transform``
    # itself, so the rest of the engine sees identical effects either way.
    gpu_managed: bool = False
    # When ``True`` the simulator's :func:`_integrate_chain` SKIPS this
    # chain — no spring step, no collider push-out, no attract pass.
    # The joints keep whatever world transforms they had at the time
    # ``frozen`` flipped to True. Used by the per-pose drape snapshot
    # to lock chain pose at the authored drape — same intent HoYoverse
    # uses for *Honkai: Star Rail* / *Genshin* cutscene cloth locks.
    # Distinct from :attr:`enabled` because the renderer still picks
    # up the joint rotations via ``node.transform`` (which we already
    # pushed in ``restore_chain_state``) — disabling would still hide
    # the chain's effect from any caller that filters by ``enabled``.
    frozen: bool = False

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
        # Precompute each joint's rest orientation in the anchor's frame
        # — the cumulative product of all parent joints' rest local
        # rotations × this joint's own. The cone clamp uses this so its
        # constraint follows the anchor (head) but never compounds
        # across the chain (joint[K] cannot piggyback on joint[K-1]'s
        # already-rotated frame to drift far from rest).
        cumulative = quat_identity()
        for joint in joints:
            cumulative = quat_normalize(quat_mul(cumulative, joint.rest_local_rotation))
            joint.rest_in_anchor_frame = cumulative
        return cls(
            name=name,
            anchor=anchor,
            joints=joints,
            stiffness=params.stiffness,
            damping=params.damping,
            max_swing_rad=params.max_swing_rad,
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
    # Body-collider list. After each joint's angular step, the integrator
    # rotates the bone around its pivot so the tip is pushed to the surface
    # of any penetrating sphere / capsule. Shared by reference with the
    # cloth host's collider list, so a single bone-follow driver update
    # benefits both systems and hair no longer clips through the dress /
    # body when the character poses deeply.
    colliders: list[SphereCollider | CapsuleCollider] = field(default_factory=list)
    fixed_dt: float = _DEFAULT_FIXED_DT
    time: float = 0.0
    # Local-space sim & inertia carry need anchor pose from the PREVIOUS
    # frame to compute deltas. Keyed by id(chain).
    _prev_anchor_pose: dict[int, tuple[Vec3, Quat]] = field(default_factory=dict)

    def add_chain(self, chain: SpringChain) -> None:
        self.chains.append(chain)

    def add_force(self, force: ExternalForce) -> None:
        self.global_forces.append(force)

    def add_collider(self, collider: SphereCollider | CapsuleCollider) -> None:
        """Register one body collider for hair-vs-body push-out (see :attr:`colliders`)."""
        self.colliders.append(collider)

    def find_chain(self, name: str) -> SpringChain | None:
        """Return the first chain matching ``name`` (or ``None``)."""
        for chain in self.chains:
            if chain.name == name:
                return chain
        return None

    def step(self, dt: float) -> None:  # noqa: PLR0912  # 12-branch limit too tight for combined per-frame setup + substep + post-step passes; splitting into helpers would just thread more parameters
        """Advance simulation by ``dt`` seconds (substepped to ``fixed_dt``).

        Anchor world poses are snapshotted once at the start of the step
        and reused across every substep — the anchor is a static bone
        for the duration of one user-frame, so re-walking its parent
        chain every substep is pure waste. On a 3-chain rig with 2
        substeps per frame this drops the per-frame anchor-walk count
        from 6 to 3.
        """
        if dt <= 0.0:
            return
        anchor_poses: dict[int, tuple[Vec3, Quat]] = {}
        # Per-chain anchor translation delta for local-space sim & per-chain
        # angular-velocity carry for inertia. Computed once per user-frame
        # from the previous tick's snapshot.
        anchor_translation_delta: dict[int, Vec3] = {}
        anchor_angular_velocity: dict[int, Vec3] = {}
        for chain in self.chains:
            if chain.frozen or not (chain.enabled and not chain.gpu_managed):
                continue
            cur = node_world_pose(chain.anchor)
            anchor_poses[id(chain)] = cur
            prev = self._prev_anchor_pose.get(id(chain))
            if prev is not None and dt > _NUMERIC_EPSILON:
                anchor_translation_delta[id(chain)] = np.asarray(
                    cur[0], dtype=np.float32,
                ) - np.asarray(prev[0], dtype=np.float32)
                # Angular delta as axis-angle / dt.
                delta_q = quat_normalize(
                    quat_mul(cur[1], quat_inverse(prev[1])),
                )
                axis, angle = quat_to_axis_angle(delta_q)
                anchor_angular_velocity[id(chain)] = np.asarray(
                    axis, dtype=np.float32,
                ) * (angle / dt)
            self._prev_anchor_pose[id(chain)] = cur
        # Local-space simulation: subtract the anchor's frame delta from
        # each joint's world_position BEFORE the substep loop. Re-applied
        # in a post-pass below. Reduces jitter when the character runs at
        # high speed — only relative motion drives the spring.
        for chain in self.chains:
            if chain.frozen or not chain.local_space or not chain.enabled or chain.gpu_managed:
                continue
            delta = anchor_translation_delta.get(id(chain))
            if delta is None:
                continue
            for j in chain.joints:
                if j.initialized:
                    j.world_position = (
                        np.asarray(j.world_position, dtype=np.float32) - delta
                    ).astype(np.float32)
        remaining = float(dt)
        while remaining > _NUMERIC_EPSILON:
            sub = min(self.fixed_dt, remaining)
            for chain in self.chains:
                if chain.frozen or not chain.enabled or chain.gpu_managed:
                    # GPU-managed chains run their integration in the
                    # renderer's compute dispatcher. The CPU path leaves
                    # them alone to avoid double-stepping. Frozen chains
                    # (typically locked by a drape snapshot) skip all
                    # integration so the authored pose holds.
                    continue
                _integrate_chain(
                    chain, sub, self.time, self.global_forces,
                    anchor_pose=anchor_poses[id(chain)],
                    colliders=self.colliders,
                    anchor_angular_velocity=anchor_angular_velocity.get(id(chain)),
                )
            # Post-integrator passes: tether & self-collision. Run once
            # per substep after every chain has stepped, so cross-chain
            # interactions are consistent.
            for chain in self.chains:
                if chain.frozen or not chain.enabled or chain.gpu_managed:
                    continue
                if chain.tether_max_stretch > 1.0 + _NUMERIC_EPSILON:
                    _apply_tether_constraint(chain, anchor_poses[id(chain)])
            if any(
                c.enabled and not c.gpu_managed and c.self_collision_radius > _NUMERIC_EPSILON
                for c in self.chains
            ):
                _apply_self_collision(self.chains)
            self.time += sub
            remaining -= sub

    def snapshot_chain_state(self) -> dict[str, list[tuple[float, float, float, float, float, float, float]]]:  # noqa: E501
        """Capture every joint's world pose for pre-sim pose baking.

        Returns ``{chain_name: [(px, py, pz, qx, qy, qz, qw), ...]}``. Pass
        the dict back to :meth:`restore_chain_state` to skip a warmup at
        load time — the chain comes up already settled. Magica Cloth's
        "Pre-Simulation" feature.
        """
        snap: dict[str, list[tuple[float, float, float, float, float, float, float]]] = {}
        for chain in self.chains:
            joints_state = []
            for j in chain.joints:
                p = np.asarray(j.world_position, dtype=np.float32)
                q = np.asarray(j.world_rotation, dtype=np.float32)
                joints_state.append((
                    float(p[0]), float(p[1]), float(p[2]),
                    float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                ))
            snap[chain.name] = joints_state
        return snap

    def restore_chain_state(
        self,
        snap: Mapping[str, Sequence[tuple[float, float, float, float, float, float, float]]],
    ) -> None:
        """Apply a snapshot from :meth:`snapshot_chain_state` to every matching chain.

        Only the spring-state scratch (``world_position`` / ``world_rotation``)
        is restored. The next simulator step propagates these back into each
        joint's ``node.transform`` via the standard projection pipeline, so
        the snapshot must be loaded with ``freeze=False`` in the drape
        snapshot apply (the default) for the visible bones to update —
        a single tick after restore is enough.
        """
        for chain in self.chains:
            joints_state = snap.get(chain.name)
            if joints_state is None or len(joints_state) != len(chain.joints):
                continue
            for j, packed in zip(chain.joints, joints_state, strict=True):
                px, py, pz, qx, qy, qz, qw = packed
                j.world_position = vec3(px, py, pz)
                j.world_rotation = np.asarray([qx, qy, qz, qw], dtype=np.float32)
                j.initialized = True

    def mark_gpu_managed(self, chain: SpringChain) -> None:
        """Flip a chain to GPU-managed mode. CPU integration becomes a
        no-op for it; the renderer's hair compute dispatcher takes over.
        """
        chain.gpu_managed = True

    def gpu_managed_chains(self) -> list[SpringChain]:
        """Return chains currently owned by the GPU compute path."""
        return [c for c in self.chains if c.gpu_managed]


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
    *,
    anchor_pose: tuple[Vec3, Quat] | None = None,
    colliders: Sequence[SphereCollider | CapsuleCollider] = (),
    anchor_angular_velocity: Vec3 | None = None,
) -> None:
    """Step every joint in ``chain`` once. Walks root → tip so each joint sees its parent's
    updated world transform."""
    if anchor_pose is None:
        anchor_pos, anchor_rot = node_world_pose(chain.anchor)
    else:
        anchor_pos, anchor_rot = anchor_pose
    # Inertia carry: imprint the anchor's angular velocity onto every
    # joint's angular_velocity before integration, scaled by
    # ``chain.inertia_carry``. The joint then feels the anchor's spin
    # at the next damping/spring step. Used so head-turn motions
    # translate into hair-swing instead of rigid follow. Gated on
    # inertia_carry > 0 so default chains pay no cost.
    if (
        chain.inertia_carry > _NUMERIC_EPSILON
        and anchor_angular_velocity is not None
        and chain.joints
    ):
        carry = np.asarray(anchor_angular_velocity, dtype=np.float32) * float(
            chain.inertia_carry,
        )
        for j in chain.joints:
            if j.initialized:
                j.angular_velocity = (
                    np.asarray(j.angular_velocity, dtype=np.float32) + carry
                ).astype(np.float32)
    parent = _ParentFrame(position=anchor_pos, rotation=anchor_rot)
    # Static-drape fast path. When the chain is flagged for scripted
    # drape, skip physics entirely and place every joint such that its
    # bone vector aligns with ``gravity_override`` in world space. This
    # matches HSR / Genshin cutscene hair behaviour: physics simulation
    # cannot reliably wrap a multi-joint chain around the bent body of
    # a dog-crawl pose, so the author picks a drape direction and the
    # engine just snaps the chain to it.
    if chain.static_drape and chain.gravity_override is not None:
        _apply_static_drape(chain, parent, anchor_rot, colliders=colliders)
        return
    # Filter / replace global gravity if this chain carries an override.
    effective_forces: Sequence[ExternalForce] = global_forces
    if chain.gravity_override is not None:
        non_gravity = [f for f in global_forces if not isinstance(f, Gravity)]
        non_gravity.append(Gravity(force=np.asarray(chain.gravity_override, dtype=np.float32)))
        effective_forces = non_gravity
    for joint in chain.joints:
        if not joint.initialized:
            _initialize_joint_world(joint, parent)
        _step_joint(joint, parent, chain, time, effective_forces, dt, anchor_rot)
        if colliders:
            _project_joint_against_colliders(joint, parent, colliders)
            if chain.attract_to_bones:
                _attract_joint_to_colliders(
                    joint, parent, colliders,
                    chain.attract_to_bones,
                    chain.attract_max_distance,
                )
        parent = _ParentFrame(position=joint.world_position, rotation=joint.world_rotation)


def _count_anchor_joints_inside_colliders(
    chain: SpringChain,
    parent: _ParentFrame,
    drape_dir: Vec3,
    colliders: Sequence[SphereCollider | CapsuleCollider],
) -> int:
    """Return the count of leading chain joints whose drape-position is
    inside any provided collider.

    Walks the chain along ``drape_dir`` joint-by-joint until a joint's
    mid-bone sample is OUTSIDE every collider. Used by
    :func:`_apply_static_drape` to skip collision push-out for the
    joints that geometrically can't escape the body (they're inside
    the chest capsule when the back-hair anchor is on the base of the
    skull and the head is bent forward).

    Returns 0 when no leading joints are penetrating — collision runs
    for the entire chain in the normal case.
    """
    current_pos = parent.position
    current_rot = parent.rotation
    count = 0
    for joint in chain.joints:
        bone_len = float(np.linalg.norm(joint.bone_vector_local))
        if bone_len < _NUMERIC_EPSILON:
            count += 1
            continue
        pivot = current_pos + quat_rotate_vec(current_rot, joint.rest_local_position)
        mid = pivot + drape_dir * (bone_len * 0.5)
        if not _point_is_inside_any_collider(mid, colliders):
            return count
        count += 1
        # Advance the walk position to the joint's tip so the next
        # joint's pivot sits at the correct world position. Rotation
        # stays at parent rot since static_drape doesn't propagate
        # full orientation along the chain.
        current_pos = pivot + drape_dir * bone_len
    return count


def _point_is_inside_any_collider(
    point: Vec3,
    colliders: Sequence[SphereCollider | CapsuleCollider],
) -> bool:
    """True if ``point`` lies inside any sphere or capsule collider's
    (radius + skin_offset) threshold."""
    px = float(point[0])
    py = float(point[1])
    pz = float(point[2])
    for col in colliders:
        if isinstance(col, SphereCollider):
            cx, cy, cz = float(col.center[0]), float(col.center[1]), float(col.center[2])
            dx, dy, dz = px - cx, py - cy, pz - cz
            thresh = float(col.radius) + float(col.skin_offset)
            if dx * dx + dy * dy + dz * dz < thresh * thresh:
                return True
        elif isinstance(col, CapsuleCollider):
            ax, ay, az = float(col.a[0]), float(col.a[1]), float(col.a[2])
            bx, by, bz = float(col.b[0]), float(col.b[1]), float(col.b[2])
            seg_x, seg_y, seg_z = bx - ax, by - ay, bz - az
            seg_len_sq = seg_x * seg_x + seg_y * seg_y + seg_z * seg_z
            if seg_len_sq < _NUMERIC_EPSILON:
                continue
            t = (
                (px - ax) * seg_x + (py - ay) * seg_y + (pz - az) * seg_z
            ) / seg_len_sq
            t = max(0.0, min(1.0, t))
            cx, cy, cz = ax + seg_x * t, ay + seg_y * t, az + seg_z * t
            dx, dy, dz = px - cx, py - cy, pz - cz
            thresh = float(col.radius) + float(col.skin_offset)
            if dx * dx + dy * dy + dz * dz < thresh * thresh:
                return True
    return False


def _apply_tether_constraint(
    chain: SpringChain,
    anchor_pose: tuple[Vec3, Quat],
) -> None:
    """Cap parent→joint distance at ``chain.tether_max_stretch × rest_bone_length``.

    Walks root → tip. Each joint that exceeded the cap is pulled back
    along the bone vector toward its parent until it sits on the boundary.
    Pure positional correction; angular velocity is left untouched so the
    chain still oscillates naturally, just within a bounded reach.
    """
    anchor_pos, _ = anchor_pose
    parent_pos = np.asarray(anchor_pos, dtype=np.float32)
    max_ratio = float(chain.tether_max_stretch)
    for joint in chain.joints:
        if not joint.initialized:
            parent_pos = np.asarray(joint.world_position, dtype=np.float32)
            continue
        bone_len = float(np.linalg.norm(joint.bone_vector_local))
        if bone_len < _NUMERIC_EPSILON:
            parent_pos = np.asarray(joint.world_position, dtype=np.float32)
            continue
        jp = np.asarray(joint.world_position, dtype=np.float32)
        delta = jp - parent_pos
        dist = float(np.linalg.norm(delta))
        max_dist = bone_len * max_ratio
        if dist > max_dist and dist > _NUMERIC_EPSILON:
            jp = parent_pos + delta * (max_dist / dist)
            joint.world_position = jp
        parent_pos = jp


def _apply_self_collision(chains: Sequence[SpringChain]) -> None:
    """Project chain joints apart when within sum of self-collision radii.

    O(N²) pass over every pair of joints whose chain has
    ``self_collision_radius > 0``. Adjacent joints in the SAME chain are
    skipped — their spacing is already governed by rest bone length and
    enforcing a sphere-vs-sphere separation there would fight the chain
    structure. Cross-chain joints and non-adjacent same-chain joints
    DO participate, which is the typical "left strand crosses right
    strand" case we want to fix.

    For Herta's 14-joint rig the per-substep cost is well under a
    millisecond on CPU. Spatial-hash optimisation is straightforward
    when joint counts grow but isn't worth the code complexity at
    current scales.
    """
    items: list[tuple[int, int, float, np.ndarray]] = []
    for ci, c in enumerate(chains):
        if not c.enabled or c.gpu_managed or c.self_collision_radius <= _NUMERIC_EPSILON:
            continue
        for ji, j in enumerate(c.joints):
            if not j.initialized:
                continue
            items.append((
                ci, ji, float(c.self_collision_radius),
                np.asarray(j.world_position, dtype=np.float32).copy(),
            ))
    n = len(items)
    if n < 2:                                                # noqa: PLR2004
        return
    # Accumulate corrections, then apply at the end so the order of
    # iteration doesn't bias the result.
    corrections = [np.zeros(3, dtype=np.float32) for _ in range(n)]
    for i in range(n):
        ci_i, ji_i, r_i, p_i = items[i]
        for k in range(i + 1, n):
            ci_k, ji_k, r_k, p_k = items[k]
            # Skip adjacent joints in the same chain.
            if ci_i == ci_k and abs(ji_i - ji_k) == 1:
                continue
            delta = p_i + corrections[i] - (p_k + corrections[k])
            dist = float(np.linalg.norm(delta))
            min_dist = r_i + r_k
            if dist >= min_dist or dist < _NUMERIC_EPSILON:
                continue
            push = delta * ((min_dist - dist) * 0.5 / dist)
            corrections[i] += push
            corrections[k] -= push
    for idx, (ci, ji, _r, _p) in enumerate(items):
        if np.linalg.norm(corrections[idx]) < _NUMERIC_EPSILON:
            continue
        j = chains[ci].joints[ji]
        new_pos = np.asarray(j.world_position, dtype=np.float32) + corrections[idx]
        j.world_position = new_pos


def _apply_static_drape(
    chain: SpringChain,
    parent: _ParentFrame,
    anchor_rot: Quat,
    colliders: Sequence[SphereCollider | CapsuleCollider] = (),
) -> None:
    """Place every joint along the gravity-override direction.

    Each joint's rotation is computed so that its local bone vector
    lands on the drape direction. No physics, no spring, no inertia,
    no oscillation — the author's authored direction is the law.

    Layered on top is a **collider push-out** pass per joint. The
    author picks the drape direction (e.g. "back along the spine"
    for dog crawl); collision then rotates any joint whose bone
    sample-points are inside a body capsule back out to the surface.
    This is the Magica-Cloth / HSR "scripted hair + collide" model:
    authored direction wins overall, body shape wins locally where
    the chain crosses it. Far cleaner than pure physics for poses
    where the anchor pivot sits inside a body capsule (dog_crawl,
    prone, hand-stand) and no physically-natural drape direction
    reaches the author's intended look.
    """
    g_norm = float(np.linalg.norm(chain.gravity_override))
    if g_norm < _NUMERIC_EPSILON:
        return
    drape_dir = chain.gravity_override / g_norm
    current_parent = parent
    # Implicit floor at world Y=0 — when the drape direction would
    # send a joint's tip below this plane, we bend the bone into a
    # horizontal direction so the tip lays flat on the floor instead
    # of continuing through. Mimics a floor collider for hair without
    # the SpringSimulator needing a plane-collider type — important
    # because the chain has NO floor collision otherwise, so a
    # downward drape direction would extend the tip arbitrarily far
    # below the world plane (visible as "back portion floats in
    # air above ground" once the chain has fully crossed).
    floor_y = 0.0
    # When the chain anchor sits inside a body collider (common for
    # back-hair anchored at the base of the skull while the head is
    # forward-bent in dog-crawl-class poses), the first few joints
    # are GEOMETRICALLY trapped inside the body and any collision
    # push-out would fully override the author's drape direction by
    # routing them around the body. Skip collision for the joints
    # that start inside ANY collider — they're hidden under the head /
    # body mesh anyway, and the rest of the chain (the visible part)
    # still gets collision so it doesn't punch through arms.
    skip_collision_until = _count_anchor_joints_inside_colliders(
        chain, parent, drape_dir, colliders,
    ) if colliders else 0
    for joint_idx, joint in enumerate(chain.joints):
        bone_len = float(np.linalg.norm(joint.bone_vector_local))
        if bone_len < _NUMERIC_EPSILON:
            joint.world_position = current_parent.position
            current_parent = _ParentFrame(
                position=joint.world_position,
                rotation=current_parent.rotation,
            )
            continue
        bone_local_dir = joint.bone_vector_local / bone_len
        # Floor-aware bone direction. The naive drape direction is
        # ``drape_dir``; if applied straight, the next-joint position
        # (pivot + drape_dir * bone_len) would drop below ``floor_y``,
        # tilt the direction so the tip lands exactly at floor_y
        # instead. Keeps tail of the chain on the floor with no
        # plane-collider primitive required. Horizontal component
        # is preserved so the chain still "knows" which way it was
        # draping when it hits the floor.
        pivot = current_parent.position + quat_rotate_vec(
            current_parent.rotation, joint.rest_local_position,
        )
        effective_dir = drape_dir
        projected_tip_y = float(pivot[1]) + float(drape_dir[1]) * bone_len
        if projected_tip_y < floor_y:
            # Need to compute horizontal_extent such that bone vertical
            # = (floor_y - pivot_y) and horizontal² + vertical² = bone_len².
            dy = floor_y - float(pivot[1])
            # Clamp the vertical descent — if pivot is already below
            # floor (shouldn't happen but defensive), aim STRAIGHT
            # horizontal in the drape's XZ direction.
            dy_clamped = max(-bone_len, min(bone_len, dy))
            horiz_sq = bone_len * bone_len - dy_clamped * dy_clamped
            horiz_extent = float(np.sqrt(max(horiz_sq, 0.0)))
            drape_horiz = np.array(
                [float(drape_dir[0]), 0.0, float(drape_dir[2])], dtype=np.float32,
            )
            horiz_norm = float(np.linalg.norm(drape_horiz))
            if horiz_norm > _NUMERIC_EPSILON and bone_len > _NUMERIC_EPSILON:
                drape_horiz = drape_horiz * (horiz_extent / horiz_norm / bone_len)
                effective_dir = np.array(
                    [drape_horiz[0], dy_clamped / bone_len, drape_horiz[2]],
                    dtype=np.float32,
                )
                eff_norm = float(np.linalg.norm(effective_dir))
                if eff_norm > _NUMERIC_EPSILON:
                    effective_dir = effective_dir / eff_norm
        # World rotation = quat that maps bone_local_dir to effective_dir.
        joint.world_rotation = _shortest_arc_quat(bone_local_dir, effective_dir)
        joint.angular_velocity = vec3(0.0, 0.0, 0.0)
        joint.world_position = pivot
        joint.initialized = True
        if colliders and joint_idx >= skip_collision_until:
            _project_joint_against_colliders(
                joint,
                _ParentFrame(
                    position=current_parent.position,
                    rotation=current_parent.rotation,
                ),
                colliders,
            )
        else:
            new_local = quat_normalize(
                quat_mul(quat_inverse(current_parent.rotation), joint.world_rotation),
            )
            joint.node.transform.set_rotation(new_local)
        current_parent = _ParentFrame(
            position=pivot,
            rotation=joint.world_rotation,
        )
    del anchor_rot


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
    anchor_rotation: Quat,
) -> None:
    """Apply spring + damping + external torques and integrate one joint by ``dt``."""
    pivot = parent.position + quat_rotate_vec(parent.rotation, joint.rest_local_position)
    # Resolve effective per-joint stiffness. When the chain defines a
    # ``stiffness_tip``, lerp from root (``chain.stiffness``) to tip by
    # the joint's normalised position along the chain. Magica's typical
    # setup: roots stiff, tips soft.
    if chain.stiffness_tip is not None and len(chain.joints) > 1:
        joint_idx = chain.joints.index(joint)
        t = joint_idx / (len(chain.joints) - 1)
        effective_stiffness = chain.stiffness + (chain.stiffness_tip - chain.stiffness) * t
    else:
        effective_stiffness = chain.stiffness
    # When ``gravity_override`` is set, hijack the rest target so every
    # joint pulls its bone toward the same world direction (the override
    # vector normalised). This is what makes a multi-joint chain settle
    # in a CLEAN line along the drape direction instead of the S-curve
    # produced by chasing per-joint anchor-frame rest poses that were
    # rigged for the standing bind pose. CPU + GPU both honour this so
    # the visible behaviour matches whichever solver is active.
    if chain.gravity_override is not None:
        g_norm = float(np.linalg.norm(chain.gravity_override))
        bone_local_norm = float(np.linalg.norm(joint.bone_vector_local))
        if g_norm > _NUMERIC_EPSILON and bone_local_norm > _NUMERIC_EPSILON:
            target_world_dir = chain.gravity_override / g_norm
            bone_local_dir = joint.bone_vector_local / bone_local_norm
            # ``rest_world`` is the joint's world rotation that maps
            # bone_vector_local to the drape direction in world. Spring
            # restoring + cone clamp both center on this so every joint
            # in the chain pulls toward the SAME world direction — gives
            # a clean line along gravity, not the S-curve produced by
            # the bind-pose-rigged per-joint anchor-frame rest.
            rest_world = _shortest_arc_quat(bone_local_dir, target_world_dir)
        else:
            rest_world = quat_normalize(quat_mul(parent.rotation, joint.rest_local_rotation))
    else:
        rest_world = quat_normalize(quat_mul(parent.rotation, joint.rest_local_rotation))

    spring_torque = _spring_restoring_torque(joint.world_rotation, rest_world, effective_stiffness)
    force_torque = _force_torque(joint, pivot, chain.forces, global_forces, time)
    # Per-joint mass: divide external (gravity/wind/user) force torque
    # by mass. Spring + damping operate on inertia, not mass — they're
    # internal constraint torques, not Newtonian forces. Skip the divide
    # when mass==1.0 (the universal default) so existing call sites that
    # never set per-joint mass observe bit-identical behaviour and a
    # baseline regression test like ``test_collider_push_rotates_joint``
    # — sensitive to a few mm of integrator output — stays green.
    if joint.mass != 1.0:                                                # noqa: PLR2004
        mass = max(joint.mass, _NUMERIC_EPSILON)
        force_torque = force_torque / mass

    # Quadratic air drag: opposes the joint's current angular velocity
    # with a force ~ -k·ω·|ω|. Dominates at high spin (whip motion); the
    # linear ``damping`` term still handles low-speed settling. Drag
    # impulse is CLAMPED so it can only reduce omega magnitude to zero
    # within one substep, never overshoot and reverse the spin
    # direction. Without the clamp, large ``air_drag`` values turn the
    # quadratic term into a destabilising amplifier under explicit Euler.
    if chain.air_drag > _NUMERIC_EPSILON:
        omega = np.asarray(joint.angular_velocity, dtype=np.float32)
        omega_mag = float(np.linalg.norm(omega))
        if omega_mag > _NUMERIC_EPSILON:
            ideal_drag_mag = chain.air_drag * omega_mag * omega_mag
            joint_inertia = max(joint.inertia, _NUMERIC_EPSILON)
            max_drag_mag = joint_inertia * omega_mag / dt
            scale = (
                min(1.0, max_drag_mag / ideal_drag_mag)
                if ideal_drag_mag > _NUMERIC_EPSILON
                else 0.0
            )
            drag_torque = omega * (-chain.air_drag * omega_mag * scale)
            force_torque = force_torque + drag_torque

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

    # Angular-cone constraint. Center: the drape-aligned rest if a
    # ``gravity_override`` is in effect, otherwise the anchor-frame
    # rest pose (see comment above). Using the same center for both
    # spring restoring and cone clamp is what makes the chain settle
    # in a clean line along the drape direction.
    if chain.max_swing_rad < math.pi:
        if chain.gravity_override is None:
            cone_center = quat_normalize(
                quat_mul(anchor_rotation, joint.rest_in_anchor_frame),
            )
        else:
            cone_center = rest_world
        _apply_swing_cone(joint, cone_center, chain.max_swing_rad)

    new_local = quat_normalize(quat_mul(quat_inverse(parent.rotation), joint.world_rotation))
    joint.node.transform.set_rotation(new_local)


def _apply_swing_cone(
    joint: SpringJoint,
    rest_world: Quat,
    max_swing_rad: float,
) -> None:
    """Clamp ``joint.world_rotation`` to within ``max_swing_rad`` of ``rest_world``.

    Computes the swing quaternion ``Δq = q_world ⊗ q_rest⁻¹``,
    decomposes to axis+angle, clamps the angle, rebuilds ``q_world``.
    Any angular velocity component aligned with the clamp axis (i.e.
    pushing further into the wall) is removed so the next sub-step
    doesn't re-accumulate force against the cone — the joint slides
    along the boundary instead of bouncing or stalling.
    """
    delta = quat_normalize(quat_mul(joint.world_rotation, quat_inverse(rest_world)))
    axis, angle = quat_to_axis_angle(delta)
    if angle <= max_swing_rad:
        return
    clamped_delta = quat_from_axis_angle(axis, max_swing_rad)
    joint.world_rotation = quat_normalize(quat_mul(clamped_delta, rest_world))
    # Project out the velocity component along the clamp axis. Angular
    # velocity is in world frame; the axis we just clamped against IS
    # the world-frame swing axis, so the dot product gives the offending
    # component directly.
    omega_along = float(np.dot(joint.angular_velocity, axis))
    if omega_along > 0.0:
        joint.angular_velocity = joint.angular_velocity - axis * omega_along


def _project_joint_against_colliders(
    joint: SpringJoint,
    parent: _ParentFrame,
    colliders: Sequence[SphereCollider | CapsuleCollider],
) -> None:
    """Rotate ``joint`` around its pivot so its bone tip exits every penetrating collider.

    Spring physics is angular — joint position is fixed by the parent
    pivot, so we cannot simply translate the tip out of a sphere. Instead
    we compute the corrective swing rotation that takes the current bone
    direction to a direction whose endpoint lies on the collider surface
    (plus skin offset). Up to ``_COLLIDER_PROJECTION_PASSES`` passes
    handle multi-collider stacks (e.g. a chest + upper-arm capsule).

    The corrective rotation is reflected back into ``joint.node`` so the
    rest of the engine sees the projected pose immediately.
    """
    pivot = parent.position + quat_rotate_vec(parent.rotation, joint.rest_local_position)
    bone_world = quat_rotate_vec(joint.world_rotation, joint.bone_vector_local)
    bone_len = float(np.linalg.norm(bone_world))
    if bone_len < _NUMERIC_EPSILON:
        joint.world_position = pivot
        return
    bone_dir = bone_world / bone_len
    for _ in range(_COLLIDER_PROJECTION_PASSES):
        any_hit = False
        for frac, sample_key in _BONE_SAMPLE_FRACTIONS:
            sample_pt = pivot + bone_dir * (bone_len * frac)
            # Pass the current bone direction as the fallback push direction
            # for any degenerate (sample-at-collider-center) hit. Without
            # this, ``_project_sphere`` defaulted to world +Y which launched
            # any hair joint anchored at the head-sphere centre straight up
            # over the hat.
            target_pt, hit = _push_tip_out_of_colliders(
                sample_pt, colliders, bone_dir, joint=joint,
                sample_key=sample_key,
            )
            if not hit:
                continue
            # The sample at ``frac`` along the bone needs to land at
            # ``target_pt``. Since the joint swings around its pivot, the
            # new bone direction is just (target_pt - pivot) normalised —
            # the sample point along the bone moves with the rotation.
            target_offset = target_pt - pivot
            target_len = float(np.linalg.norm(target_offset))
            if target_len < _NUMERIC_EPSILON:
                continue
            target_dir = target_offset / target_len
            swing = _shortest_arc_quat(bone_dir, target_dir)
            joint.world_rotation = quat_normalize(quat_mul(swing, joint.world_rotation))
            # Project out any angular velocity component along the swing
            # axis — the same trick :func:`_apply_swing_cone` uses for
            # the cone-clamp constraint. Without it, velocity keeps
            # driving the bone INTO the collider next substep, the
            # collision pushes it back, and the chain oscillates around
            # the contact instead of settling. The "hair flies up first
            # then falls" symptom on the first viewport frame after a
            # CPU-warmup settle was this oscillation visualising —
            # warmup ended with the bone touching a capsule and a
            # residual velocity along the push direction, and the GPU
            # dispatch substeps walked that velocity into a visible
            # transient before damping decayed it.
            swing_axis, swing_angle = quat_to_axis_angle(swing)
            if swing_angle > _NUMERIC_EPSILON:
                omega_along = float(np.dot(joint.angular_velocity, swing_axis))
                if omega_along < 0.0:
                    joint.angular_velocity = (
                        joint.angular_velocity - swing_axis * omega_along
                    )
            bone_dir = target_dir
            any_hit = True
            break  # restart the pass with the updated direction
        if not any_hit:
            break
    joint.world_position = pivot
    new_local = quat_normalize(quat_mul(quat_inverse(parent.rotation), joint.world_rotation))
    joint.node.transform.set_rotation(new_local)


def _attract_joint_to_colliders(
    joint: SpringJoint,
    parent: _ParentFrame,
    colliders: Sequence[SphereCollider | CapsuleCollider],
    attract_to_bones: tuple[str, ...],
    attract_max_distance: float,
) -> None:
    """Pull the joint's bone tip onto the nearest matching collider's surface.

    Runs AFTER :func:`_project_joint_against_colliders` — push-out has
    already kicked the tip out of any body capsule it landed inside, so
    this pass only ever moves the tip TOWARD a body surface it's near.
    A chain configured with ``attract_to_bones`` thus drapes onto the
    named surfaces (hair onto back, cape onto shoulders) instead of
    floating at ``radius + skin_offset`` from them.

    Only colliders whose ``bone_tag`` is in ``attract_to_bones`` and
    whose surface lies within ``attract_max_distance`` of the current
    tip are considered. The closest qualifying surface point wins; the
    joint rotates around its pivot so the bone vector points at it.
    """
    pivot = parent.position + quat_rotate_vec(parent.rotation, joint.rest_local_position)
    bone_world = quat_rotate_vec(joint.world_rotation, joint.bone_vector_local)
    bone_len = float(np.linalg.norm(bone_world))
    if bone_len < _NUMERIC_EPSILON:
        return
    bone_dir = bone_world / bone_len
    tip = pivot + bone_world
    best_dist = attract_max_distance
    best_surface_pt: Vec3 | None = None
    for collider in colliders:
        bone_tag = getattr(collider, "bone_tag", "")
        if not bone_tag or bone_tag not in attract_to_bones:
            continue
        surface_pt = _nearest_surface_point(tip, collider)
        if surface_pt is None:
            continue
        d = float(np.linalg.norm(tip - surface_pt))
        if d < best_dist:
            best_dist = d
            best_surface_pt = surface_pt
    if best_surface_pt is None:
        return
    target_offset = best_surface_pt - pivot
    target_len = float(np.linalg.norm(target_offset))
    if target_len < _NUMERIC_EPSILON:
        return
    target_dir = target_offset / target_len
    swing = _shortest_arc_quat(bone_dir, target_dir)
    joint.world_rotation = quat_normalize(quat_mul(swing, joint.world_rotation))
    # Project out any velocity component along the swing axis so the
    # attract motion doesn't bounce back on the next substep.
    swing_axis, swing_angle = quat_to_axis_angle(swing)
    if swing_angle > _NUMERIC_EPSILON:
        omega_along = float(np.dot(joint.angular_velocity, swing_axis))
        if abs(omega_along) > 0.0:
            joint.angular_velocity = joint.angular_velocity - swing_axis * omega_along
    joint.world_position = pivot
    new_local = quat_normalize(quat_mul(quat_inverse(parent.rotation), joint.world_rotation))
    joint.node.transform.set_rotation(new_local)


def _nearest_surface_point(
    point: Vec3,
    collider: SphereCollider | CapsuleCollider,
) -> Vec3 | None:
    """Closest point on ``collider``'s surface to ``point`` (in world space).

    Returns ``None`` for collider types this helper doesn't support (e.g.
    :class:`MeshCollider` / SDFs — those have their own attract paths if
    ever added). The returned point sits at radius + skin_offset from the
    collider's axis so the chain's mesh skin lies just outside the body
    rather than embedded in it.
    """
    if isinstance(collider, SphereCollider):
        delta = np.asarray(point - collider.center, dtype=np.float32)
        d = float(np.linalg.norm(delta))
        r = collider.radius + collider.skin_offset
        if d < _NUMERIC_EPSILON:
            # Degenerate: point at sphere centre. Pick any radial direction;
            # +Y is a reasonable default for body capsules whose long axis
            # is roughly vertical at character scale.
            return collider.center + vec3(0.0, 1.0, 0.0) * r
        return collider.center + (delta / d) * r
    if isinstance(collider, CapsuleCollider):
        ab = np.asarray(collider.b - collider.a, dtype=np.float32)
        ab_len_sq = float(np.dot(ab, ab))
        r = collider.radius + collider.skin_offset
        if ab_len_sq < _NUMERIC_EPSILON:
            # Degenerate capsule = sphere at a
            return _nearest_surface_point(
                point,
                SphereCollider(center=collider.a, radius=collider.radius,
                               skin_offset=collider.skin_offset),
            )
        t = float(np.clip(
            float(np.dot(np.asarray(point - collider.a, dtype=np.float32), ab)) / ab_len_sq,
            0.0, 1.0,
        ))
        axis_pt = collider.a + ab * t
        delta = np.asarray(point - axis_pt, dtype=np.float32)
        d = float(np.linalg.norm(delta))
        if d < _NUMERIC_EPSILON:
            # On the capsule axis — pick any perpendicular. Try world +Y first;
            # if axis is vertical, fall back to +X.
            axis_unit = ab / float(np.sqrt(ab_len_sq))
            up = vec3(0.0, 1.0, 0.0)
            perp = np.cross(axis_unit, up).astype(np.float32)
            perp_len = float(np.linalg.norm(perp))
            if perp_len < _NUMERIC_EPSILON:
                perp = np.cross(axis_unit, vec3(1.0, 0.0, 0.0)).astype(np.float32)
                perp_len = float(np.linalg.norm(perp))
            return axis_pt + (perp / perp_len) * r
        return axis_pt + (delta / d) * r
    return None


def _push_tip_out_of_colliders(
    tip: Vec3,
    colliders: Sequence[SphereCollider | CapsuleCollider],
    fallback_dir: Vec3 | None = None,
    joint: SpringJoint | None = None,
    sample_key: int = 100,
) -> tuple[Vec3, bool]:
    """Return ``(corrected_tip, any_hit)`` after one pass of collider projection.

    ``fallback_dir`` is used when the tip lands exactly at a collider's
    centre (rare but possible for hair bones anchored at the head sphere's
    centre). Without a sensible fallback, the projector defaults to world
    +Y — which launches the hair upward over the head. Passing the bone's
    current direction keeps the fallback aligned with where the bone
    actually points.

    ``joint`` enables a temporal cache on mesh-collider tests: when the
    joint's tip hasn't moved more than 5 mm since the last query AND the
    mesh hasn't re-skinned (version bump), the previous corrected tip +
    hit flag are returned without re-running closest-point. Massive win
    for a stable pose where the spring chain has settled — typical
    hair-joint per-substep movement is sub-millimetre once gravity has
    balanced spring + collider push-out forces.
    """
    from posecascade.animation.cloth import (  # noqa: PLC0415
        MeshCollider,
        SDFCollider,
    )

    out = tip
    any_hit = False
    # Cheap per-sample early-out: extract the point's scalar coords once,
    # then compare against each collider's precomputed AABB in pure
    # Python. Far-away colliders short-circuit before any numpy call,
    # which is the difference between "settle a back-hair chain against
    # one of 22 body capsules" and "do 22 closest-point computations
    # per substep". The AABB itself is stored on the collider by the
    # bone-following update — see :meth:`SphereCollider._refresh_aabb`
    # and :meth:`CapsuleCollider._refresh_aabb`.
    tx = float(out[0])
    ty = float(out[1])
    tz = float(out[2])
    for collider in colliders:
        aabb = getattr(collider, "_cached_aabb", None)
        if aabb is not None:
            mn_x, mn_y, mn_z, mx_x, mx_y, mx_z = aabb
            if tx < mn_x or tx > mx_x or ty < mn_y or ty > mx_y or tz < mn_z or tz > mx_z:
                continue
        if isinstance(collider, SphereCollider):
            out, hit = _project_sphere(out, collider, fallback_dir)
        elif isinstance(collider, CapsuleCollider):
            out, hit = _project_capsule_cached(out, collider, joint, sample_key)
        elif isinstance(collider, MeshCollider):
            out, hit = _project_mesh_cached(out, collider, joint, sample_key)
        elif isinstance(collider, SDFCollider):
            out, hit = _project_sdf(out, collider)
        else:
            continue
        if hit:
            # The corrected point changed — re-snapshot scalar coords so
            # subsequent AABB checks reflect the new position. Without
            # this the next collider would early-out against the OLD
            # sample location and miss a stacked collision.
            tx = float(out[0])
            ty = float(out[1])
            tz = float(out[2])
            any_hit = True
    return out, any_hit


def _project_sdf(point: Vec3, sdf: object) -> tuple[Vec3, bool]:
    """Push ``point`` outside the SDF collider's iso-surface if within
    ``skin_offset``. O(1) — eight voxel reads + trilinear interp + a
    central-difference gradient lookup.

    The SDF stores SIGNED distance: positive = outside, negative = inside.
    Push direction = gradient (∇SDF), normalised. Push magnitude = how
    much further the point needs to move to clear ``skin_offset``.
    """
    grid = sdf.grid
    origin = sdf.grid_origin
    voxel = sdf.voxel_size
    p = np.asarray(point, dtype=np.float32)
    # Convert world → voxel float coordinates.
    fx = (p[0] - origin[0]) / voxel
    fy = (p[1] - origin[1]) / voxel
    fz = (p[2] - origin[2]) / voxel
    nx, ny, nz = grid.shape
    # Clamp inside grid (sample boundary if outside).
    ix = int(np.clip(fx, 0.0, nx - 2.0))
    iy = int(np.clip(fy, 0.0, ny - 2.0))
    iz = int(np.clip(fz, 0.0, nz - 2.0))
    tx = float(np.clip(fx - ix, 0.0, 1.0))
    ty = float(np.clip(fy - iy, 0.0, 1.0))
    tz = float(np.clip(fz - iz, 0.0, 1.0))
    # Trilinear interpolation of the signed distance at the query point.
    c000 = float(grid[ix,     iy,     iz])
    c100 = float(grid[ix + 1, iy,     iz])
    c010 = float(grid[ix,     iy + 1, iz])
    c110 = float(grid[ix + 1, iy + 1, iz])
    c001 = float(grid[ix,     iy,     iz + 1])
    c101 = float(grid[ix + 1, iy,     iz + 1])
    c011 = float(grid[ix,     iy + 1, iz + 1])
    c111 = float(grid[ix + 1, iy + 1, iz + 1])
    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    distance = c0 * (1 - tz) + c1 * tz
    if distance >= sdf.skin_offset:
        return point, False
    # Central-difference gradient (one voxel step in each axis). Cheap
    # and adequate since the SDF is smooth-by-construction at the
    # voxel scale.
    ix1 = min(ix + 1, nx - 1)
    iy1 = min(iy + 1, ny - 1)
    iz1 = min(iz + 1, nz - 1)
    ix0 = max(ix - 1, 0)
    iy0 = max(iy - 1, 0)
    iz0 = max(iz - 1, 0)
    gx = float(grid[ix1, iy, iz] - grid[ix0, iy, iz])
    gy = float(grid[ix, iy1, iz] - grid[ix, iy0, iz])
    gz = float(grid[ix, iy, iz1] - grid[ix, iy, iz0])
    glen = float(np.sqrt(gx * gx + gy * gy + gz * gz))
    if glen < _NUMERIC_EPSILON:
        return point, False
    # Push to skin_offset distance along the outward gradient.
    push = sdf.skin_offset - distance
    out = p + np.asarray([gx, gy, gz], dtype=np.float32) * (push / glen)
    return out.astype(np.float32), True


def _project_capsule_cached(
    point: Vec3,
    capsule: object,
    joint: SpringJoint | None,
    sample_key: int = 100,
) -> tuple[Vec3, bool]:
    """Per-joint temporal cache wrapping :func:`_project_capsule`. Same
    pattern as :func:`_project_mesh_cached` — capsules update their
    endpoints every frame via bone-follow drivers, so cache validity
    requires both joint tip stability AND capsule endpoint stability.
    Cache stores capsule.a/b at the time of last query and invalidates
    when either endpoint has moved more than the tolerance.

    Hot path: cache HITS used to call ``np.linalg.norm`` three times
    per check, each of which allocates a temp array, calls a C
    function, and returns a Python float. That worked out to ~17 µs
    per hit and ~30% of the entire physics frame on a settled chain.
    Squared-distance comparisons via ``v @ v`` are 6× faster (no sqrt,
    no temp scalar) and give identical hit/miss semantics — the
    tolerance is just precomputed as ``tol_sq``.
    """
    if joint is None:
        return _project_capsule(point, capsule)
    # Per-capsule AABB pre-filter: if the sample sits more than
    # ``radius + skin_offset + bone_len_pad`` outside the capsule's
    # axis-aligned bounding box, no closest-point test can hit. Saves
    # the cache lookup AND the projection call on the typical 80-90%
    # of sample-vs-collider pairs that are clearly far apart (hair
    # joint vs the foot capsule, etc.).
    p = np.asarray(point, dtype=np.float32)
    cap_a = capsule.a
    cap_b = capsule.b
    thresh = capsule.radius + capsule.skin_offset
    ax_min = min(float(cap_a[0]), float(cap_b[0])) - thresh
    ax_max = max(float(cap_a[0]), float(cap_b[0])) + thresh
    if float(p[0]) < ax_min or float(p[0]) > ax_max:
        return point, False
    ay_min = min(float(cap_a[1]), float(cap_b[1])) - thresh
    ay_max = max(float(cap_a[1]), float(cap_b[1])) + thresh
    if float(p[1]) < ay_min or float(p[1]) > ay_max:
        return point, False
    az_min = min(float(cap_a[2]), float(cap_b[2])) - thresh
    az_max = max(float(cap_a[2]), float(cap_b[2])) + thresh
    if float(p[2]) < az_min or float(p[2]) > az_max:
        return point, False
    cache = getattr(joint, "_capsule_test_cache", None)
    if cache is None:
        cache = {}
        joint._capsule_test_cache = cache                                    # noqa: SLF001
    cache_id = (id(capsule), sample_key)
    entry = cache.get(cache_id)
    if entry is not None:
        last_pos, last_a, last_b, last_out, last_hit = entry
        tol_sq = _MESH_CACHE_TOLERANCE_M * _MESH_CACHE_TOLERANCE_M
        dp = p - last_pos
        da = cap_a - last_a
        db = cap_b - last_b
        if (
            float(dp @ dp) < tol_sq
            and float(da @ da) < tol_sq
            and float(db @ db) < tol_sq
        ):
            return last_out, last_hit
    out, hit = _project_capsule(point, capsule)
    cache[cache_id] = (
        p.copy(),
        np.asarray(capsule.a, dtype=np.float32).copy(),
        np.asarray(capsule.b, dtype=np.float32).copy(),
        out, hit,
    )
    return out, hit


def _project_mesh_cached(
    point: Vec3,
    mesh: object,
    joint: SpringJoint | None,
    sample_key: int = 100,
) -> tuple[Vec3, bool]:
    """Joint-level temporal cache wrapping :func:`_project_mesh`.

    ``sample_key`` namespaces the cache by where along the bone the
    sample lives — multi-sample bone tests (mid + 3/4 + tip) all hit
    the cache instead of overwriting each other.
    """
    if joint is None:
        return _project_mesh(point, mesh)
    cache = getattr(joint, "_mesh_test_cache", None)
    if cache is None:
        cache = {}
        joint._mesh_test_cache = cache                                       # noqa: SLF001
    cache_id = (id(mesh), sample_key)
    entry = cache.get(cache_id)
    p = np.asarray(point, dtype=np.float32)
    if entry is not None:
        last_pos, last_out, last_hit, last_version = entry
        if (
            last_version == mesh.update_version
            and np.linalg.norm(p - last_pos) < _MESH_CACHE_TOLERANCE_M
        ):
            if _DEBUG_MESH_STATS:
                _mesh_stats["cache_hit"] += 1
            return last_out, last_hit
    out, hit = _project_mesh(point, mesh)
    if _DEBUG_MESH_STATS:
        _mesh_stats["cache_miss_hit" if hit else "cache_miss_nohit"] += 1
    cache[cache_id] = (p.copy(), out, hit, mesh.update_version)
    return out, hit


_MESH_CACHE_TOLERANCE_M = 0.005  # 5 mm

# Debug counters — set _DEBUG_MESH_STATS=True to populate at runtime.
_DEBUG_MESH_STATS = False
_mesh_stats = {"cache_hit": 0, "cache_miss_hit": 0, "cache_miss_nohit": 0}


def reset_mesh_stats() -> None:
    """Zero the mesh-collider debug counters."""
    _mesh_stats["cache_hit"] = 0
    _mesh_stats["cache_miss_hit"] = 0
    _mesh_stats["cache_miss_nohit"] = 0


def get_mesh_stats() -> dict:
    """Read current mesh-collider debug counters."""
    return dict(_mesh_stats)


def _project_mesh(point: Vec3, mesh: object) -> tuple[Vec3, bool]:                # noqa: PLR0912
    """Push ``point`` outside the mesh collider's nearest triangle if within
    ``skin_offset``. Pure numpy with per-triangle AABB pre-filter so the
    per-joint cost stays at O(T_in_box) instead of O(T_total) — critical
    for a 30 k-triangle body mesh where checking every triangle every tick
    would burn ~10 ms per substep per joint.
    """
    tris = mesh.triangle_world_positions
    if tris is None or len(tris) == 0:
        return point, False
    margin = mesh.skin_offset + 0.02                                 # noqa: PLR2004
    p = np.asarray(point, dtype=np.float32)
    # Whole-mesh AABB early out.
    if (
        mesh.aabb_min is not None
        and mesh.aabb_max is not None
        and (
            p[0] < mesh.aabb_min[0] - margin
            or p[0] > mesh.aabb_max[0] + margin
            or p[1] < mesh.aabb_min[1] - margin
            or p[1] > mesh.aabb_max[1] + margin
            or p[2] < mesh.aabb_min[2] - margin
            or p[2] > mesh.aabb_max[2] + margin
        )
    ):
        return point, False
    # Spatial-grid lookup — O(1) avg vs O(T) for the AABB-mask approach.
    # Each cell holds the triangle indices whose AABB overlaps it; we
    # gather indices from the point's cell + 1-cell neighbourhood, then
    # run closest-point on just that subset. Uses numpy concatenate
    # (not python set) so the hot path stays in C.
    if mesh.grid_cells is not None and mesh.grid_cell_size > 0:
        origin = mesh.grid_origin
        cell_size = mesh.grid_cell_size
        inv_cell = 1.0 / cell_size
        cx = int((p[0] - origin[0]) * inv_cell)
        cy = int((p[1] - origin[1]) * inv_cell)
        cz = int((p[2] - origin[2]) * inv_cell)
        grid = mesh.grid_cells
        # 1-cell neighbour walk — cell_size matches skin_offset so the
        # 3³ cube around the query point covers the entire push-out search
        # radius without needing a 2-cell walk.
        cell_arrays = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cell = grid.get((cx + dx, cy + dy, cz + dz))
                    if cell is not None:
                        cell_arrays.append(cell)
        if not cell_arrays:
            return point, False
        # Concatenate then unique (sorted) — both numpy C ops, much faster
        # than per-cell python set unions for the typical 10-100 candidates.
        idx_arr = np.unique(np.concatenate(cell_arrays))
        tris = tris[idx_arr]
        normals_arr = mesh.triangle_world_normals[idx_arr]
    elif mesh.triangle_aabb_min is not None:
        tri_min = mesh.triangle_aabb_min
        tri_max = mesh.triangle_aabb_max
        overlap = (
            (p[0] >= tri_min[:, 0] - margin)
            & (p[0] <= tri_max[:, 0] + margin)
            & (p[1] >= tri_min[:, 1] - margin)
            & (p[1] <= tri_max[:, 1] + margin)
            & (p[2] >= tri_min[:, 2] - margin)
            & (p[2] <= tri_max[:, 2] + margin)
        )
        if not overlap.any():
            return point, False
        tris = tris[overlap]
        normals_arr = mesh.triangle_world_normals[overlap]
    else:
        normals_arr = mesh.triangle_world_normals
    v0 = tris[:, 0]
    v1 = tris[:, 1]
    v2 = tris[:, 2]
    closest = _closest_point_on_triangles(p, v0, v1, v2)
    diffs = closest - p
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    nearest_dist = float(dists[idx])
    if nearest_dist >= mesh.skin_offset:
        return point, False
    normal = normals_arr[idx]
    signed = float(np.dot(p - closest[idx], normal))
    if signed < 0:
        normal = -normal
    pushed = closest[idx] + normal * mesh.skin_offset
    return pushed.astype(np.float32), True


def _closest_point_on_triangles(
    p: object, v0: object, v1: object, v2: object,
) -> object:
    """Per-triangle closest point computation, vectorised over all triangles.

    Standard barycentric-coordinate solver: clamp the projection of ``p`` onto
    the triangle's plane into the triangle's interior, falling back to the
    nearest edge / vertex when the projection lies outside.
    """
    edge0 = v1 - v0
    edge1 = v2 - v0
    v0_to_p = p - v0
    a = np.einsum("tj,tj->t", edge0, edge0)
    b = np.einsum("tj,tj->t", edge0, edge1)
    c = np.einsum("tj,tj->t", edge1, edge1)
    d = np.einsum("tj,tj->t", edge0, v0_to_p)
    e = np.einsum("tj,tj->t", edge1, v0_to_p)
    det = a * c - b * b
    # Avoid divide-by-zero for degenerate triangles (collapse to v0).
    safe_det = np.where(np.abs(det) < 1e-10, 1.0, det)               # noqa: PLR2004
    s = (b * e - c * d) / safe_det
    t = (b * d - a * e) / safe_det
    # Clamp barycentrics into the triangle (s>=0, t>=0, s+t<=1).
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    over = s + t > 1.0
    if np.any(over):
        # Compute renormalisation scale only where s+t > 1 (where the
        # projection landed outside the triangle and needs the s+t=1
        # edge constraint). Using ``np.where`` for the denominator
        # avoids the divide-by-zero warning that would fire for
        # triangles where s+t == 0 — those triangles are NOT in the
        # ``over`` mask so their s/t stay untouched, but numpy still
        # evaluates the division across the full array before
        # masking.
        sum_st = s + t
        scale = np.where(over, 1.0 / np.where(over, sum_st, 1.0), 1.0)
        s = np.where(over, s * scale, s)
        t = np.where(over, t * scale, t)
    return v0 + edge0 * s[:, None] + edge1 * t[:, None]


def _project_sphere(
    point: Vec3,
    sphere: SphereCollider,
    fallback_dir: Vec3 | None = None,
) -> tuple[Vec3, bool]:
    """Push ``point`` to the sphere surface (+ skin offset) when inside; else return unchanged.

    When ``point`` coincides with the sphere's centre (degenerate), uses
    ``fallback_dir`` (a unit-length-ish vector) to choose which way to
    push. Falls back to world +Y only when no direction is supplied.
    """
    delta = point - sphere.center
    dist = float(np.linalg.norm(delta))
    threshold = sphere.radius + sphere.skin_offset
    if dist >= threshold:
        return point, False
    if dist < _NUMERIC_EPSILON:
        if fallback_dir is not None:
            dir_arr = np.asarray(fallback_dir, dtype=np.float32)
            dir_len = float(np.linalg.norm(dir_arr))
            if dir_len > _NUMERIC_EPSILON:
                push = (dir_arr / dir_len) * threshold
                return (sphere.center + push).astype(np.float32), True
        # No usable fallback — default to +Y as a last resort.
        return (sphere.center + vec3(0.0, threshold, 0.0)).astype(np.float32), True
    return (sphere.center + delta * (threshold / dist)).astype(np.float32), True


def _project_capsule(point: Vec3, capsule: CapsuleCollider) -> tuple[Vec3, bool]:
    """Push ``point`` to the capsule surface when inside; else return unchanged."""
    segment = capsule.b - capsule.a
    seg_len_sq = float(np.dot(segment, segment))
    if seg_len_sq < _NUMERIC_EPSILON:
        return _project_sphere(point, SphereCollider(
            center=capsule.a, radius=capsule.radius, skin_offset=capsule.skin_offset,
        ))
    t = float(np.dot(point - capsule.a, segment) / seg_len_sq)
    t_clamped = max(0.0, min(1.0, t))
    closest = capsule.a + segment * t_clamped
    delta = point - closest
    dist = float(np.linalg.norm(delta))
    threshold = capsule.radius + capsule.skin_offset
    if dist >= threshold:
        return point, False
    if dist < _NUMERIC_EPSILON:
        # Tip on the axis — push perpendicular to the segment along world up.
        return (closest + vec3(0.0, threshold, 0.0)).astype(np.float32), True
    return (closest + delta * (threshold / dist)).astype(np.float32), True


def _shortest_arc_quat(from_dir: Vec3, to_dir: Vec3) -> Quat:
    """Shortest-arc quaternion mapping unit vector ``from_dir`` to unit vector ``to_dir``."""
    dot = float(np.clip(np.dot(from_dir, to_dir), -1.0, 1.0))
    if dot > 1.0 - _NUMERIC_EPSILON:
        return quat_identity()
    if dot < -1.0 + _NUMERIC_EPSILON:
        # 180° flip — pick any axis perpendicular to from_dir.
        axis = np.cross(from_dir, vec3(1.0, 0.0, 0.0))
        if float(np.linalg.norm(axis)) < _NUMERIC_EPSILON:
            axis = np.cross(from_dir, vec3(0.0, 1.0, 0.0))
        axis = axis / float(np.linalg.norm(axis))
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float32)
    axis = np.cross(from_dir, to_dir)
    w = 1.0 + dot
    q = np.array([axis[0], axis[1], axis[2], w], dtype=np.float32)
    return quat_normalize(q)


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
#
# The underscore is OPTIONAL so MMD-style names ship out of the box:
# ``前髪01`` → prefix ``前髪`` / index ``1``, ``リボン2`` → prefix ``リボン`` / index ``2``.
# Non-greedy ``.+?`` + the anchored ``\d+`` makes the regex backtrack from
# the end, so the underscore in ``hair_C_0`` still belongs to the prefix
# (``hair_C``) rather than being eaten by the optional ``_?``.
#
# The trailing ``(?:_.*)?`` swallows any Maya / HSR-style suffix after
# the chain index — ``HairA_00_JNT_060`` parses as prefix ``HairA`` /
# index ``00`` so March 7th's strands group into chains the same way
# ``hair_C_0..3`` did historically. Without the optional tail, the
# greedy ``\d+`` would lock onto the trailing global joint ID
# (``060``), every strand bone would land in its own prefix bucket,
# and chain detection would emit zero chains on FBX-style rigs.
# ``side`` capture group recognises the standard left/right suffix that HSR
# and similar FBX rigs append AFTER the index (``BackHair1_L_0210``). When
# present, the side is folded into the chain key so left and right strands
# rig as independent chains instead of fighting for the same bucket.
_DEFAULT_CHAIN_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_?(?P<index>\d+)(?:_(?P<side>[LR]))?(?:_.*)?$",
)
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


# Sensible per-prefix defaults the importer (or any caller) can fall back on.
# Keys cover the conventional glTF naming (``hair_C_0..3``) AND the canonical
# MMD bone names (``前髪01..08``, ``リボン1..3``, ``スカート_0_0..3`` …) so PMD /
# PMX imports get the same auto-rig glTF has always had. Add new entries here
# when you encounter a new model whose author tags physics chains with an
# unfamiliar prefix.
DEFAULT_CHAIN_PROFILES: Mapping[str, SpringParams] = {
    # Latin / glTF conventions
    "hair": SpringParams(stiffness=12.0, damping=2.5, inertia=0.05),
    # HSR-style rigs that split back / side / inner hair into separate
    # prefix buckets (``BackHair1_L``, ``BackHairUpper2``, …). Add new
    # variant prefixes as you encounter rigs that use them.
    "BackHair": SpringParams(stiffness=10.0, damping=2.2, inertia=0.05),
    "BackHairUpper": SpringParams(stiffness=14.0, damping=2.7, inertia=0.04),
    "SideHair": SpringParams(stiffness=11.0, damping=2.4, inertia=0.05),
    # Variants seen on HSR-style rigs (e.g. Herta) where Chinese-prefix
    # side hair gets renamed to ASCII at import: "SideHairTwo_L",
    # "SideTwistHair_R", etc. Same physics as base SideHair.
    "SideHairTwo": SpringParams(stiffness=11.0, damping=2.4, inertia=0.05),
    "SideTwistHair": SpringParams(stiffness=11.0, damping=2.4, inertia=0.05),
    "FrontHair": SpringParams(stiffness=12.0, damping=2.5, inertia=0.05),
    "Ponytail": SpringParams(stiffness=10.0, damping=2.2, inertia=0.05),
    "orn": SpringParams(stiffness=24.0, damping=4.0, inertia=0.05),
    "skirt": SpringParams(stiffness=10.0, damping=2.2, inertia=0.04),
    # Re-parented Herta coat/skirt bones — each angular position is
    # chain ``Skirt_<letter>`` (letter a..p for 16 positions around the
    # hem) with up to 13 joints from hips down to floor. Conservative
    # stiffness/damping to keep the 16-chain bundle stable while still
    # allowing the coat to fall naturally in extreme poses.
    "Skirt": SpringParams(stiffness=8.0, damping=2.5, inertia=0.05),
    "ribbon": SpringParams(stiffness=18.0, damping=3.0, inertia=0.03),
    # NOTE: HSR / Aplaybox FBX rigs ship per-strand prefixes (``HairA``,
    # ``skirtA``, ``ribbonA``, ``beltA``, ``earringA``, ``Xiudai``,
    # ``WeaponA``, ``tongue``). Auto-rigging the whole set on import
    # destabilises the cloth/PBD solver on this asset — some strands
    # explode out from the head down past the feet on the first few
    # frames. They are intentionally OMITTED here so March 7th loads
    # statically; scripts that need decorative physics can opt in
    # explicitly by adding a profile via the document-level
    # ``physics_chains`` block (see ``examples/scripts/idle.json``).
    # MMD / PMX / PMD conventions — bones named in Japanese.
    "前髪": SpringParams(stiffness=12.0, damping=2.5, inertia=0.05),   # front hair
    "後髪": SpringParams(stiffness=10.0, damping=2.2, inertia=0.05),   # back hair
    "横髪": SpringParams(stiffness=11.0, damping=2.4, inertia=0.05),   # side hair
    "髪": SpringParams(stiffness=12.0, damping=2.5, inertia=0.05),     # generic hair
    "リボン": SpringParams(stiffness=18.0, damping=3.0, inertia=0.03), # ribbon
    "スカート": SpringParams(stiffness=10.0, damping=2.2, inertia=0.04), # skirt
    "ｽｶｰﾄ": SpringParams(stiffness=10.0, damping=2.2, inertia=0.04),  # half-width skirt
    "袖": SpringParams(stiffness=11.0, damping=2.3, inertia=0.04),    # sleeve
    "尻尾": SpringParams(stiffness=14.0, damping=2.8, inertia=0.05),  # tail
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


def resolve_chain_params_or_none(
    chain_name: str,
    profiles: Mapping[str, SpringParams] = DEFAULT_CHAIN_PROFILES,
) -> SpringParams | None:
    """Strict variant: return ``None`` when no profile prefix matches ``chain_name``.

    Use this in auto-rig paths so detected chains that *look* like
    physics-eligible names but actually aren't (fingers, twist bones,
    PMX append bones, …) get skipped instead of receiving the
    fallback profile and silently swaying at runtime.
    """
    candidates = sorted(profiles.keys(), key=len, reverse=True)
    for key in candidates:
        # Exact match (chain name == profile key) OR underscore-separated
        # token boundary (``hair_C`` starts with ``hair_``) OR digit
        # boundary (``SkirtA00`` starts with ``SkirtA`` followed by a
        # digit). A naked ``startswith(key)`` would over-match
        # (``hairs`` → ``hair``); the boundary guards against that for
        # ASCII names. MMD bones don't use underscore separators, so we
        # accept a bare ``startswith`` for non-ASCII keys (``前髪``
        # matches ``前髪`` only).
        if chain_name == key or chain_name.startswith(f"{key}_"):
            return profiles[key]
        # Digit-boundary form: key followed immediately by a digit. Used
        # by per-angular-position chains like ``SkirtA00`` ↔ ``SkirtA``.
        if (
            chain_name.startswith(key)
            and len(chain_name) > len(key)
            and chain_name[len(key)].isdigit()
        ):
            return profiles[key]
    return None


def resolve_chain_params(
    chain_name: str,
    profiles: Mapping[str, SpringParams] = DEFAULT_CHAIN_PROFILES,
) -> SpringParams:
    """Pick a :class:`SpringParams` preset for ``chain_name`` by leading-token match.

    Tries the longest matching profile key first (so ``"hair_C"`` picks ``"hair"``
    over an empty fallback). Falls back to default :class:`SpringParams` if nothing
    matches — useful for custom prefixes the engine doesn't know about.
    """
    strict = resolve_chain_params_or_none(chain_name, profiles)
    if strict is not None:
        return strict
    return _FALLBACK_PROFILE


def _group_bones_by_prefix(
    bones: Sequence[Node],
    pattern: re.Pattern[str],
) -> dict[str, list[tuple[int, Node]]]:
    """Bucket ``bones`` by name prefix; values are sorted ``(index, node)`` pairs.

    When the pattern captures a ``side`` group (HSR-style ``_L``/``_R``
    suffix after the index), it is folded into the bucket key so left
    and right strands form independent chains. Patterns without a
    ``side`` group fall back to grouping by ``prefix`` alone.
    """
    grouped: dict[str, list[tuple[int, Node]]] = {}
    has_side_group = "side" in pattern.groupindex
    for bone in bones:
        match = pattern.match(bone.name)
        if match is None:
            continue
        prefix = match.group("prefix")
        index = int(match.group("index"))
        if has_side_group:
            side = match.group("side")
            key = f"{prefix}_{side}" if side else prefix
        else:
            key = prefix
        grouped.setdefault(key, []).append((index, bone))
    for entries in grouped.values():
        entries.sort(key=lambda pair: pair[0])
    return grouped


def _validate_group(
    prefix: str,
    indexed: Sequence[tuple[int, Node]],
) -> DetectedChain | None:
    """Verify ``indexed`` is uniformly-spaced and parent-linked. Returns ``None`` on rejection."""
    indices = [pair[0] for pair in indexed]
    # Accept any arithmetic progression with a positive step. glTF authors
    # typically index from 0 with step 1 (``hair_C_0..3``), MMD authors
    # from 1 with step 1 (``前髪01..08``), but HSR FBX rigs commonly use
    # step 2 (``BackHair1, BackHair3, BackHair5, ...``) where the gap
    # holds twist / auxiliary bones authored elsewhere in the rig. Both
    # should rig — the parent-chain check below is the real validator.
    if len(indices) >= 2:                                                # noqa: PLR2004
        step = indices[1] - indices[0]
        expected = list(range(indices[0], indices[0] + step * len(indices), step or 1))
        if step <= 0 or indices != expected:
            # DEBUG (not WARNING) — these are mostly false-positive
            # rejections for MMD 2D cloth grids (Lower-skirt, Sleeve,
            # Breast jiggle bones) and Chinese-named side hair whose
            # indices are non-sequential by rig design. Re-firing every
            # model load polluted INFO logs without indicating any
            # actual problem. Real broken chains still log via the
            # parent-link warning below.
            _log.debug("chain %r rejected: indices not uniformly spaced: %s", prefix, indices)
            return None
    # Single-bone "chains" are noise — a hair chain by definition has
    # multiple joints. This guard also stops the underscore-optional
    # pattern from sweeping in singletons like ``spine01`` or ``neck1``.
    if len(indices) < 2:                                              # noqa: PLR2004
        _log.debug("chain %r ignored: single bone, treating as standalone", prefix)
        return None
    joints = tuple(pair[1] for pair in indexed)
    anchor = joints[0].parent
    if anchor is None:
        _log.debug("chain %r rejected: joint 0 (%s) has no parent", prefix, joints[0].name)
        return None
    for idx in range(1, len(joints)):
        if joints[idx].parent is not joints[idx - 1]:
            _log.debug(
                "chain %r rejected: %s.parent is not %s",
                prefix,
                joints[idx].name,
                joints[idx - 1].name,
            )
            return None
    return DetectedChain(name=prefix, anchor=anchor, joints=joints)
