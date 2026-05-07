"""Engine-level foot-to-ground collision resolver.

Animation scripts can drive bones freely without worrying about feet
clipping into floor / stair geometry — the host calls
:meth:`FootPlanter.apply` each frame after script tick, and any foot
whose world Y has dropped below the registered ground surface gets
snapped back via 2-bone IK so the leg deforms naturally instead of
the foot just teleporting.

The "ground surface" is a callable ``Y = f(X, Z)`` (the
:data:`GroundProvider` protocol), so the same code serves flat floors,
staircases, slopes, or any other parametric elevation field. The
:func:`stair_ground` helper builds a provider for a regular flight of
N steps without any per-mesh collision detection.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from posecascade.animation.ik import (
    _add_world_delta,
    _quat_from_to,
    solve_two_bone_analytic,
)
from posecascade.scene.node import Node

GroundProvider = Callable[[float, float], float]

_DEFAULT_PENETRATION_TOLERANCE = 1e-4
_PARENT_Y_PROJECTION_DEGENERATE = 1e-4
# Below this dot-product gap from ±1, slerp falls back to linear
# interpolation — at very small angles the spherical formula's
# ``sin_omega`` denominator goes to zero and amplifies error.
_SLERP_LINEAR_THRESHOLD = 1e-6
# Two samples (heel + toe at minimum) are needed to fit a slope; with
# fewer the planter falls back to "sole flat in world".
_MIN_SAMPLES_FOR_SLOPE = 2


@dataclass(frozen=True)
class FootIKChain:
    """A 2-bone IK chain (hip → knee → foot) bound to a ground provider.

    The chain is described in node terms — the planter never reaches
    into bone-name strings, so the same primitive serves quadrupeds,
    arms used as feet for crawling animations, etc. ``ground`` is the
    elevation function the foot must stay on top of.

    ``sole_samples`` is a tuple of ``(local_axis, distance_world)``
    pairs describing where the foot's actual contact surface sits
    relative to the foot bone joint. Each ``local_axis`` is a vector
    in the foot bone's *pre-scale* local frame (the engine extracts
    a unit rotation from the bone's world matrix so the user only
    has to think about direction, not the rig's import scale), and
    ``distance_world`` is the world-space distance from the joint to
    that sample point along that direction. Pass two samples (heel
    and toe at sole-down direction with different forward offsets)
    to catch both ends of the foot rotating below ground; pass one
    to treat the foot as a single contact point. Defaults to a
    single sample at the joint origin (no offset), preserving the
    legacy "joint = sole" behaviour for back-compat.

    ``sole_down_local`` is the foot bone's local-frame axis that
    points toward the sole at rest pose. When set, the planter
    rotates the foot bone after lift so this axis aligns with world
    -Y — keeping the sole flat on whatever surface the IK lifted to,
    instead of the foot inheriting whatever tilt the upper/lower leg
    happened to land at. Pass ``None`` to leave foot rotation alone
    (the script controls dorsi/plantar flexion itself).

    ``knee_limit_*`` pass through to the IK solver to keep the knee
    on a hinge axis when the script-level IK runs (the planter's
    own corrective IK ignores them — see :meth:`FootPlanter.apply`).
    """

    root: Node
    mid: Node
    end: Node
    ground: GroundProvider
    sole_samples: tuple[tuple[tuple[float, float, float], float], ...] = (
        ((0.0, 0.0, 0.0), 0.0),
    )
    sole_down_local: tuple[float, float, float] | None = None
    toe_forward_local: tuple[float, float, float] | None = None
    knee_limit_min: tuple[float, float, float] | None = None
    knee_limit_max: tuple[float, float, float] | None = None


@dataclass
class FootPlanter:
    """Per-frame foot-floor collision resolver.

    Lives on :class:`~posecascade.app.registry.Services`. Scripts (via
    the sandbox ``floor`` API) or the host register chains with
    :meth:`bind`; each tick :meth:`apply` walks the chains and runs
    2-bone IK on any foot whose world Y has dropped below its ground
    surface, returning it to just above the surface.

    ``foot_offset`` is the rest-pose distance from the foot bone's
    head joint down to the sole — the IK target is set to
    ``ground + foot_offset`` so the foot's plant point lands AT the
    ground rather than the joint origin sinking into it.

    ``y_clamp_fallback`` is a *destructive* safety net: when IK
    undershoots, the planter directly translates the foot bone's
    local position to force its world Y to match the target. This
    detaches the foot from lower_leg's tail, so any mesh vertex
    blended across the ankle joint stretches visibly — useful only
    for diagnostics. The default is OFF; we'd rather show a couple
    of millimetres of clip-through than a deformed ankle. If IK
    really can't reach the target (chain too short for the lift),
    bump ``iterations`` or move the body up at the script level.
    """

    chains: list[FootIKChain] = field(default_factory=list)
    foot_offset: float = 0.0
    penetration_tolerance: float = _DEFAULT_PENETRATION_TOLERANCE
    # Body's current forward direction in world coordinates. Used for:
    # (1) the bend-hint axis of the analytical IK so the knee folds
    #     forward along the body's facing direction (NOT a fixed
    #     world axis — descending stairs flips body yaw so a fixed
    #     bend_hint folds the knee BACKWARD, hyperextending the
    #     leg = "下樓梯腳變形");
    # (2) the secondary aim target for ``toe_forward_local`` so toes
    #     point along body's facing.
    # Update this every frame from the script side as the body yaw
    # changes (e.g. ``floor.set_body_forward(world_dir)``).
    body_forward_world: tuple[float, float, float] = (0.0, 0.0, -1.0)
    # Cap on the per-frame "lift, re-sample, lift again" loop. Each
    # IK pass moves the foot in 3D, so a sample that was below stair N
    # may end up above stair N+1 (whose top is higher) — we re-sample
    # and lift again until either we're above ground or we've exhausted
    # the budget. Three passes converge in practice for staircase
    # geometry; bump this only if you have very steep slopes.
    max_resolution_passes: int = 3
    # Align the sole to world -Y whenever any sample is within this
    # vertical distance of the ground. Above this distance the foot
    # is "in the air" mid-swing and we leave its rotation alone — so
    # the leading leg can still dorsi/plantar-flex naturally during
    # the lift phase. Set to ``inf`` to always align, or 0 to never.
    sole_contact_threshold: float = 0.025
    # Cap on the foot's bend angle relative to the lower-leg's "up
    # the leg" axis. Real ankles bend 30-40°; the planter's sole
    # alignment, run on a leg the IK has tilted forward 50°+, would
    # otherwise produce ~80° ankle bend (sole flat on stair, foot
    # mesh stretched at the ankle joint — what the user sees as
    # "腳變形"). After alignment we measure ankle angle and pull the
    # foot back toward the lower_leg axis by the excess. Set to π
    # (180°) to disable the cap entirely.
    max_ankle_bend_rad: float = 0.6  # ≈ 34°
    # Maximum frame-to-frame change in the IK lift target's world Y,
    # in metres. Originally 0.02 m to avoid leg-pose snaps on stair
    # edges, but with foot dorsiflexion + sole alignment now off,
    # the leg's IK rotation already changes smoothly frame to
    # frame; capping lift just leaves the foot below ground for a
    # few frames during catch-up. Default to ``inf`` (instant lift,
    # no clip-through residue); set to a finite value if a future
    # gait re-enables abrupt foot-rotation changes.
    lift_velocity_cap: float = float("inf")
    # Maximum angular delta the engine will apply to the foot bone in
    # one frame. Without this, ``_align_sole_to_ground`` can produce
    # corrections of ~180° when the foot's toe direction crosses the
    # singularity (= "前後相反" — toes facing backward in one tick).
    # Capping per-frame rotation slerps the foot toward the desired
    # orientation across multiple frames; flips become smooth turns.
    rotation_velocity_cap_rad: float = 0.5  # ≈ 28°/frame
    _prev_target_y: dict[int, float] = field(default_factory=dict)
    _prev_foot_local_rot: dict[int, np.ndarray] = field(default_factory=dict)

    def bind(self, chain: FootIKChain) -> None:
        """Register a leg chain. Idempotent on equal chains."""
        if chain not in self.chains:
            self.chains.append(chain)

    def clear(self) -> None:
        """Drop all registered chains. Used when the scene changes."""
        self.chains.clear()

    def apply(self) -> int:
        """Resolve collisions for every registered chain.

        Per chain we run an outer loop of {lift via analytical IK →
        align sole / toe → cap ankle bend} until either no sample
        penetrates or the resolution budget is exhausted. The loop
        is needed because each step can disturb the others: IK
        moves the foot horizontally onto a different ground height,
        alignment rotates the foot which moves the sole samples,
        and the ankle cap rolls the foot back which un-flattens
        the sole. Iterating the whole pipeline lets them converge
        on a pose that's lifted, aligned, AND anatomically natural.

        Returns the number of feet adjusted — useful for debug HUDs.
        """
        bend_hint = np.asarray(self.body_forward_world, dtype=np.float32)
        adjusted = 0
        for chain in self.chains:
            lifted = False
            for _ in range(max(1, self.max_resolution_passes)):
                step_lifted = self._lift_pass(chain, bend_hint)
                if chain.sole_down_local is not None:
                    self._maybe_align_sole(chain)
                if not step_lifted:
                    break
                lifted = True
            # Apply rotation smoothing ONCE per chain per frame, after
            # the resolution loop has converged on a desired pose.
            self._smooth_foot_rotation(chain)
            if lifted:
                adjusted += 1
        return adjusted

    def _lift_pass(self, chain: FootIKChain, bend_hint: np.ndarray) -> bool:
        """One lift step: measure worst penetration, run analytical IK
        to clear it. Returns ``True`` iff we ran the IK.

        ``lift_velocity_cap`` smooths the IK target across frames —
        a sudden 2 cm jump in ground height (foot crossing a stair
        edge) otherwise triggers a 2 cm leg-pose snap in one tick.
        Capping frame-to-frame Y change at the configured limit
        spreads the same lift over multiple frames; the foot may
        briefly clip below ground during the catch-up but the
        upper / lower leg bones rotate smoothly.
        """
        end_world = _world_matrix(chain.end)
        max_penetration = self._max_sample_penetration(chain, end_world)
        if max_penetration <= self.penetration_tolerance:
            # Decay the cached target so smoothing doesn't keep
            # pinning the foot above ground after the ramp ends.
            self._prev_target_y.pop(id(chain), None)
            return False
        joint_y_now = float(end_world[1, 3])
        target_joint_y = joint_y_now + max_penetration
        cap = self.lift_velocity_cap
        if cap > 0.0:
            prev = self._prev_target_y.get(id(chain), joint_y_now)
            delta = target_joint_y - prev
            if abs(delta) > cap:
                delta = cap if delta > 0 else -cap
            target_joint_y = prev + delta
        self._prev_target_y[id(chain)] = target_joint_y
        target = np.array(
            [float(end_world[0, 3]), target_joint_y, float(end_world[2, 3])],
            dtype=np.float32,
        )
        solve_two_bone_analytic(
            chain.root, chain.mid, chain.end, target, bend_hint=bend_hint,
        )
        return True


    def _maybe_align_sole(self, chain: FootIKChain) -> None:
        """Run sole alignment if any sample is within the contact threshold.

        Inner-loop variant: no rotation smoothing here (we may be
        called multiple times per frame inside the resolution loop).
        Frame-to-frame smoothing happens once in :meth:`_smooth_foot_rotation`
        after the loop converges.
        """
        end_world = _world_matrix(chain.end)
        unit_rotation = _extract_unit_rotation(end_world)
        for axis_local, distance in chain.sole_samples:
            axis_world = unit_rotation @ np.asarray(axis_local, dtype=np.float64)
            sample_world = end_world[:3, 3] + axis_world * float(distance)
            ground_y = chain.ground(
                float(sample_world[0]), float(sample_world[2]),
            ) + self.foot_offset
            if abs(float(sample_world[1]) - ground_y) <= self.sole_contact_threshold:
                self._align_sole_to_ground(chain, end_world, unit_rotation)
                return

    def _smooth_foot_rotation(self, chain: FootIKChain) -> None:
        """Cap per-frame foot rotation against the previous frame's result.

        Runs once per chain at the end of the apply loop, so the
        cap applies to "what the foot looked like last frame" vs
        "what alignment wants this frame" — not to the intra-frame
        passes that the iterative resolution makes. Without this
        outer-vs-inner separation, three inner passes could each
        slerp by the cap and frame-to-frame change would be 3× cap.
        """
        if self.rotation_velocity_cap_rad >= np.pi:
            return
        if chain.sole_down_local is None:
            return
        desired_local = np.asarray(chain.end.transform.rotation, dtype=np.float64).copy()
        prev_local = self._prev_foot_local_rot.get(id(chain))
        if prev_local is None:
            self._prev_foot_local_rot[id(chain)] = desired_local
            return
        angular = _quat_angle_between(prev_local, desired_local)
        if angular > self.rotation_velocity_cap_rad:
            t = float(self.rotation_velocity_cap_rad / angular)
            applied = _quat_slerp(prev_local, desired_local, t)
        else:
            applied = desired_local
        chain.end.transform.set_rotation(applied.astype(np.float32, copy=False))
        self._prev_foot_local_rot[id(chain)] = applied

    def _align_sole_to_ground(
        self,
        chain: FootIKChain,
        end_world: np.ndarray,
        unit_rotation: np.ndarray,
    ) -> None:
        """Tilt the foot to match the local ground slope across its samples.

        Per-sample: compute world position now, compute world target
        (= ground at sample's XZ + foot offset). The foot's "tilt
        axis" is heel→toe. Rotating heel→toe NOW so it aligns with
        heel→toe TARGET makes the foot pivot to span both samples'
        grounds — flat on level ground, tilted up where the toe
        sample sits over a higher stair, tilted down where it sits
        over a lower one. Without this, forcing sole flat over a
        stair edge leaves the heel touching the lower stair and
        toe floating above the higher one (or vice versa) =
        "只用後腳跟走樓梯".

        Falls back to single-axis sole-down → world -Y when there's
        only one sample (no slope info available). The ankle cap
        runs in either case to keep the foot vs lower-leg angle
        within anatomical range.
        """
        sole_axis_local = np.asarray(chain.sole_down_local, dtype=np.float64)
        sole_norm = float(np.linalg.norm(sole_axis_local))
        if sole_norm < _DEFAULT_PENETRATION_TOLERANCE:
            return
        sole_axis_local /= sole_norm
        if len(chain.sole_samples) >= _MIN_SAMPLES_FOR_SLOPE:
            self._tilt_foot_to_match_ground(chain, end_world, unit_rotation)
        else:
            self._sole_down_to_world_minus_y(chain, end_world, unit_rotation, sole_axis_local)
        # Optional toe twist around the new sole axis to keep toes
        # pointing along ``body_forward_world``.
        if chain.toe_forward_local is not None:
            self._twist_toe_to_body_forward(chain, sole_axis_local)

    def _tilt_foot_to_match_ground(
        self,
        chain: FootIKChain,
        end_world: np.ndarray,
        unit_rotation: np.ndarray,
    ) -> None:
        """Rotate the foot so heel→toe direction matches the ground slope."""
        knee_world = _world_matrix(chain.mid)[:3, 3]
        joint_world = end_world[:3, 3]
        sample_now: list[np.ndarray] = []
        sample_target: list[np.ndarray] = []
        for axis_local, distance in chain.sole_samples:
            axis_world = unit_rotation @ np.asarray(axis_local, dtype=np.float64)
            now = joint_world + axis_world * float(distance)
            ground_y = chain.ground(float(now[0]), float(now[2])) + self.foot_offset
            target = np.array([float(now[0]), ground_y, float(now[2])], dtype=np.float64)
            sample_now.append(now.astype(np.float64, copy=False))
            sample_target.append(target)
        # Heel = first sample, toe = second. Heel→toe vector in world.
        forward_now = sample_now[1] - sample_now[0]
        forward_tgt = sample_target[1] - sample_target[0]
        fn_norm = float(np.linalg.norm(forward_now))
        ft_norm = float(np.linalg.norm(forward_tgt))
        if fn_norm < _DEFAULT_PENETRATION_TOLERANCE:
            return
        if ft_norm < _DEFAULT_PENETRATION_TOLERANCE:
            return
        forward_now_unit = forward_now / fn_norm
        forward_tgt_unit = forward_tgt / ft_norm
        # Cap the rotation so the foot doesn't twist relative to the
        # leg by more than max_ankle_bend_rad in one step.
        leg_vec = knee_world - joint_world
        leg_norm = float(np.linalg.norm(leg_vec))
        leg_up_world = (
            leg_vec / leg_norm if leg_norm >= _DEFAULT_PENETRATION_TOLERANCE
            else np.array([0.0, 1.0, 0.0])
        )
        capped_target = _slerp_axis_capped(
            forward_now_unit, forward_tgt_unit, self.max_ankle_bend_rad,
        )
        delta = _quat_from_to(forward_now_unit, capped_target)
        _add_world_delta(chain.end, delta)
        # Use the leg axis to also clamp how far the foot's "up"
        # direction has drifted from the leg's up axis (sole staying
        # roughly perpendicular to the local leg).
        end_after = _world_matrix(chain.end)
        unit_after = _extract_unit_rotation(end_after)
        sole_world = unit_after @ np.asarray(chain.sole_down_local, dtype=np.float64)
        # Project sole down direction onto plane perpendicular to
        # ``capped_target`` so a residual roll around the heel→toe
        # axis doesn't spin the sole away from the ground normal.
        forward_axis = capped_target
        sole_proj = sole_world - float(np.dot(sole_world, forward_axis)) * forward_axis
        sole_proj_norm = float(np.linalg.norm(sole_proj))
        if sole_proj_norm < _DEFAULT_PENETRATION_TOLERANCE:
            return
        sole_proj /= sole_proj_norm
        # Target sole direction = perpendicular to forward_axis,
        # pointing toward gravity (world -Y).
        target_down = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        target_proj = (
            target_down - float(np.dot(target_down, forward_axis)) * forward_axis
        )
        target_norm = float(np.linalg.norm(target_proj))
        if target_norm < _DEFAULT_PENETRATION_TOLERANCE:
            return
        target_proj /= target_norm
        # Cap the roll correction by max_ankle_bend so we don't fight
        # the leg here either.
        capped_sole = _slerp_axis_capped(
            sole_proj, target_proj, self.max_ankle_bend_rad,
        )
        roll_delta = _quat_from_to(sole_proj, capped_sole)
        _add_world_delta(chain.end, roll_delta)
        # Keep ``leg_up_world`` referenced for diagnostic readability.
        _ = leg_up_world

    def _sole_down_to_world_minus_y(
        self,
        chain: FootIKChain,
        end_world: np.ndarray,
        unit_rotation: np.ndarray,
        sole_axis_local: np.ndarray,
    ) -> None:
        """Single-sample fallback: align sole-down to world -Y, ankle-capped."""
        foot_up_local = -sole_axis_local
        foot_up_world = unit_rotation @ foot_up_local
        knee_world = _world_matrix(chain.mid)[:3, 3]
        joint_world = end_world[:3, 3]
        leg_vec = knee_world - joint_world
        leg_norm = float(np.linalg.norm(leg_vec))
        leg_up_world = (
            leg_vec / leg_norm if leg_norm >= _DEFAULT_PENETRATION_TOLERANCE
            else np.array([0.0, 1.0, 0.0])
        )
        target_up = _slerp_axis_capped(
            leg_up_world,
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            self.max_ankle_bend_rad,
        )
        _add_world_delta(chain.end, _quat_from_to(foot_up_world, target_up))

    def _twist_toe_to_body_forward(
        self, chain: FootIKChain, sole_axis_local: np.ndarray,
    ) -> None:
        """Pure twist around foot-up so toes face ``body_forward_world``."""
        end_world = _world_matrix(chain.end)
        unit_rotation = _extract_unit_rotation(end_world)
        toe_axis_local = np.asarray(chain.toe_forward_local, dtype=np.float64)
        toe_norm = float(np.linalg.norm(toe_axis_local))
        if toe_norm < _DEFAULT_PENETRATION_TOLERANCE:
            return
        toe_axis_local /= toe_norm
        foot_up_world = unit_rotation @ (-sole_axis_local)
        toe_world = unit_rotation @ toe_axis_local
        toe_proj = toe_world - float(np.dot(toe_world, foot_up_world)) * foot_up_world
        toe_proj_norm = float(np.linalg.norm(toe_proj))
        body_fwd = np.asarray(self.body_forward_world, dtype=np.float64)
        target_proj = (
            body_fwd - float(np.dot(body_fwd, foot_up_world)) * foot_up_world
        )
        target_proj_norm = float(np.linalg.norm(target_proj))
        if toe_proj_norm < _DEFAULT_PENETRATION_TOLERANCE:
            return
        if target_proj_norm < _DEFAULT_PENETRATION_TOLERANCE:
            return
        toe_proj /= toe_proj_norm
        target_proj /= target_proj_norm
        _add_world_delta(chain.end, _quat_from_to(toe_proj, target_proj))

    def _max_sample_penetration(
        self, chain: FootIKChain, end_world: np.ndarray,
    ) -> float:
        unit_rotation = _extract_unit_rotation(end_world)
        worst = 0.0
        for axis_local, distance in chain.sole_samples:
            axis_world = unit_rotation @ np.asarray(axis_local, dtype=np.float64)
            sample_world = end_world[:3, 3] + axis_world * float(distance)
            ground_y = chain.ground(
                float(sample_world[0]), float(sample_world[2]),
            ) + self.foot_offset
            pen = ground_y - float(sample_world[1])
            worst = max(worst, pen)
        return worst


def flat_ground(y: float = 0.0) -> GroundProvider:
    """Return a constant-elevation provider — useful for level floors."""
    return lambda _x, _z, _y=y: _y


def stair_ground(
    *,
    base_z: float,
    step_depth: float,
    step_rise: float,
    count: int,
    base_y: float = 0.0,
    forward_sign: int = -1,
    edge_smooth: float | None = None,
) -> GroundProvider:
    """Build a provider for a regular flight of ``count`` steps.

    ``base_z`` is the front edge of the bottom step in world Z;
    ``forward_sign`` is +1 if walking up moves Z toward +∞, or -1 if
    toward -∞ (the demo scene has stairs in -Z, so default is -1).
    Outside the staircase footprint the provider returns ``base_y``
    in front of the first step and ``base_y + count*step_rise``
    past the top step.

    ``edge_smooth`` is the width (in world Z) over which the ground
    linearly ramps from the previous step's height up to the current
    step's at the front edge of each step. Without this, a foot
    sliding forward across an edge sees the ground height jump by
    ``step_rise`` in one frame — the foot planter responds with an
    instantaneous IK lift that snaps the leg pose visibly. Default
    is ``step_depth * 0.25`` (a quarter-step ramp), which smooths
    the transition without flattening the visible step geometry.
    Pass ``0`` for the legacy hard-edged behaviour.
    """
    if step_depth <= 0.0:
        raise ValueError(f"step_depth must be positive, got {step_depth}")
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    if edge_smooth is None:
        edge_smooth = step_depth * 0.25
    edge_smooth = max(0.0, min(float(edge_smooth), step_depth * 0.5))
    top_y = base_y + count * step_rise

    def provider(_x: float, z: float) -> float:
        # ``progressed`` is the foot's distance past the front edge of
        # the staircase, measured in the walking direction. Negative
        # before the staircase, positive once on it.
        progressed = (z - base_z) * forward_sign
        if progressed <= 0.0:
            return base_y
        step_pos = progressed / step_depth  # continuous index, e.g. 1.7 = 70% into step 2
        step_index = int(step_pos)
        if step_index >= count:
            return top_y
        current_h = base_y + (step_index + 1) * step_rise
        if edge_smooth <= 0.0:
            return current_h
        prev_h = base_y + step_index * step_rise
        edge_frac = edge_smooth / step_depth
        step_frac = step_pos - step_index
        if step_frac < edge_frac:
            ramp = step_frac / edge_frac
            return prev_h + (current_h - prev_h) * ramp
        return current_h

    return provider


def _world_position(node: Node) -> np.ndarray:
    matrix = node.transform.to_matrix()
    parent = node.parent
    while parent is not None:
        matrix = parent.transform.to_matrix() @ matrix
        parent = parent.parent
    return np.array([matrix[0, 3], matrix[1, 3], matrix[2, 3]], dtype=np.float32)


def _quat_angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Shortest unsigned angle between two unit quaternions in radians."""
    cos_a = abs(float(np.dot(a, b)))
    cos_a = max(-1.0, min(1.0, cos_a))
    return 2.0 * float(np.arccos(cos_a))


def _quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between two unit quaternions.

    Pre-flip ``b`` if its dot product with ``a`` is negative so we
    take the shorter arc — without this, slerp through 180° follows
    the longer route and the foot tumbles when the alignment crosses
    the antipodal singularity.
    """
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 1.0 - _SLERP_LINEAR_THRESHOLD:
        return ((1.0 - t) * a + t * b).astype(np.float64)
    omega = float(np.arccos(min(1.0, max(-1.0, dot))))
    sin_omega = float(np.sin(omega))
    if sin_omega < _SLERP_LINEAR_THRESHOLD:
        return ((1.0 - t) * a + t * b).astype(np.float64)
    s_a = float(np.sin((1.0 - t) * omega)) / sin_omega
    s_b = float(np.sin(t * omega)) / sin_omega
    return (s_a * a + s_b * b).astype(np.float64)


def _slerp_axis_capped(
    from_axis: np.ndarray, to_axis: np.ndarray, max_angle: float,
) -> np.ndarray:
    """Slerp ``from_axis`` toward ``to_axis`` by at most ``max_angle`` rad.

    Returns ``from_axis`` if they're already aligned, ``to_axis`` if
    the full rotation is within the cap, or a linearly interpolated
    direction at exactly ``max_angle`` from ``from_axis`` toward
    ``to_axis`` otherwise.
    """
    from_axis = from_axis / max(float(np.linalg.norm(from_axis)), _DEFAULT_PENETRATION_TOLERANCE)
    to_axis = to_axis / max(float(np.linalg.norm(to_axis)), _DEFAULT_PENETRATION_TOLERANCE)
    cos_a = float(np.clip(np.dot(from_axis, to_axis), -1.0, 1.0))
    angle = float(np.arccos(cos_a))
    if angle <= max_angle:
        return to_axis
    sin_total = float(np.sin(angle))
    if sin_total < _DEFAULT_PENETRATION_TOLERANCE:
        return to_axis
    t = max_angle / angle
    a_w = float(np.sin((1.0 - t) * angle)) / sin_total
    b_w = float(np.sin(t * angle)) / sin_total
    result = a_w * from_axis + b_w * to_axis
    return result / max(float(np.linalg.norm(result)), _DEFAULT_PENETRATION_TOLERANCE)


def _world_matrix(node: Node) -> np.ndarray:
    matrix = node.transform.to_matrix()
    parent = node.parent
    while parent is not None:
        matrix = parent.transform.to_matrix() @ matrix
        parent = parent.parent
    return matrix


def _extract_unit_rotation(world_matrix: np.ndarray) -> np.ndarray:
    """Strip per-axis scale from a 4×4 transform's 3×3 block.

    Imported VRoid characters often carry a non-unit scale on the
    Sketchfab wrapper that propagates into every bone's world matrix
    — so applying a "(0, 0, -0.014)" local-frame offset would scale
    by ~0.146 instead of producing a 14 mm world displacement. We
    normalise each column independently so user-supplied sole-sample
    directions can be expressed as pure rotation axes (unit vectors)
    paired with explicit world distances.
    """
    block = np.asarray(world_matrix[:3, :3], dtype=np.float64)
    out = np.empty_like(block)
    for col in range(3):
        norm = float(np.linalg.norm(block[:, col]))
        if norm < _PARENT_Y_PROJECTION_DEGENERATE:
            out[:, col] = (1.0, 0.0, 0.0) if col == 0 else \
                          (0.0, 1.0, 0.0) if col == 1 else \
                          (0.0, 0.0, 1.0)
            continue
        out[:, col] = block[:, col] / norm
    return out


def auto_foot_samples(
    foot_node: Node,
    skins,
    meshes,
    *,
    weight_threshold: float = 0.3,
    safety_margin: float = 0.005,
) -> tuple[
    tuple[tuple[tuple[float, float, float], float], ...],
    float,
]:
    """Derive ``(sole_samples, foot_offset)`` from the foot bone's skin geometry.

    Finds every mesh vertex weighted ``> weight_threshold`` to ``foot_node``
    in any registered skin, computes their REST world positions via the
    skin's inverse-bind matrix, and returns:

    * Three sample axes (heel-bottom, sole-center, toe-bottom) covering
      the foot's actual ground-contact extents — so the planter lifts
      whichever real mesh point is deepest into the floor instead of a
      script-guessed proxy axis.
    * A ``foot_offset`` set to ``safety_margin`` (sample distances
      already encode the rest-pose mesh extent; offset is just numerical
      slack for sample-vs-mesh XZ jitter on stair edge ramps).

    Returns ``((), 0.0)`` if the foot bone isn't found in any skin or
    has no dominant vertices — caller should fall back to script-supplied
    samples in that case.
    """
    foot_idx = -1
    skin_match = None
    for skin in skins:
        for i, joint in enumerate(skin.joints):
            if joint is foot_node:
                foot_idx = i
                skin_match = skin
                break
        if skin_match is not None:
            break
    if skin_match is None:
        return ((), 0.0)
    inv_bind = np.asarray(skin_match.inverse_bind_matrices[foot_idx], dtype=np.float64)
    foot_world = _world_matrix(foot_node).astype(np.float64, copy=False)
    skin_matrix = foot_world @ inv_bind
    foot_origin = foot_world[:3, 3]
    verts_world: list[np.ndarray] = []
    for mesh in meshes:
        if mesh.joints_0 is None or mesh.weights_0 is None or mesh.positions is None:
            continue
        for v_idx in range(mesh.positions.shape[0]):
            for k in range(_SKIN_INFLUENCES_PER_VERT):
                if (
                    int(mesh.joints_0[v_idx, k]) == foot_idx
                    and float(mesh.weights_0[v_idx, k]) > weight_threshold
                ):
                    v_pos = mesh.positions[v_idx]
                    v_h = np.array(
                        [float(v_pos[0]), float(v_pos[1]), float(v_pos[2]), 1.0],
                        dtype=np.float64,
                    )
                    verts_world.append((skin_matrix @ v_h)[:3])
                    break
    if not verts_world:
        return ((), 0.0)
    arr = np.asarray(verts_world)
    rel = arr - foot_origin
    # Pick the bottom 5 % of vertices (lowest in world Y) — these are
    # the ground-contact zone. Within that band, three extremes:
    # backmost (heel), central (sole), foremost (toe).
    sole_band_threshold = arr[:, 1].min() + (arr[:, 1].max() - arr[:, 1].min()) * 0.05
    sole_mask = arr[:, 1] <= sole_band_threshold
    if not np.any(sole_mask):
        return ((), 0.0)
    sole_indices = np.where(sole_mask)[0]
    sole_arr = arr[sole_mask]
    heel_idx = int(sole_indices[sole_arr[:, 2].argmin()])
    toe_idx = int(sole_indices[sole_arr[:, 2].argmax()])
    centre_idx = int(sole_indices[arr[sole_indices, 1].argmin()])
    extreme_offsets: dict[int, np.ndarray] = {}
    for idx in (heel_idx, centre_idx, toe_idx):
        extreme_offsets[idx] = rel[idx]
    unit_rotation = _extract_unit_rotation(foot_world)
    rotation_inv = unit_rotation.T  # orthonormal inverse
    samples: list[tuple[tuple[float, float, float], float]] = []
    for offset in extreme_offsets.values():
        norm = float(np.linalg.norm(offset))
        if norm < _DEFAULT_PENETRATION_TOLERANCE:
            continue
        unit_world = offset / norm
        local_axis = rotation_inv @ unit_world
        samples.append(
            (
                (
                    float(local_axis[0]),
                    float(local_axis[1]),
                    float(local_axis[2]),
                ),
                norm,
            ),
        )
    if not samples:
        return ((), 0.0)
    return tuple(samples), float(safety_margin)


_SKIN_INFLUENCES_PER_VERT = 4


__all__ = [
    "FootIKChain",
    "FootPlanter",
    "GroundProvider",
    "auto_foot_samples",
    "flat_ground",
    "stair_ground",
]
