"""Tests for :mod:`posecascade.animation.foot_planting` and the
analytical 2-bone IK that drives it.

Covers the :func:`stair_ground` provider for the staircase elevation
field, the constant-elevation :func:`flat_ground` shortcut, and the
:class:`FootPlanter` collision resolver — both the no-op fast path
(foot already above ground) and the corrective IK call (foot below
ground gets pulled back up).
"""
from __future__ import annotations

import numpy as np

from posecascade.animation.foot_planting import (
    FootIKChain,
    FootPlanter,
    auto_foot_samples,
    flat_ground,
    stair_ground,
)
from posecascade.assets.types import Mesh, Skin
from posecascade.scene.node import Node
from posecascade.scene.transform import Transform
from posecascade.utils.math3d import vec3


def _make_node(name: str, *, translation=None) -> Node:
    transform = Transform()
    if translation is not None:
        transform.set_translation(translation)
    return Node(name=name, transform=transform)


def _make_chain(*, foot_y: float, ground) -> tuple[FootIKChain, Node]:
    """Three-bone chain (root → mid → end) with the end at world ``foot_y``.

    Layout: root at world Y=0.4 over origin, mid 0.2 below it, end at
    ``foot_y``. Returns the chain + the end node so the test can
    assert on its world position after ``apply``.
    """
    root = _make_node("hip", translation=vec3(0.0, 0.4, 0.0))
    mid = _make_node("knee", translation=vec3(0.0, -0.2, 0.0))
    end = _make_node("foot", translation=vec3(0.0, foot_y - 0.2, 0.0))
    root.add_child(mid)
    mid.add_child(end)
    return FootIKChain(root=root, mid=mid, end=end, ground=ground), end


# --- ground providers ------------------------------------------------------


def test_flat_ground_returns_constant() -> None:
    ground = flat_ground(y=0.05)
    assert ground(0.0, 0.0) == 0.05
    assert ground(1.5, -3.0) == 0.05


def test_stair_ground_before_first_step_returns_base() -> None:
    """A foot ahead of the staircase footprint sits at base_y."""
    ground = stair_ground(
        base_z=-0.20, step_depth=0.04, step_rise=0.02,
        count=5, base_y=0.0, forward_sign=-1,
    )
    # forward_sign=-1 means stairs go in -Z; Z > base_z means BEFORE the stairs.
    assert ground(0.0, 0.0) == 0.0
    assert ground(0.0, -0.10) == 0.0  # still in front of the bottom stair


def test_stair_ground_on_each_step() -> None:
    ground = stair_ground(
        base_z=-0.20, step_depth=0.04, step_rise=0.02,
        count=5, base_y=0.0, forward_sign=-1,
    )
    # Step 1 at z in (-0.24, -0.20): height = 0.02.
    assert ground(0.0, -0.22) == 0.02
    # Step 3 at z in (-0.32, -0.28): height = 0.06.
    assert ground(0.0, -0.30) == 0.06
    # Step 5 (top) at z in (-0.40, -0.36): height = 0.10.
    assert ground(0.0, -0.38) == 0.10


def test_stair_ground_past_top_returns_max_height() -> None:
    ground = stair_ground(
        base_z=-0.20, step_depth=0.04, step_rise=0.02,
        count=5, base_y=0.0, forward_sign=-1,
    )
    assert ground(0.0, -0.50) == 0.10  # well past the top step


def test_stair_ground_smooths_edges_by_default() -> None:
    """Each step's front edge linearly ramps from the previous step's
    height up to the current — without this, a foot sliding over an
    edge sees the ground jump by ``step_rise`` in one frame and the
    foot planter's IK lift snaps the whole leg pose. The ramp width
    defaults to ``step_depth/4`` so it's visible to the foot but
    barely changes the apparent stair geometry."""
    ground = stair_ground(
        base_z=-0.20, step_depth=0.04, step_rise=0.02,
        count=5, base_y=0.0, forward_sign=-1,
    )
    # Just BEFORE the stair-1/stair-2 boundary (z=-0.24): on stair 1.
    assert ground(0.0, -0.239) == 0.02
    # Just past the boundary: ramp begins. Height between 0.02 and 0.04.
    mid_ramp = ground(0.0, -0.245)
    assert 0.02 < mid_ramp < 0.04, f"mid-ramp height {mid_ramp} not in (0.02, 0.04)"
    # End of ramp (1 cm past boundary): full stair-2 height.
    assert abs(ground(0.0, -0.250) - 0.04) < 1e-6


def test_stair_ground_edge_smooth_zero_disables_ramp() -> None:
    """Pass ``edge_smooth=0`` to restore the legacy hard-edged stair
    profile — useful for tests that need the deterministic step-up."""
    ground = stair_ground(
        base_z=-0.20, step_depth=0.04, step_rise=0.02,
        count=5, base_y=0.0, forward_sign=-1, edge_smooth=0.0,
    )
    # Hard edge: just past the boundary jumps straight to stair-2 height.
    assert ground(0.0, -0.241) == 0.04


def test_stair_ground_supports_positive_forward_sign() -> None:
    """A staircase climbing in +Z direction (mirror of the demo scene)."""
    ground = stair_ground(
        base_z=0.20, step_depth=0.04, step_rise=0.02,
        count=5, base_y=0.0, forward_sign=+1,
    )
    assert ground(0.0, 0.10) == 0.0   # in front
    assert ground(0.0, 0.22) == 0.02  # step 1
    assert ground(0.0, 0.50) == 0.10  # past top


# --- FootPlanter -----------------------------------------------------------


def test_foot_planter_no_op_when_foot_above_ground() -> None:
    """Feet already at or above ground level mean ``apply`` does nothing."""
    chain, end = _make_chain(foot_y=0.10, ground=flat_ground(y=0.0))
    rest_translation = np.array(end.transform.translation, dtype=np.float64).copy()
    planter = FootPlanter()
    planter.bind(chain)
    adjusted = planter.apply()
    assert adjusted == 0
    np.testing.assert_allclose(end.transform.translation, rest_translation, atol=1e-6)


def test_foot_planter_lifts_when_foot_below_ground() -> None:
    """A foot starting below the ground gets pulled back via IK."""
    chain, end = _make_chain(foot_y=-0.05, ground=flat_ground(y=0.0))
    planter = FootPlanter()
    planter.bind(chain)
    adjusted = planter.apply()
    assert adjusted == 1


def test_foot_planter_clear_drops_chains() -> None:
    """``clear`` removes all bindings — used when the scene reloads."""
    chain, _ = _make_chain(foot_y=0.10, ground=flat_ground(y=0.0))
    planter = FootPlanter()
    planter.bind(chain)
    planter.clear()
    assert planter.chains == []
    assert planter.apply() == 0


def test_foot_planter_bind_is_idempotent() -> None:
    """Binding the same chain twice doesn't double up."""
    chain, _ = _make_chain(foot_y=0.10, ground=flat_ground(y=0.0))
    planter = FootPlanter()
    planter.bind(chain)
    planter.bind(chain)
    assert len(planter.chains) == 1


def test_foot_planter_uses_foot_offset_for_target() -> None:
    """``foot_offset`` adds a skin-thickness margin between the joint
    origin and the resolved ground plane — feet at world Y=0 with a
    +1 cm offset get treated as still penetrating ground at Y=0."""
    chain, _ = _make_chain(foot_y=0.005, ground=flat_ground(y=0.0))
    planter = FootPlanter(foot_offset=0.01)
    planter.bind(chain)
    adjusted = planter.apply()
    # foot_y=0.005 < ground(0)+offset(0.01)=0.01 → adjusts.
    assert adjusted == 1


def test_foot_planter_analytic_ik_lands_exactly_on_ground() -> None:
    """Analytical 2-bone IK puts the foot exactly on the ground plane —
    no iteration count to tune, no residual undershoot. Verifies the
    new closed-form solver replaces the old CCD + Y-clamp pair."""
    chain, end = _make_chain(foot_y=-0.05, ground=flat_ground(y=0.0))
    planter = FootPlanter()
    planter.bind(chain)
    planter.apply()
    world = end.transform.to_matrix()
    parent = end.parent
    while parent is not None:
        world = parent.transform.to_matrix() @ world
        parent = parent.parent
    # Foot should land within numerical tolerance of the ground.
    assert abs(world[1, 3]) < 1e-3, f"foot at {world[1, 3]} not on ground"


def test_foot_planter_iterates_to_clear_stepped_ground() -> None:
    """When IK lifts the foot, the foot also moves horizontally as the
    leg rotates — over a staircase that means a sample previously
    just below stair N can land just below stair N+1 (whose top is
    higher). The planter's iteration loop catches this and lifts
    again until samples clear; without it the demo's worst frame
    showed 18 mm of residual clip on the upper stair edge.
    """
    # Synthetic two-stair surface: ground=0 for x<0.10, =0.05 for x>=0.10.
    def stepped_ground(x: float, _z: float) -> float:
        return 0.05 if x >= 0.10 else 0.0

    root = _make_node("hip", translation=vec3(0.0, 0.4, 0.0))
    mid = _make_node("knee", translation=vec3(0.0, -0.2, 0.0))
    end = _make_node("foot", translation=vec3(0.05, -0.2, 0.0))
    root.add_child(mid)
    mid.add_child(end)
    chain = FootIKChain(
        root=root, mid=mid, end=end, ground=stepped_ground,
        # Two samples on either side of the joint along world X — when
        # the IK lifts the foot, the +X sample crosses onto the higher
        # step and needs another lift pass.
        sole_samples=(((+1.0, 0.0, 0.0), 0.06), ((-1.0, 0.0, 0.0), 0.06)),
    )
    planter = FootPlanter(max_resolution_passes=4)
    planter.bind(chain)
    planter.apply()
    # After convergence, both samples should be above their respective
    # ground levels.
    end_world = _world_matrix(end)
    unit_rotation = _extract_unit_rotation(end_world)
    for axis_local, distance in chain.sole_samples:
        axis_world = unit_rotation @ np.asarray(axis_local, dtype=np.float64)
        pt = end_world[:3, 3] + axis_world * distance
        ground_y = stepped_ground(float(pt[0]), float(pt[2]))
        assert pt[1] >= ground_y - 1e-3, (
            f"sample {axis_local} at Y={pt[1]:.4f} below ground Y={ground_y:.4f}"
        )


def test_foot_planter_rotation_cap_holds_across_resolution_passes() -> None:
    """Within one apply() call the resolution loop runs alignment up
    to ``max_resolution_passes`` times, and EACH pass would be
    capped if smoothing ran inside the loop — frame-to-frame change
    could end up at N × cap. The cap must run AFTER the loop, against
    the previous *frame's* result, not previous *pass's*. This
    regression locks the bug in by checking the cap holds even with
    a multi-pass setup that previously leaked rotation."""
    chain, end = _make_chain(foot_y=-0.05, ground=flat_ground(y=0.0))
    chain = FootIKChain(
        root=chain.root, mid=chain.mid, end=chain.end,
        ground=chain.ground,
        sole_samples=(((0.0, 0.0, -0.005), 0.020), ((0.0, 0.0, +0.005), 0.020)),
        sole_down_local=(0.0, -1.0, 0.0),
    )
    planter = FootPlanter(
        rotation_velocity_cap_rad=0.3,  # ~17°/frame
        max_resolution_passes=4,
    )
    planter.bind(chain)
    # Establish baseline rotation across two frames so prev-cache populates.
    planter.apply()
    planter.apply()
    rot_before = np.asarray(end.transform.rotation, dtype=np.float64).copy()
    # Now perturb the foot wildly to force a big alignment correction
    # and run another tick.
    end.transform.set_rotation(
        np.array([0.0, 0.0, 0.5, 0.866], dtype=np.float32),
    )
    planter.apply()
    rot_after = np.asarray(end.transform.rotation, dtype=np.float64)
    # Compute angular delta from the baseline — should not exceed cap
    # plus a small numerical slack.
    cos_a = abs(float(np.dot(rot_before, rot_after)))
    cos_a = max(-1.0, min(1.0, cos_a))
    delta = 2.0 * float(np.arccos(cos_a))
    assert delta <= 0.3 + 1e-3, (
        f"rotation delta {np.degrees(delta):.1f}° exceeds cap 17.2° + slack"
    )


def test_foot_planter_lift_velocity_cap_smooths_target_per_frame() -> None:
    """The IK target Y is capped to ``lift_velocity_cap`` change per
    apply() call. A foot 5 cm below ground with a 2 cm cap reaches
    only 2 cm above its initial position on the first frame, then
    catches up over subsequent frames. Without this cap, ground
    geometry that jumps by a step_rise in one frame triggers a
    matching whole-leg IK pose change in one tick — visible as
    "瞬間變形" at stair edges.
    """
    chain, end = _make_chain(foot_y=-0.05, ground=flat_ground(y=0.0))
    planter = FootPlanter(lift_velocity_cap=0.02, max_resolution_passes=1)
    planter.bind(chain)
    initial_y = float(_world_matrix(end)[1, 3])
    planter.apply()
    after_one_frame = float(_world_matrix(end)[1, 3])
    rise = after_one_frame - initial_y
    # Should rise by ~2 cm (the cap), not all 5 cm to ground.
    assert abs(rise - 0.02) < 1e-3, (
        f"first-frame rise {rise:.4f}m did not match velocity cap 0.02m"
    )
    # Two more frames bring it within tolerance of ground.
    planter.apply()
    planter.apply()
    final_y = float(_world_matrix(end)[1, 3])
    assert abs(final_y) < 1e-3, f"foot at Y={final_y:.4f} did not converge after 3 frames"


def test_foot_planter_single_pass_when_max_passes_is_one() -> None:
    """``max_resolution_passes=1`` reproduces the legacy "one lift,
    accept residual" behaviour, useful when the user wants speed
    over guaranteed clearance on stepped ground."""
    chain, end = _make_chain(foot_y=-0.05, ground=flat_ground(y=0.0))
    # Disable lift velocity cap so a single pass can fully reach the
    # target — this test pins the IK convergence property, not the
    # frame-to-frame smoothing behaviour.
    planter = FootPlanter(max_resolution_passes=1, lift_velocity_cap=float("inf"))
    planter.bind(chain)
    planter.apply()
    # Single pass on flat ground still hits target exactly (analytical IK).
    world = _world_matrix(end)
    assert abs(world[1, 3]) < 1e-3


def _world_matrix(node):
    """Local copy used by stair-step test (the module's helper is private)."""
    m = node.transform.to_matrix()
    p = node.parent
    while p is not None:
        m = p.transform.to_matrix() @ m
        p = p.parent
    return m


def _extract_unit_rotation(matrix):
    block = np.asarray(matrix[:3, :3], dtype=np.float64)
    out = np.empty_like(block)
    for col in range(3):
        norm = float(np.linalg.norm(block[:, col]))
        if norm < 1e-4:
            out[:, col] = (1, 0, 0) if col == 0 else (0, 1, 0) if col == 1 else (0, 0, 1)
        else:
            out[:, col] = block[:, col] / norm
    return out


def test_foot_planter_aligns_sole_to_ground_after_lift() -> None:
    """When ``sole_down_local`` is set, the planter rotates the foot
    bone after lift so its sole-down axis points world -Y. Without
    this, after the upper / lower leg rotate to lift the foot, the
    foot inherits the chain's tumble — sole points sideways or even
    upward — which reads as "foot is deformed while walking up".
    """
    chain, end = _make_chain(foot_y=-0.05, ground=flat_ground(y=0.0))
    # Galaxia-ish: sole_down_local along bone-local -Y, since the test
    # chain's "foot" hangs straight down from its parent at rest.
    chain = FootIKChain(
        root=chain.root, mid=chain.mid, end=chain.end,
        ground=chain.ground,
        sole_samples=chain.sole_samples,
        sole_down_local=(0.0, -1.0, 0.0),
    )
    planter = FootPlanter()
    planter.bind(chain)
    planter.apply()
    end_world = _world_matrix(end)
    unit_rotation = _extract_unit_rotation(end_world)
    sole_world = unit_rotation @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(sole_world, [0.0, -1.0, 0.0], atol=1e-3)


def test_foot_planter_tilts_to_match_local_ground_slope() -> None:
    """When heel and toe samples sit over different ground heights
    (foot spanning a stair edge), the planter rotates the foot so
    BOTH samples land on their respective ground surfaces — instead
    of forcing sole flat in world and leaving one sample floating
    over the higher / lower stair (= "只用後腳跟走樓梯").
    """
    # Heel sits over flat 0, toe sits over a 1 cm step in +Z (toe
    # ground higher than heel ground).
    def stepped_ground(_x: float, z: float) -> float:
        return 0.01 if z >= 0.02 else 0.0

    root = _make_node("hip", translation=vec3(0.0, 0.4, 0.0))
    mid = _make_node("knee", translation=vec3(0.0, -0.2, 0.0))
    end = _make_node("foot", translation=vec3(0.0, -0.22, 0.0))  # foot 2 cm below ground
    root.add_child(mid)
    mid.add_child(end)
    chain = FootIKChain(
        root=root, mid=mid, end=end,
        ground=stepped_ground,
        sole_samples=(
            ((0.0, 0.0, -1.0), 0.005),  # heel at -Z, 5 mm down  — straddle the step
            ((0.0, 0.0, +1.0), 0.005),  # toe  at +Z, 5 mm down
        ),
        sole_down_local=(0.0, -1.0, 0.0),
    )
    planter = FootPlanter(max_ankle_bend_rad=1.5)
    planter.bind(chain)
    # Two ticks: first lifts the foot, second settles after edge tilt.
    planter.apply()
    planter.apply()
    end_world = _world_matrix(end)
    unit = _extract_unit_rotation(end_world)
    heel_world = end_world[:3, 3] + unit @ np.array([0.0, 0.0, -1.0]) * 0.005
    toe_world = end_world[:3, 3] + unit @ np.array([0.0, 0.0, +1.0]) * 0.005
    heel_g = stepped_ground(float(heel_world[0]), float(heel_world[2]))
    toe_g = stepped_ground(float(toe_world[0]), float(toe_world[2]))
    # After tilt, both samples should sit close to their respective
    # ground heights. Tolerance accounts for ankle cap residual + the
    # synthetic chain's coarse fit (only 2 lift passes).
    assert abs(heel_world[1] - heel_g) < 0.008, (
        f"heel at Y={heel_world[1]:.4f} far from heel ground={heel_g:.4f}"
    )
    assert abs(toe_world[1] - toe_g) < 0.008, (
        f"toe at Y={toe_world[1]:.4f} far from toe ground={toe_g:.4f}"
    )


def test_foot_planter_caps_ankle_bend_to_anatomical_range() -> None:
    """When the leg is at an extreme tilt, fully flattening the foot
    to world -Y would bend the ankle 70°+ (real ankles bend ≤ 40°),
    stretching skinned mesh at the joint. The capped alignment slerps
    the sole-up target toward the lower-leg axis until the bend is
    within ``max_ankle_bend_rad`` — sole as flat as the rig allows
    without a deformed ankle. This pins that property.
    """
    # Synthetic chain pre-rotated so the lower_leg points at ~50° from
    # vertical (much more tilted than max_ankle_bend_rad would allow
    # if we forced the foot flat). The IK lift then forces alignment.
    root = _make_node("hip", translation=vec3(0.0, 0.4, 0.0))
    mid = _make_node("knee", translation=vec3(0.10, -0.15, 0.0))   # knee forward
    end = _make_node("foot", translation=vec3(0.05, -0.15, 0.0))   # foot below knee+forward
    root.add_child(mid)
    mid.add_child(end)
    chain = FootIKChain(
        root=root, mid=mid, end=end,
        ground=flat_ground(y=0.0),
        sole_samples=(((0.0, 0.0, 0.0), 0.0),),  # joint = sole, no offset
        sole_down_local=(0.0, -1.0, 0.0),
    )
    planter = FootPlanter(max_ankle_bend_rad=0.6)  # ≈ 34°
    planter.bind(chain)
    planter.apply()
    # Measure resulting ankle bend.
    knee_world = _world_matrix(mid)[:3, 3]
    foot_world = _world_matrix(end)[:3, 3]
    leg_up = (knee_world - foot_world) / max(
        float(np.linalg.norm(knee_world - foot_world)), 1e-6,
    )
    foot_up_world = _extract_unit_rotation(_world_matrix(end)) @ np.array(
        [0.0, 1.0, 0.0], dtype=np.float64,
    )
    cos_a = float(np.clip(leg_up @ foot_up_world, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos_a)))
    # Allow a few degrees of overshoot from the toe twist re-aligning
    # in the orthogonal plane.
    assert angle <= np.degrees(0.6) + 5.0, (
        f"ankle bent {angle:.1f}° past cap (≤ 34° + 5° tolerance)"
    )


def test_foot_planter_uses_body_forward_for_knee_bend() -> None:
    """The IK bend hint must follow ``body_forward_world`` so the knee
    bends in the body's facing direction. With a fixed ``-Z`` hint the
    descend phase (yaw=0, body facing +Z) folded the knee BACKWARD,
    hyperextending the leg = "下樓梯腳變形". This pins the fix:
    setting body_forward_world to +Z makes the knee bend toward +Z
    when the IK lift kicks in.
    """
    chain, end = _make_chain(foot_y=-0.10, ground=flat_ground(y=0.0))
    planter = FootPlanter(body_forward_world=(0.0, 0.0, 1.0))
    planter.bind(chain)
    planter.apply()
    knee_pos = _world_matrix(chain.mid)[:3, 3]
    hip_pos = _world_matrix(chain.root)[:3, 3]
    foot_pos = _world_matrix(end)[:3, 3]
    leg_dir = foot_pos - hip_pos
    leg_dir = leg_dir / max(float(np.linalg.norm(leg_dir)), 1e-6)
    knee_off = knee_pos - hip_pos
    knee_perp = knee_off - leg_dir * float(np.dot(knee_off, leg_dir))
    perp_norm = float(np.linalg.norm(knee_perp))
    assert perp_norm > 1e-4, "knee did not bend (chain may be straight)"
    # With body_forward = +Z, knee should be on +Z side of the hip-foot line.
    assert knee_perp[2] > 0, (
        f"knee perpendicular Z={knee_perp[2]:.4f} not on body-forward (+Z) side"
    )


def test_auto_foot_samples_derives_extremes_from_skin() -> None:
    """``auto_foot_samples`` walks the foot bone's skinned vertices,
    finds the lowest 5 % (= sole), and emits sample axes at the
    heel / centre / toe extremes. This pins the engine-internal
    auto-derivation so scripts can call ``floor.bind_foot`` with
    just (chain, ground) and skip the hand-tuned axis bookkeeping
    each rig requires."""
    foot = _make_node("foot", translation=vec3(0.0, 0.0, 0.0))
    # Synthetic skin with foot bone at index 0 + a single mesh
    # whose dominant vertices are arranged like a foot's sole:
    # heel at z=-0.005, sole-centre at z=0, toe at z=+0.025, all
    # at y=-0.014 (14 mm below the bone's joint origin).
    skin = Skin(
        name="rig",
        joints=(foot,),
        inverse_bind_matrices=np.tile(np.eye(4, dtype=np.float32), (1, 1, 1)),
    )
    positions = np.array(
        [
            [0.000, -0.014, -0.005],  # heel
            [0.000, -0.014, +0.000],  # sole centre
            [0.000, -0.014, +0.025],  # toe
            [0.000, +0.005, +0.000],  # top of foot — should NOT become a sample
        ],
        dtype=np.float32,
    )
    joints_0 = np.zeros((4, 4), dtype=np.uint32)
    weights_0 = np.zeros((4, 4), dtype=np.float32)
    weights_0[:, 0] = 1.0  # all dominantly bound to foot bone
    mesh = Mesh(
        name="foot_mesh",
        positions=positions,
        indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32),
        joints_0=joints_0,
        weights_0=weights_0,
    )
    samples, offset = auto_foot_samples(foot, (skin,), (mesh,), safety_margin=0.005)
    assert len(samples) >= 1, f"no samples derived (got {samples})"
    # Each sample axis should point downward (component along world -Y > 0).
    for axis, _distance in samples:
        assert axis[1] < 0, f"sample axis {axis} doesn't point down"
    # Sample distances should match the synthetic foot's geometry
    # (sqrt(0.014² + Δz²) = 14 mm to 28 mm).
    for _axis, distance in samples:
        assert 0.013 < distance < 0.030, f"sample distance {distance} out of expected band"
    assert offset == 0.005, f"expected safety margin 5 mm, got {offset}"


def test_auto_foot_samples_returns_empty_when_foot_not_in_skin() -> None:
    """Foot bone not present in any skin → empty samples + zero offset
    so the caller can fall back to a hand-supplied default."""
    foot = _make_node("foot", translation=vec3(0.0, 0.0, 0.0))
    other = _make_node("other", translation=vec3(0.0, 0.0, 0.0))
    skin = Skin(
        name="rig",
        joints=(other,),
        inverse_bind_matrices=np.tile(np.eye(4, dtype=np.float32), (1, 1, 1)),
    )
    samples, offset = auto_foot_samples(foot, (skin,), ())
    assert samples == ()
    assert offset == 0.0


def test_foot_planter_aligns_toe_to_body_forward() -> None:
    """When ``toe_forward_local`` is set on a chain, the planter's
    secondary alignment step rotates the foot around the now-vertical
    sole axis until its toe-forward axis points along
    ``body_forward_world`` — fixing front/back foot wobble during
    walking. Without this the foot is sole-flat but its toes drift
    sideways or backward as the leg rotates through stride.
    """
    chain, end = _make_chain(foot_y=-0.05, ground=flat_ground(y=0.0))
    chain = FootIKChain(
        root=chain.root, mid=chain.mid, end=chain.end,
        ground=chain.ground,
        sole_samples=chain.sole_samples,
        sole_down_local=(0.0, -1.0, 0.0),
        toe_forward_local=(0.0, 0.0, 1.0),  # local +Z = "toe forward" in this synthetic rig
    )
    planter = FootPlanter(body_forward_world=(0.0, 0.0, -1.0))
    planter.bind(chain)
    planter.apply()
    end_world = _world_matrix(end)
    unit_rotation = _extract_unit_rotation(end_world)
    toe_world = unit_rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # Toe should be aligned with body forward (-Z world). Y component
    # should also be ~0 (toe is horizontal because sole is flat).
    np.testing.assert_allclose(toe_world[0], 0.0, atol=1e-3)
    assert toe_world[2] < -0.95, f"toe Z={toe_world[2]} not pointing world -Z"


def test_foot_planter_skips_sole_align_when_foot_in_air() -> None:
    """Above ``sole_contact_threshold`` the foot is mid-swing (not on
    ground) — leaving its rotation alone preserves natural dorsi /
    plantar flexion during the lift phase. Without this gate the
    alignment fights the gait and the leg ends up flat-footed in
    the air, which looks robotic."""
    # Foot already comfortably above ground (foot_y=0.10, well past
    # the 25 mm contact threshold).
    chain, end = _make_chain(foot_y=0.10, ground=flat_ground(y=0.0))
    # Manually tilt the foot to a non-flat orientation.
    end.transform.set_rotation(
        np.array([0.0, 0.0, 0.3, 0.954], dtype=np.float32),  # ~35° around Z
    )
    rest_rot = end.transform.rotation.copy()
    chain = FootIKChain(
        root=chain.root, mid=chain.mid, end=chain.end,
        ground=chain.ground,
        sole_samples=chain.sole_samples,
        sole_down_local=(0.0, -1.0, 0.0),
    )
    planter = FootPlanter(sole_contact_threshold=0.025)
    planter.bind(chain)
    planter.apply()
    # Foot rotation should NOT have changed — alignment skipped.
    np.testing.assert_allclose(end.transform.rotation, rest_rot, atol=1e-5)


def test_foot_planter_unreachable_target_uses_closest_point() -> None:
    """When the target sits inside the chain's fold hole (d < |L1-L2|),
    analytical IK clamps the foot to the nearest reachable distance
    along the hip→target line — guaranteeing no detachment / drift
    when the gait pushes the leg into geometrically impossible poses
    (typically "trailing foot lock target slid past the hip side").
    """
    # _make_chain layout: root at Y=0.40, mid translation -0.20 (L1=0.20),
    # end translation = foot_y - 0.20. For foot_y=-0.50, end translation
    # = -0.70 → L2=0.70. Fold hole = d < |L1-L2| = 0.50; chain max
    # reach = L1+L2 = 0.90. Lift target lands at ground Y=0.0, distance
    # from root = 0.40, well inside the fold hole — IK must place the
    # foot at distance |L1-L2| ≈ 0.50 from root (closest the chain can
    # fold to), not somewhere arbitrary.
    chain, end = _make_chain(foot_y=-0.50, ground=flat_ground(y=0.0))
    len_upper, len_lower = 0.20, 0.70
    planter = FootPlanter()
    planter.bind(chain)
    planter.apply()
    hip_world = np.array([0.0, 0.40, 0.0], dtype=np.float64)
    world = end.transform.to_matrix()
    parent = end.parent
    while parent is not None:
        world = parent.transform.to_matrix() @ world
        parent = parent.parent
    end_pos = np.array([world[0, 3], world[1, 3], world[2, 3]], dtype=np.float64)
    distance = float(np.linalg.norm(end_pos - hip_world))
    fold_min = abs(len_upper - len_lower)
    fold_max = len_upper + len_lower
    assert fold_min - 1e-3 <= distance <= fold_max + 1e-3, (
        f"foot at distance {distance:.4f} from root outside reachable "
        f"annulus [{fold_min}, {fold_max}]"
    )


# --- helpers ---------------------------------------------------------------


__all__ = ["FootPlanter"]
