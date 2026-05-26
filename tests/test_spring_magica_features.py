"""Tests for the Magica-Cloth-style extended SpringChain features.

Covers per-chain self collision, tether (max stretch), inertia carry,
stiffness gradient, per-joint mass, air drag, and the pose snapshot /
restore round-trip used for pre-simulation pose baking.
"""
from __future__ import annotations

import numpy as np

from posecascade.animation.spring import (
    SpringChain,
    SpringParams,
    SpringSimulator,
)
from posecascade.scene.node import Node
from posecascade.utils.math3d import vec3


def _make_chain(joint_count: int = 3, segment_length: float = 0.2) -> tuple[Node, list[Node]]:
    anchor = Node("anchor")
    joints: list[Node] = []
    for i in range(joint_count):
        j = Node(f"j{i}")
        j.transform.set_translation(vec3(0.0, -segment_length, 0.0))
        joints.append(j)
    return anchor, joints


# -------- F1: Self collision -------------------------------------------------

def test_self_collision_pushes_overlapping_joints_apart() -> None:
    """The self-collision pass projects two overlapping joints apart.

    We invoke the helper directly so the test exercises the resolver
    independent of the integrator (which would otherwise re-set
    world_position from rest each substep).
    """
    from posecascade.animation.spring import _apply_self_collision  # noqa: PLC0415
    anchor_a, joints_a = _make_chain(joint_count=2, segment_length=0.2)
    anchor_b, joints_b = _make_chain(joint_count=2, segment_length=0.2)
    chain_a = SpringChain.from_node_chain("a", anchor_a, joints_a)
    chain_b = SpringChain.from_node_chain("b", anchor_b, joints_b)
    chain_a.self_collision_radius = 0.10
    chain_b.self_collision_radius = 0.10
    # Two tip joints overlapping at roughly the same world point with
    # a tiny offset so the delta has a defined direction.
    chain_a.joints[-1].world_position = vec3(+0.001, -0.2, 0.0)
    chain_a.joints[-1].initialized = True
    chain_b.joints[-1].world_position = vec3(-0.001, -0.2, 0.0)
    chain_b.joints[-1].initialized = True
    _apply_self_collision([chain_a, chain_b])
    tip_a = np.asarray(chain_a.joints[-1].world_position, dtype=np.float32)
    tip_b = np.asarray(chain_b.joints[-1].world_position, dtype=np.float32)
    sep = float(np.linalg.norm(tip_a - tip_b))
    assert sep >= 0.10 + 0.10 - 5.0e-3, f"tips not pushed apart: sep={sep}"


def test_self_collision_disabled_when_radius_zero() -> None:
    """No correction when self_collision_radius == 0 (back-compat default)."""
    anchor, joints = _make_chain(joint_count=2, segment_length=0.2)
    chain = SpringChain.from_node_chain(
        "a", anchor, joints, params=SpringParams(stiffness=2.0, damping=0.3),
    )
    assert chain.self_collision_radius == 0.0
    sim = SpringSimulator(chains=[chain])
    sim.step(1.0 / 60.0)
    # No assertion errors raised → pass


# -------- F2: Tether / max stretch ------------------------------------------

def test_tether_caps_parent_to_joint_distance() -> None:
    """An over-stretched joint is pulled back along the bone vector."""
    anchor, joints = _make_chain(joint_count=2, segment_length=0.2)
    chain = SpringChain.from_node_chain(
        "a", anchor, joints, params=SpringParams(stiffness=2.0, damping=0.3),
    )
    chain.tether_max_stretch = 1.10
    sim = SpringSimulator(chains=[chain])
    # Place tip far from anchor (2× rest), then step — tether should pull
    # it back to 1.10 × 0.2 = 0.22.
    chain.joints[0].world_position = vec3(0.0, -0.2, 0.0)
    chain.joints[0].initialized = True
    chain.joints[1].world_position = vec3(0.0, -0.6, 0.0)
    chain.joints[1].initialized = True
    sim.step(1.0 / 60.0)
    parent = np.asarray(chain.joints[0].world_position, dtype=np.float32)
    tip = np.asarray(chain.joints[1].world_position, dtype=np.float32)
    dist = float(np.linalg.norm(tip - parent))
    assert dist <= 0.20 * 1.10 + 1.0e-3, f"tether did not cap: dist={dist}"


def test_tether_off_by_default() -> None:
    anchor, joints = _make_chain()
    chain = SpringChain.from_node_chain("a", anchor, joints)
    assert chain.tether_max_stretch == 0.0


# -------- F4: Inertia carry --------------------------------------------------

def test_inertia_carry_imprints_anchor_velocity() -> None:
    """Rotating the anchor between two ticks gives joints non-zero angular vel."""
    anchor, joints = _make_chain(joint_count=2, segment_length=0.2)
    chain = SpringChain.from_node_chain(
        "a", anchor, joints,
        params=SpringParams(stiffness=0.5, damping=0.05, inertia=0.05),
    )
    chain.inertia_carry = 0.8
    sim = SpringSimulator(chains=[chain])
    # Tick once to initialise joints.
    sim.step(1.0 / 60.0)
    for j in chain.joints:
        j.angular_velocity = vec3(0.0, 0.0, 0.0)
    # Rotate the anchor and step again — the inertia-carry pass should
    # imprint the anchor's angular velocity onto every joint.
    from posecascade.utils.math3d import quat_from_axis_angle  # noqa: PLC0415
    anchor.transform.set_rotation(quat_from_axis_angle(vec3(0.0, 1.0, 0.0), 0.5))
    sim.step(1.0 / 60.0)
    speed = float(np.linalg.norm(chain.joints[0].angular_velocity))
    assert speed > 1.0e-3, f"inertia carry did not transfer: speed={speed}"


# -------- F5: Stiffness gradient --------------------------------------------

def test_stiffness_gradient_lerps_root_to_tip() -> None:
    """When stiffness_tip is set, tip stiffness comes from the tip param."""
    anchor, joints = _make_chain(joint_count=3, segment_length=0.2)
    chain = SpringChain.from_node_chain(
        "a", anchor, joints, params=SpringParams(stiffness=10.0, damping=0.3),
    )
    chain.stiffness_tip = 1.0  # very soft tip
    sim = SpringSimulator(chains=[chain])
    # Apply a strong off-rest world rotation to all joints, then step.
    # The TIP joint (low stiffness) should restore less per step than
    # the ROOT joint (high stiffness).
    from posecascade.utils.math3d import quat_from_axis_angle  # noqa: PLC0415
    for j in chain.joints:
        j.world_rotation = quat_from_axis_angle(vec3(0.0, 0.0, 1.0), 0.7)
        j.initialized = True
    sim.step(1.0 / 60.0)
    root_spin = float(np.linalg.norm(chain.joints[0].angular_velocity))
    tip_spin = float(np.linalg.norm(chain.joints[-1].angular_velocity))
    assert root_spin > tip_spin, (
        f"root should restore faster than tip: root={root_spin} tip={tip_spin}"
    )


# -------- F6: Per-joint mass ------------------------------------------------

def test_heavier_joint_accelerates_less() -> None:
    """A heavy joint reaches less angular speed under the same gravity."""
    from posecascade.animation.spring import Gravity  # noqa: PLC0415
    anchor_a, joints_a = _make_chain(joint_count=1, segment_length=0.3)
    anchor_b, joints_b = _make_chain(joint_count=1, segment_length=0.3)
    chain_light = SpringChain.from_node_chain(
        "light", anchor_a, joints_a,
        params=SpringParams(stiffness=0.0, damping=0.0, inertia=0.5),
    )
    chain_heavy = SpringChain.from_node_chain(
        "heavy", anchor_b, joints_b,
        params=SpringParams(stiffness=0.0, damping=0.0, inertia=0.5),
    )
    chain_light.joints[0].mass = 1.0
    chain_heavy.joints[0].mass = 5.0
    sim = SpringSimulator(
        chains=[chain_light, chain_heavy],
        global_forces=[Gravity(force=np.asarray([1.0, 0.0, 0.0], dtype=np.float32))],
    )
    sim.step(1.0 / 60.0)
    sl = float(np.linalg.norm(chain_light.joints[0].angular_velocity))
    sh = float(np.linalg.norm(chain_heavy.joints[0].angular_velocity))
    assert sl > sh, f"heavy joint should accelerate less: light={sl} heavy={sh}"


# -------- F7: Pose snapshot / restore ---------------------------------------

def test_snapshot_and_restore_round_trip() -> None:
    """Snapshot a settled chain, perturb, restore — state matches."""
    anchor, joints = _make_chain(joint_count=2, segment_length=0.2)
    chain = SpringChain.from_node_chain("a", anchor, joints)
    sim = SpringSimulator(chains=[chain])
    # Bring chain into some settled state.
    sim.step(1.0 / 60.0)
    snap = sim.snapshot_chain_state()
    assert "a" in snap and len(snap["a"]) == 2                              # noqa: PLR2004
    # Perturb joints, then restore.
    for j in chain.joints:
        j.world_position = vec3(99.0, 99.0, 99.0)
    sim.restore_chain_state(snap)
    for j, packed in zip(chain.joints, snap["a"], strict=True):
        px, py, pz, *_ = packed
        wp = np.asarray(j.world_position, dtype=np.float32)
        assert abs(float(wp[0]) - px) < 1.0e-5
        assert abs(float(wp[1]) - py) < 1.0e-5
        assert abs(float(wp[2]) - pz) < 1.0e-5


# -------- F8: Local space simulation ----------------------------------------

def test_local_space_cancels_anchor_translation() -> None:
    """Translating the anchor with local_space=True doesn't drag the chain."""
    anchor, joints = _make_chain(joint_count=2, segment_length=0.2)
    chain = SpringChain.from_node_chain(
        "a", anchor, joints,
        params=SpringParams(stiffness=0.0, damping=0.0, inertia=0.5),
    )
    chain.local_space = True
    sim = SpringSimulator(chains=[chain])
    sim.step(1.0 / 60.0)
    pos_before = np.asarray(chain.joints[-1].world_position, dtype=np.float32).copy()
    # Translate the anchor by a large delta. With local_space=True the
    # spring step should subtract this out so the joint pose relative
    # to the anchor stays the same.
    anchor.transform.set_translation(vec3(2.0, 0.0, 0.0))
    sim.step(1.0 / 60.0)
    pos_after = np.asarray(chain.joints[-1].world_position, dtype=np.float32)
    rel_before = pos_before
    rel_after = pos_after - np.asarray([2.0, 0.0, 0.0], dtype=np.float32)
    # Joint position relative to anchor should be roughly unchanged.
    assert float(np.linalg.norm(rel_after - rel_before)) < 0.05               # noqa: PLR2004


# -------- F9: Air drag ------------------------------------------------------

def test_air_drag_decelerates_high_speed_joint() -> None:
    """A spinning joint with air_drag set loses speed faster than without."""
    anchor_a, joints_a = _make_chain(joint_count=1, segment_length=0.3)
    anchor_b, joints_b = _make_chain(joint_count=1, segment_length=0.3)
    no_drag = SpringChain.from_node_chain(
        "a", anchor_a, joints_a,
        params=SpringParams(stiffness=0.0, damping=0.0, inertia=0.5),
    )
    with_drag = SpringChain.from_node_chain(
        "b", anchor_b, joints_b,
        params=SpringParams(stiffness=0.0, damping=0.0, inertia=0.5),
    )
    with_drag.air_drag = 50.0
    sim = SpringSimulator(chains=[no_drag, with_drag])
    for chain in (no_drag, with_drag):
        chain.joints[0].angular_velocity = vec3(0.0, 0.0, 5.0)
        chain.joints[0].initialized = True
    sim.step(1.0 / 60.0)
    sn = float(np.linalg.norm(no_drag.joints[0].angular_velocity))
    sd = float(np.linalg.norm(with_drag.joints[0].angular_velocity))
    assert sd < sn, f"drag should decelerate: no_drag={sn} with_drag={sd}"


# -------- Field defaults preserve backwards-compat --------------------------

def test_new_fields_have_safe_defaults() -> None:
    anchor, joints = _make_chain()
    chain = SpringChain.from_node_chain("a", anchor, joints)
    assert chain.self_collision_radius == 0.0
    assert chain.tether_max_stretch == 0.0
    assert chain.stiffness_tip is None
    assert chain.air_drag == 0.0
    assert chain.inertia_carry == 0.0
    assert chain.local_space is False
    for j in chain.joints:
        assert j.mass == 1.0
