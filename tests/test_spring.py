"""Tests for :mod:`posecascade.animation.spring` — angular spring chain physics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from posecascade.animation.spring import (
    DEFAULT_CHAIN_PROFILES,
    Gravity,
    PointForce,
    SpringChain,
    SpringParams,
    SpringSimulator,
    Wind,
    detect_chains,
    node_world_pose,
    resolve_chain_params,
)
from posecascade.scene.component import SpringChainComponent
from posecascade.scene.node import Node
from posecascade.scene.transform import Transform
from posecascade.utils.math3d import (
    quat_from_axis_angle,
    quat_identity,
    vec3,
)


def _make_chain(joint_count: int = 3, segment_length: float = 0.1) -> tuple[Node, list[Node]]:
    """Build an anchor + ``joint_count`` joints hanging along -Y in local space.

    Returns ``(anchor, joint_nodes)``. Joints are parented in a linear chain so the
    deepest joint walks back through ``joint_count`` parent steps to reach ``anchor``.
    """
    anchor = Node(name="anchor", transform=Transform())
    parent = anchor
    joints = []
    for i in range(joint_count):
        joint = Node(
            name=f"joint_{i}",
            transform=Transform(
                translation=vec3(0.0, -segment_length, 0.0),
                rotation=quat_identity(),
                scale=vec3(1.0, 1.0, 1.0),
            ),
        )
        parent.add_child(joint)
        joints.append(joint)
        parent = joint
    return anchor, joints


def _run(sim: SpringSimulator, total_time: float, dt: float = 1.0 / 60.0) -> None:
    steps = int(total_time / dt)
    for _ in range(steps):
        sim.step(dt)


def test_rest_chain_stays_at_rest_no_forces() -> None:
    anchor, joints = _make_chain()
    chain = SpringChain.from_node_chain("test", anchor, joints)
    sim = SpringSimulator(chains=[chain])

    _run(sim, total_time=2.0)

    for joint in chain.joints:
        np.testing.assert_allclose(joint.angular_velocity, vec3(0.0, 0.0, 0.0), atol=1.0e-6)
        np.testing.assert_allclose(
            joint.node.transform.rotation, joint.rest_local_rotation, atol=1.0e-5
        )


def test_gravity_pulls_chain_below_rest_then_settles() -> None:
    # Rest chain hangs straight down (-Y). Gravity is also -Y, so it pulls
    # the chain ALONG its axis → no lever arm → no torque → joint stays at rest.
    # To see deflection we orient the rest chain along +X (horizontal) and apply
    # gravity in -Y; the chain should swing down and damp out.
    anchor = Node(name="anchor", transform=Transform())
    parent = anchor
    joints = []
    segment = 0.1
    for i in range(3):
        joint = Node(
            name=f"joint_{i}",
            transform=Transform(translation=vec3(segment, 0.0, 0.0)),
        )
        parent.add_child(joint)
        joints.append(joint)
        parent = joint

    chain = SpringChain.from_node_chain(
        "test",
        anchor,
        joints,
        params=SpringParams(stiffness=2.0, damping=2.0, inertia=0.05),
    )
    sim = SpringSimulator(chains=[chain])
    sim.add_force(Gravity(force=vec3(0.0, -1.0, 0.0)))

    _run(sim, total_time=4.0)

    # Final angular velocities should have damped to near zero.
    for joint in chain.joints:
        speed = float(np.linalg.norm(joint.angular_velocity))
        assert speed < 0.05, f"{joint.node.name} ω={speed} not damped"

    # Tip should be deflected DOWN (Y < 0) compared to rest pose at +X axis.
    tip = chain.joints[-1]
    assert tip.world_position[1] < -0.005, f"tip Y not deflected down: {tip.world_position}"


def test_chain_returns_to_rest_after_disturbance() -> None:
    anchor, joints = _make_chain()
    chain = SpringChain.from_node_chain(
        "test",
        anchor,
        joints,
        params=SpringParams(stiffness=20.0, damping=4.0, inertia=0.05),
    )
    sim = SpringSimulator(chains=[chain])

    # Initialise tracked state with one zero-step.
    sim.step(1.0 / 60.0)
    # Manually displace the first joint's world rotation.
    chain.joints[0].world_rotation = quat_from_axis_angle(vec3(0.0, 0.0, 1.0), math.pi / 4)

    _run(sim, total_time=5.0)

    for joint in chain.joints:
        speed = float(np.linalg.norm(joint.angular_velocity))
        assert speed < 0.01, f"{joint.node.name} ω={speed} did not settle"
        np.testing.assert_allclose(
            joint.node.transform.rotation, joint.rest_local_rotation, atol=5.0e-3
        )


def test_higher_damping_settles_faster() -> None:
    """A more-damped chain reaches equilibrium with less total motion than a lightly damped one."""

    def total_swing(damping: float) -> float:
        anchor = Node(name="anchor", transform=Transform())
        parent = anchor
        joints = []
        for i in range(2):
            joint = Node(
                name=f"j{i}",
                transform=Transform(translation=vec3(0.1, 0.0, 0.0)),
            )
            parent.add_child(joint)
            joints.append(joint)
            parent = joint
        chain = SpringChain.from_node_chain(
            "test",
            anchor,
            joints,
            params=SpringParams(stiffness=10.0, damping=damping, inertia=0.05),
        )
        sim = SpringSimulator(chains=[chain])
        sim.add_force(Gravity(force=vec3(0.0, -3.0, 0.0)))
        # Track the path-length of the tip's world position over time.
        last_pos = None
        path = 0.0
        for _ in range(120):
            sim.step(1.0 / 60.0)
            tip_pos = chain.joints[-1].world_position
            if last_pos is not None:
                path += float(np.linalg.norm(tip_pos - last_pos))
            last_pos = tip_pos.copy()
        return path

    light = total_swing(damping=0.5)
    heavy = total_swing(damping=4.0)
    assert heavy < light, f"expected heavy damping to reduce swing distance: {heavy=} {light=}"


def test_bone_length_preserved_under_swing() -> None:
    """Distance from each joint's pivot to its world position must equal its rest local length."""
    anchor = Node(name="anchor", transform=Transform())
    parent = anchor
    joints = []
    segment = 0.15
    for i in range(3):
        joint = Node(name=f"j{i}", transform=Transform(translation=vec3(segment, 0.0, 0.0)))
        parent.add_child(joint)
        joints.append(joint)
        parent = joint

    chain = SpringChain.from_node_chain("test", anchor, joints)
    sim = SpringSimulator(chains=[chain])
    sim.add_force(Gravity(force=vec3(0.0, -2.0, 0.0)))

    _run(sim, total_time=2.0)

    # Walk world positions ourselves and verify each consecutive pair is one segment apart.
    anchor_pos, _anchor_rot = node_world_pose(anchor)
    prev = anchor_pos
    for joint in chain.joints:
        d = float(np.linalg.norm(joint.world_position - prev))
        assert math.isclose(d, segment, abs_tol=1.0e-5), (
            f"bone length drifted: {joint.node.name} got {d}, expected {segment}"
        )
        prev = joint.world_position


def test_anchor_rotation_propagates_to_chain() -> None:
    """Rotating the anchor should drag the chain — tip world rotation tracks anchor."""
    anchor, joints = _make_chain()
    chain = SpringChain.from_node_chain(
        "test",
        anchor,
        joints,
        params=SpringParams(stiffness=80.0, damping=8.0, inertia=0.02),
    )
    sim = SpringSimulator(chains=[chain])

    _run(sim, total_time=0.5)
    initial_tip = chain.joints[-1].world_position.copy()

    # Rotate the anchor 60° around Z axis. Strong stiffness + low inertia ⇒ chain should follow.
    anchor.transform.set_rotation(quat_from_axis_angle(vec3(0.0, 0.0, 1.0), math.radians(60)))

    _run(sim, total_time=2.0)

    final_tip = chain.joints[-1].world_position
    moved = float(np.linalg.norm(final_tip - initial_tip))
    assert moved > 0.05, f"tip did not follow anchor rotation: moved {moved}"


def test_wind_pushes_chain_sideways() -> None:
    """Wind in +X should deflect a vertically-hanging chain in +X."""
    anchor, joints = _make_chain(joint_count=3, segment_length=0.1)
    chain = SpringChain.from_node_chain(
        "test",
        anchor,
        joints,
        params=SpringParams(stiffness=2.0, damping=3.0, inertia=0.05),
    )
    sim = SpringSimulator(chains=[chain])
    sim.add_force(Wind(direction=vec3(1.0, 0.0, 0.0), speed=2.0))

    _run(sim, total_time=4.0)

    tip = chain.joints[-1].world_position
    assert tip[0] > 0.005, f"tip not pushed in +X by wind: {tip}"


def test_reset_returns_chain_to_rest_state() -> None:
    anchor, joints = _make_chain()
    chain = SpringChain.from_node_chain("test", anchor, joints)
    sim = SpringSimulator(chains=[chain])
    sim.add_force(Gravity(force=vec3(2.0, 0.0, 0.0)))

    _run(sim, total_time=1.0)
    assert any(float(np.linalg.norm(j.angular_velocity)) > 0.01 for j in chain.joints)

    chain.reset()

    for joint in chain.joints:
        np.testing.assert_allclose(joint.angular_velocity, vec3(0.0, 0.0, 0.0), atol=1.0e-6)
        np.testing.assert_allclose(
            joint.node.transform.rotation, joint.rest_local_rotation, atol=1.0e-6
        )
        assert joint.initialized is False


def test_gravity_force_at_returns_constant() -> None:
    g = Gravity(force=vec3(0.0, -9.8, 0.0))
    f1 = g.force_at(vec3(1.0, 2.0, 3.0), 0.0)
    f2 = g.force_at(vec3(-5.0, 0.0, 7.0), 100.0)
    np.testing.assert_allclose(f1, f2, atol=1.0e-6)
    np.testing.assert_allclose(f1, vec3(0.0, -9.8, 0.0), atol=1.0e-6)


def test_wind_force_constant_when_no_turbulence() -> None:
    w = Wind(direction=vec3(1.0, 0.0, 0.0), speed=3.0)
    f = w.force_at(vec3(0.0, 0.0, 0.0), 0.5)
    np.testing.assert_allclose(f, vec3(3.0, 0.0, 0.0), atol=1.0e-6)


def test_wind_turbulence_oscillates_perpendicular() -> None:
    w = Wind(
        direction=vec3(1.0, 0.0, 0.0),
        speed=1.0,
        turbulence_amplitude=2.0,
        turbulence_frequency_hz=1.0,
    )
    samples = [w.force_at(vec3(0, 0, 0), t / 10.0) for t in range(20)]
    z_components = [float(s[2]) for s in samples]
    assert max(z_components) > 0.5
    assert min(z_components) < -0.5
    # X component (along wind direction) is unchanged by perpendicular jitter.
    for s in samples:
        assert math.isclose(float(s[0]), 1.0, abs_tol=1.0e-5)


def test_point_force_falls_off_with_distance() -> None:
    p = PointForce(source=vec3(0.0, 0.0, 0.0), magnitude=10.0, falloff_distance=2.0)
    near = p.force_at(vec3(0.5, 0.0, 0.0), 0.0)
    far = p.force_at(vec3(1.8, 0.0, 0.0), 0.0)
    outside = p.force_at(vec3(3.0, 0.0, 0.0), 0.0)
    assert float(np.linalg.norm(near)) > float(np.linalg.norm(far))
    np.testing.assert_allclose(outside, vec3(0.0, 0.0, 0.0), atol=1.0e-6)


def test_step_with_zero_or_negative_dt_is_noop() -> None:
    anchor, joints = _make_chain()
    chain = SpringChain.from_node_chain("test", anchor, joints)
    sim = SpringSimulator(chains=[chain])
    sim.add_force(Gravity(force=vec3(1.0, 0.0, 0.0)))

    sim.step(0.0)
    sim.step(-1.0)

    for joint in chain.joints:
        assert joint.initialized is False
        np.testing.assert_allclose(joint.angular_velocity, vec3(0.0, 0.0, 0.0), atol=1.0e-6)


def test_substepping_clamps_to_fixed_dt() -> None:
    """A single huge step should still produce a stable result via internal subdivision."""
    anchor, joints = _make_chain()
    chain = SpringChain.from_node_chain(
        "test",
        anchor,
        joints,
        params=SpringParams(stiffness=50.0, damping=2.0, inertia=0.03),
    )
    sim = SpringSimulator(chains=[chain], fixed_dt=1.0 / 240.0)
    sim.add_force(Gravity(force=vec3(0.5, 0.0, 0.0)))

    sim.step(0.5)  # one big step → many substeps internally

    for joint in chain.joints:
        assert np.all(np.isfinite(joint.angular_velocity))
        assert np.all(np.isfinite(joint.world_rotation))
        speed = float(np.linalg.norm(joint.angular_velocity))
        assert speed < 50.0, f"unstable {joint.node.name} ω={speed}"


def test_find_chain_by_name() -> None:
    anchor, joints = _make_chain()
    chain = SpringChain.from_node_chain("hair_C", anchor, joints)
    sim = SpringSimulator(chains=[chain])
    assert sim.find_chain("hair_C") is chain
    assert sim.find_chain("missing") is None


def test_disabled_chain_is_skipped() -> None:
    anchor, joints = _make_chain()
    chain = SpringChain.from_node_chain("test", anchor, joints)
    chain.enabled = False
    sim = SpringSimulator(chains=[chain])
    sim.add_force(Gravity(force=vec3(2.0, 0.0, 0.0)))

    _run(sim, total_time=1.0)

    for joint in chain.joints:
        assert joint.initialized is False
        np.testing.assert_allclose(joint.angular_velocity, vec3(0.0, 0.0, 0.0), atol=1.0e-6)


def test_empty_joint_chain_raises() -> None:
    anchor = Node(name="anchor", transform=Transform())
    with pytest.raises(ValueError, match="at least one joint"):
        SpringChain.from_node_chain("empty", anchor, [])


def _verify_joint_world_rotation_consistent(chain: SpringChain, anchor: Node) -> None:
    """Sanity helper: tracked world_rotation must match what the Node tree resolves to."""
    _ = node_world_pose(anchor)  # prove the anchor is reachable through the parent chain
    for joint in chain.joints:
        node_world_rot = node_world_pose(joint.node)[1]
        # Quaternion sign ambiguity — compare via |dot(a, b)| ≈ 1 instead of element-wise.
        dot = abs(float(np.dot(joint.world_rotation, node_world_rot)))
        assert dot > 0.999, f"{joint.node.name} world rotation desync"


def test_world_rotation_matches_node_tree_after_step() -> None:
    anchor, joints = _make_chain()
    chain = SpringChain.from_node_chain(
        "test",
        anchor,
        joints,
        params=SpringParams(stiffness=4.0, damping=2.0, inertia=0.05),
    )
    sim = SpringSimulator(chains=[chain])
    sim.add_force(Gravity(force=vec3(0.5, 0.0, 0.0)))

    _run(sim, total_time=0.5)
    _verify_joint_world_rotation_consistent(chain, anchor)


# --- Chain detection -------------------------------------------------------


def _build_named_chain(prefix: str, count: int, anchor: Node) -> list[Node]:
    """Build ``count`` joints named ``<prefix>_<i>`` parented anchor → 0 → 1 → ..."""
    parent = anchor
    nodes = []
    for i in range(count):
        node = Node(name=f"{prefix}_{i}", transform=Transform(translation=vec3(0.0, -0.1, 0.0)))
        parent.add_child(node)
        nodes.append(node)
        parent = node
    return nodes


def test_detect_single_chain() -> None:
    anchor = Node(name="head_anchor")
    joints = _build_named_chain("hair_C", 4, anchor)
    chains = detect_chains([anchor, *joints])
    assert len(chains) == 1
    assert chains[0].name == "hair_C"
    assert chains[0].anchor is anchor
    assert chains[0].joints == tuple(joints)


def test_detect_multiple_chains_sharing_anchor() -> None:
    anchor = Node(name="head_anchor")
    hair_c = _build_named_chain("hair_C", 4, anchor)
    hair_l = _build_named_chain("hair_L", 4, anchor)
    orn = _build_named_chain("orn", 2, anchor)
    chains = detect_chains([anchor, *hair_c, *hair_l, *orn])
    by_name = {c.name: c for c in chains}
    assert set(by_name) == {"hair_C", "hair_L", "orn"}
    assert all(c.anchor is anchor for c in chains)
    assert by_name["orn"].joints == tuple(orn)


def test_detect_skips_anchor_without_index_suffix() -> None:
    anchor = Node(name="head_anchor")  # no _<int> ⇒ never starts a chain
    joints = _build_named_chain("hair_C", 3, anchor)
    chains = detect_chains([anchor, *joints])
    assert len(chains) == 1
    assert "head_anchor" not in {c.name for c in chains}


def test_detect_accepts_uniformly_spaced_indices() -> None:
    """HSR FBX rigs index hair chains with odd-only or even-only steps
    (``BackHair1 → BackHair3 → BackHair5``); the bones still form a valid
    parent chain, just with auxiliary indices skipped between them. Accept
    any positive uniform spacing — the parent-linkage check guards against
    truly malformed groups."""
    anchor = Node(name="root")
    j0 = Node(name="bad_0", transform=Transform(translation=vec3(0.0, -0.1, 0.0)))
    j2 = Node(name="bad_2", transform=Transform(translation=vec3(0.0, -0.1, 0.0)))
    anchor.add_child(j0)
    j0.add_child(j2)
    chains = detect_chains([anchor, j0, j2])
    assert len(chains) == 1
    assert chains[0].name == "bad"
    assert chains[0].joints == (j0, j2)


def test_detect_rejects_non_uniform_indices() -> None:
    """Indices that aren't a clean arithmetic progression (e.g. [0, 1, 5])
    still get rejected — uniform spacing is required so an authored typo
    in the rig surfaces as a warning rather than silently producing a
    miss-rigged chain."""
    anchor = Node(name="root")
    j0 = Node(name="bad_0", transform=Transform(translation=vec3(0.0, -0.1, 0.0)))
    j1 = Node(name="bad_1", transform=Transform(translation=vec3(0.0, -0.1, 0.0)))
    j5 = Node(name="bad_5", transform=Transform(translation=vec3(0.0, -0.1, 0.0)))
    anchor.add_child(j0)
    j0.add_child(j1)
    j1.add_child(j5)
    chains = detect_chains([anchor, j0, j1, j5])
    assert chains == []


def test_detect_rejects_broken_parent_chain() -> None:
    anchor = Node(name="root")
    j0 = Node(name="bad_0", transform=Transform(translation=vec3(0.0, -0.1, 0.0)))
    j1 = Node(name="bad_1", transform=Transform(translation=vec3(0.0, -0.1, 0.0)))
    anchor.add_child(j0)
    anchor.add_child(j1)  # parent is anchor, not j0 — chain is broken
    chains = detect_chains([anchor, j0, j1])
    assert chains == []


def test_detect_rejects_chain_without_anchor() -> None:
    # joint 0 is a root node — no parent ⇒ cannot anchor
    j0 = Node(name="lone_0", transform=Transform())
    j1 = Node(name="lone_1", transform=Transform(translation=vec3(0.0, -0.1, 0.0)))
    j0.add_child(j1)
    chains = detect_chains([j0, j1])
    assert chains == []


def test_resolve_chain_params_picks_hair_profile() -> None:
    p = resolve_chain_params("hair_C")
    assert p == DEFAULT_CHAIN_PROFILES["hair"]


def test_resolve_chain_params_picks_orn_profile() -> None:
    p = resolve_chain_params("orn")
    assert p == DEFAULT_CHAIN_PROFILES["orn"]


def test_resolve_chain_params_falls_back_to_default() -> None:
    p = resolve_chain_params("custom_thing")
    assert p == SpringParams()


def test_spring_chain_component_is_attachable() -> None:
    """SpringChainComponent should be a regular Component subclass that nodes can hold."""
    anchor = Node(name="head_anchor")
    joints = _build_named_chain("hair_C", 3, anchor)
    component = SpringChainComponent(
        chain_name="hair_C",
        joints=tuple(joints),
        stiffness=10.0,
        damping=2.0,
        inertia=0.05,
    )
    anchor.add_component(component)
    assert anchor.components[0] is component
    assert component.chain_name == "hair_C"
    assert component.enabled is True
