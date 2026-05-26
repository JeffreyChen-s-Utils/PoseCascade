"""Angular-cone constraint on spring chains.

Verifies the ``max_swing_rad`` hard limit clamps the joint's rotation
from rest to at most that many radians, no matter how much external
force is applied. This is the engine-level equivalent of the
"Angle Limit" constraint in Magica Cloth / HSR / Genshin spring-bone
solvers, and it is what stops hair from physically swinging through
the head in extreme poses.
"""
from __future__ import annotations

import math

import numpy as np

from posecascade.animation.spring import (
    Gravity,
    SpringChain,
    SpringParams,
    SpringSimulator,
)
from posecascade.scene.node import Node
from posecascade.utils.math3d import quat_to_axis_angle, vec3


def _chain(max_swing_rad: float) -> SpringChain:
    """3-joint chain pointing +X, anchored at origin."""
    anchor = Node(name="anchor")
    j0 = Node(name="j0")
    j1 = Node(name="j1")
    j2 = Node(name="j2")
    j0.transform.set_translation(vec3(0.1, 0, 0))
    j1.transform.set_translation(vec3(0.1, 0, 0))
    j2.transform.set_translation(vec3(0.1, 0, 0))
    anchor.add_child(j0)
    j0.add_child(j1)
    j1.add_child(j2)
    return SpringChain.from_node_chain(
        "hair_C", anchor, [j0, j1, j2],
        params=SpringParams(
            stiffness=0.5, damping=0.1, inertia=1.0,
            max_swing_rad=max_swing_rad,
        ),
    )


def test_cone_clamps_swing_to_max_angle() -> None:
    """Strong gravity for 5 sim seconds — without the cone the chain
    would settle at ~90° (gravity-aligned). With a 30° cone limit the
    first joint's swing-from-rest must NOT exceed ~30° (small numerical
    overshoot is allowed)."""
    sim = SpringSimulator()
    sim.add_force(Gravity(force=vec3(0.0, -50.0, 0.0)))
    chain = _chain(max_swing_rad=math.radians(30.0))
    sim.add_chain(chain)
    # Drive for plenty of sim time to saturate against the cone wall.
    for _ in range(60):
        sim.step(1.0 / 60.0)
    # The first joint's local rotation from rest should be clamped.
    # Rest local rotation is identity (no offset baked), so the joint's
    # local rotation IS the swing-from-rest delta.
    joint = chain.joints[0]
    _, angle = quat_to_axis_angle(joint.node.transform.rotation)
    # Allow 2° of numerical overshoot (one substep can briefly poke past
    # the boundary before the next clamp catches it).
    assert angle <= math.radians(32.0), (
        f"Cone clamp failed: joint swing {math.degrees(angle):.1f}° > 32°"
    )


def test_unlimited_cone_allows_full_swing() -> None:
    """Without the cone (default π) the chain settles much further."""
    sim = SpringSimulator()
    sim.add_force(Gravity(force=vec3(0.0, -50.0, 0.0)))
    chain = _chain(max_swing_rad=math.pi)
    sim.add_chain(chain)
    for _ in range(60):
        sim.step(1.0 / 60.0)
    joint = chain.joints[0]
    _, angle = quat_to_axis_angle(joint.node.transform.rotation)
    # Without the cone the joint swings well past 30°.
    assert angle > math.radians(45.0), (
        f"Expected unrestricted chain to swing past 45° under gravity "
        f"(got {math.degrees(angle):.1f}°)"
    )


def test_cone_does_not_compound_across_chain() -> None:
    """Cone clamp is measured in the anchor's frame, NOT live-parent's.

    With a 20° cone on a 3-joint chain, the TIP joint must still be
    within 20° of its rest world direction — it cannot accumulate
    20° + 20° + 20° = 60° just because each parent joint also rotated
    20°. This is the "anchor-frame" semantics that HSR / Magica Cloth
    enforce; the live-parent variant would allow the tip to drift
    much further under sustained gravity.
    """
    sim = SpringSimulator()
    sim.add_force(Gravity(force=vec3(0.0, -100.0, 0.0)))
    chain = _chain(max_swing_rad=math.radians(20.0))
    sim.add_chain(chain)
    # Drive hard for long enough that every joint hits its cone wall.
    for _ in range(120):
        sim.step(1.0 / 60.0)
    tip = chain.joints[-1]
    # Tip's anchor-frame rest direction is identity here (chain points
    # +X with identity local rotations). Its world rotation must be
    # within 20° of identity — assert each component.
    _, tip_angle = quat_to_axis_angle(tip.world_rotation)
    assert tip_angle <= math.radians(22.0), (
        f"Tip joint swung {math.degrees(tip_angle):.1f}° from rest — "
        f"cone semantics are leaking via compounding (should be <= 22°)"
    )


def test_gravity_override_replaces_global_gravity() -> None:
    """Per-chain ``gravity_override`` overrides world gravity for that
    chain only — used by extreme poses (dog-crawl, lying) where world-
    down would push hair through the body. With high stiffness + damping
    the chain converges to its drape-aligned rest pose within ~1.5 sec;
    the bone direction must point along the override direction (-Z),
    not the global gravity direction (-Y).
    """
    sim = SpringSimulator()
    sim.add_force(Gravity(force=vec3(0.0, -50.0, 0.0)))
    # High stiffness/damping for fast convergence — matches what
    # dog_crawl.json uses in practice.
    anchor = Node(name="anchor")
    j0 = Node(name="j0")
    j0.transform.set_translation(vec3(0.1, 0, 0))
    anchor.add_child(j0)
    chain = SpringChain.from_node_chain(
        "hair_C", anchor, [j0],
        params=SpringParams(
            stiffness=50.0, damping=5.0, inertia=1.0,
            max_swing_rad=math.pi,
        ),
    )
    chain.gravity_override = vec3(0.0, 0.0, -50.0)
    sim.add_chain(chain)
    for _ in range(120):
        sim.step(1.0 / 60.0)
    joint = chain.joints[0]
    bone_local = joint.bone_vector_local / float(np.linalg.norm(joint.bone_vector_local))
    from posecascade.utils.math3d import quat_rotate_vec  # noqa: PLC0415
    bone_world = quat_rotate_vec(joint.world_rotation, bone_local)
    assert bone_world[2] < -0.85, (
        f"Expected chain to swing toward -Z under gravity_override; "
        f"got bone_world={bone_world.round(3)}"
    )


def test_cone_zeroes_velocity_into_wall() -> None:
    """After the cone clamps, the velocity component pushing into the
    wall is zeroed — the joint slides along the boundary instead of
    bouncing or stalling with stored energy."""
    sim = SpringSimulator()
    sim.add_force(Gravity(force=vec3(0.0, -50.0, 0.0)))
    chain = _chain(max_swing_rad=math.radians(20.0))
    sim.add_chain(chain)
    # Hit the wall hard.
    for _ in range(30):
        sim.step(1.0 / 60.0)
    joint = chain.joints[0]
    # Compute the swing axis. The component of angular_velocity along
    # that axis (pushing further into the wall) should be ~zero.
    _, angle = quat_to_axis_angle(joint.node.transform.rotation)
    if angle < math.radians(18.0):
        # Hadn't hit the wall yet on this run — re-run for more steps.
        for _ in range(60):
            sim.step(1.0 / 60.0)
    axis, _ = quat_to_axis_angle(joint.node.transform.rotation)
    omega_along = float(np.dot(joint.angular_velocity, axis))
    assert omega_along < 0.5, (
        f"Velocity into the cone wall should be clamped near 0, "
        f"got {omega_along:.2f} rad/s"
    )
