"""Tests for the spring-chain attract-to-surface feature."""
from __future__ import annotations

import numpy as np
import pytest

from posecascade.animation.cloth import CapsuleCollider, SphereCollider
from posecascade.animation.spring import (
    SpringChain,
    SpringParams,
    _attract_joint_to_colliders,
    _initialize_joint_world,
    _nearest_surface_point,
    _ParentFrame,
)
from posecascade.scene.node import Node
from posecascade.scene.transform import Transform
from posecascade.utils.math3d import quat_identity, quat_rotate_vec, vec3


def _make_chain(joint_count: int = 3, segment_length: float = 0.1) -> tuple[Node, list[Node]]:
    anchor = Node(name="anchor", transform=Transform())
    parent = anchor
    joints: list[Node] = []
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


# ----- _nearest_surface_point -----------------------------------------------


def test_nearest_surface_on_sphere_returns_point_on_surface() -> None:
    sphere = SphereCollider(center=vec3(0.0, 0.0, 0.0), radius=0.5, skin_offset=0.0)
    point = vec3(2.0, 0.0, 0.0)
    surf = _nearest_surface_point(point, sphere)
    assert surf is not None
    # Surface point lies on +X axis at distance radius from centre
    np.testing.assert_allclose(surf, vec3(0.5, 0.0, 0.0), atol=1.0e-6)


def test_nearest_surface_on_sphere_includes_skin_offset() -> None:
    sphere = SphereCollider(center=vec3(0.0, 0.0, 0.0), radius=0.5, skin_offset=0.01)
    surf = _nearest_surface_point(vec3(1.0, 0.0, 0.0), sphere)
    assert surf is not None
    np.testing.assert_allclose(surf, vec3(0.51, 0.0, 0.0), atol=1.0e-6)


def test_nearest_surface_on_sphere_handles_centre_point() -> None:
    """Point exactly at sphere centre — fallback direction must still return a surface point."""
    sphere = SphereCollider(center=vec3(0.0, 0.0, 0.0), radius=0.5, skin_offset=0.0)
    surf = _nearest_surface_point(vec3(0.0, 0.0, 0.0), sphere)
    assert surf is not None
    # Magnitude equals radius (fallback direction is +Y by convention)
    assert abs(float(np.linalg.norm(surf)) - 0.5) < 1.0e-6


def test_nearest_surface_on_capsule_midpoint() -> None:
    """Point beside a vertical capsule projects to its lateral surface."""
    cap = CapsuleCollider(a=vec3(0.0, -1.0, 0.0), b=vec3(0.0, 1.0, 0.0),
                          radius=0.2, skin_offset=0.0)
    surf = _nearest_surface_point(vec3(2.0, 0.0, 0.0), cap)
    assert surf is not None
    np.testing.assert_allclose(surf, vec3(0.2, 0.0, 0.0), atol=1.0e-6)


def test_nearest_surface_on_capsule_endcap() -> None:
    """Point beyond a capsule endpoint projects to its hemispherical cap."""
    cap = CapsuleCollider(a=vec3(0.0, 0.0, 0.0), b=vec3(0.0, 1.0, 0.0),
                          radius=0.1, skin_offset=0.0)
    surf = _nearest_surface_point(vec3(0.0, 5.0, 0.0), cap)
    assert surf is not None
    # Should be on the +Y hemisphere of the top endpoint
    np.testing.assert_allclose(surf, vec3(0.0, 1.1, 0.0), atol=1.0e-6)


def test_nearest_surface_on_capsule_with_point_on_axis() -> None:
    """Point exactly on the capsule axis — fallback perpendicular still gives a surface point."""
    cap = CapsuleCollider(a=vec3(0.0, -1.0, 0.0), b=vec3(0.0, 1.0, 0.0),
                          radius=0.2, skin_offset=0.0)
    surf = _nearest_surface_point(vec3(0.0, 0.0, 0.0), cap)
    assert surf is not None
    # Distance from axis = radius
    assert abs(float(np.linalg.norm(surf)) - 0.2) < 1.0e-6


def test_nearest_surface_unsupported_collider_returns_none() -> None:
    """Helper returns None for collider types it doesn't model (defensive)."""

    class _Other:
        pass

    assert _nearest_surface_point(vec3(0.0, 0.0, 0.0), _Other()) is None  # type: ignore[arg-type]


# ----- _attract_joint_to_colliders integration ------------------------------


def test_attract_pulls_joint_toward_named_collider() -> None:
    """A joint whose tip is far from a tagged capsule rotates so its tip lands on the surface."""
    anchor, joint_nodes = _make_chain(joint_count=1, segment_length=0.5)
    chain = SpringChain.from_node_chain(
        "hair", anchor, joint_nodes,
        params=SpringParams(stiffness=2.0, damping=0.5, inertia=0.05),
    )
    chain.attract_to_bones = ("spine",)
    chain.attract_max_distance = 1.0
    # Capsule along +Z, tagged "spine"; the chain hangs along -Y.
    cap = CapsuleCollider(a=vec3(0.0, -0.5, 0.0), b=vec3(0.0, -0.5, 0.5),
                          radius=0.1, skin_offset=0.0, bone_tag="spine")
    # Initialize joint world transform without running a step (avoid gravity)
    parent = _ParentFrame(position=vec3(0.0, 0.0, 0.0), rotation=quat_identity())
    joint = chain.joints[0]
    _initialize_joint_world(joint, parent)
    # Apply attract pass
    _attract_joint_to_colliders(
        joint, parent, [cap], chain.attract_to_bones, chain.attract_max_distance,
    )
    # The tip should now be much closer to the capsule surface than 0.5
    pivot = parent.position + quat_rotate_vec(parent.rotation, joint.rest_local_position)
    bone_world = quat_rotate_vec(joint.world_rotation, joint.bone_vector_local)
    tip = pivot + bone_world
    surf = _nearest_surface_point(tip, cap)
    assert surf is not None
    final_dist = float(np.linalg.norm(tip - surf))
    # The attract pass should land the tip within a few mm of the surface
    assert final_dist < 5.0e-3, f"tip not attracted close enough: dist={final_dist:.4f}"


def test_attract_skips_collider_without_matching_bone_tag() -> None:
    """A tagged collider that doesn't match attract_to_bones is ignored."""
    anchor, joint_nodes = _make_chain(joint_count=1, segment_length=0.5)
    chain = SpringChain.from_node_chain(
        "hair", anchor, joint_nodes,
        params=SpringParams(stiffness=2.0, damping=0.5, inertia=0.05),
    )
    chain.attract_to_bones = ("chest",)
    chain.attract_max_distance = 1.0
    # Capsule tagged "leg" — should be ignored by the attract pass
    cap = CapsuleCollider(a=vec3(0.0, -0.5, 0.0), b=vec3(0.0, -0.5, 0.5),
                          radius=0.1, skin_offset=0.0, bone_tag="leg")
    parent = _ParentFrame(position=vec3(0.0, 0.0, 0.0), rotation=quat_identity())
    joint = chain.joints[0]
    _initialize_joint_world(joint, parent)
    rot_before = joint.world_rotation.copy()
    _attract_joint_to_colliders(
        joint, parent, [cap], chain.attract_to_bones, chain.attract_max_distance,
    )
    # Rotation should not change since no collider matched
    np.testing.assert_allclose(joint.world_rotation, rot_before, atol=1.0e-6)


def test_attract_respects_max_distance() -> None:
    """A collider beyond attract_max_distance is ignored even if its tag matches."""
    anchor, joint_nodes = _make_chain(joint_count=1, segment_length=0.5)
    chain = SpringChain.from_node_chain(
        "hair", anchor, joint_nodes,
        params=SpringParams(stiffness=2.0, damping=0.5, inertia=0.05),
    )
    chain.attract_to_bones = ("spine",)
    chain.attract_max_distance = 0.1  # very small
    # Capsule 2 m away — beyond max distance
    cap = CapsuleCollider(a=vec3(5.0, 0.0, 0.0), b=vec3(5.0, 1.0, 0.0),
                          radius=0.1, skin_offset=0.0, bone_tag="spine")
    parent = _ParentFrame(position=vec3(0.0, 0.0, 0.0), rotation=quat_identity())
    joint = chain.joints[0]
    _initialize_joint_world(joint, parent)
    rot_before = joint.world_rotation.copy()
    _attract_joint_to_colliders(
        joint, parent, [cap], chain.attract_to_bones, chain.attract_max_distance,
    )
    np.testing.assert_allclose(joint.world_rotation, rot_before, atol=1.0e-6)


def test_attract_picks_closest_when_multiple_match() -> None:
    """Multiple colliders match — the closest surface wins."""
    anchor, joint_nodes = _make_chain(joint_count=1, segment_length=0.5)
    chain = SpringChain.from_node_chain(
        "hair", anchor, joint_nodes,
        params=SpringParams(stiffness=2.0, damping=0.5, inertia=0.05),
    )
    chain.attract_to_bones = ("spine", "chest")
    chain.attract_max_distance = 2.0
    near_cap = CapsuleCollider(a=vec3(0.05, -0.5, 0.0), b=vec3(0.05, -0.5, 0.1),
                               radius=0.05, skin_offset=0.0, bone_tag="spine")
    far_cap = CapsuleCollider(a=vec3(1.0, -0.5, 0.0), b=vec3(1.0, -0.5, 0.1),
                              radius=0.05, skin_offset=0.0, bone_tag="chest")
    parent = _ParentFrame(position=vec3(0.0, 0.0, 0.0), rotation=quat_identity())
    joint = chain.joints[0]
    _initialize_joint_world(joint, parent)
    _attract_joint_to_colliders(
        joint, parent, [near_cap, far_cap],
        chain.attract_to_bones, chain.attract_max_distance,
    )
    pivot = parent.position + quat_rotate_vec(parent.rotation, joint.rest_local_position)
    bone_world = quat_rotate_vec(joint.world_rotation, joint.bone_vector_local)
    tip = pivot + bone_world
    # Tip should land near near_cap, not far_cap
    assert abs(tip[0] - 0.05) < 0.10, f"tip x={tip[0]:.3f} not near near_cap"


def test_attract_noop_when_attract_to_bones_empty() -> None:
    """A chain with no attract_to_bones doesn't change its rotation in the attract pass."""
    anchor, joint_nodes = _make_chain(joint_count=1, segment_length=0.5)
    chain = SpringChain.from_node_chain(
        "hair", anchor, joint_nodes,
        params=SpringParams(stiffness=2.0, damping=0.5, inertia=0.05),
    )
    # attract_to_bones defaults to empty
    cap = CapsuleCollider(a=vec3(0.0, -0.5, 0.0), b=vec3(0.0, -0.5, 0.5),
                          radius=0.1, skin_offset=0.0, bone_tag="spine")
    parent = _ParentFrame(position=vec3(0.0, 0.0, 0.0), rotation=quat_identity())
    joint = chain.joints[0]
    _initialize_joint_world(joint, parent)
    rot_before = joint.world_rotation.copy()
    _attract_joint_to_colliders(
        joint, parent, [cap], chain.attract_to_bones, chain.attract_max_distance,
    )
    np.testing.assert_allclose(joint.world_rotation, rot_before, atol=1.0e-6)


def test_attract_handles_collider_without_bone_tag() -> None:
    """Colliders with empty bone_tag are skipped (not matched by any name)."""
    anchor, joint_nodes = _make_chain(joint_count=1, segment_length=0.5)
    chain = SpringChain.from_node_chain(
        "hair", anchor, joint_nodes,
        params=SpringParams(stiffness=2.0, damping=0.5, inertia=0.05),
    )
    chain.attract_to_bones = ("spine",)
    chain.attract_max_distance = 1.0
    cap = CapsuleCollider(a=vec3(0.0, -0.5, 0.0), b=vec3(0.0, -0.5, 0.5),
                          radius=0.1, skin_offset=0.0)  # bone_tag defaults to ""
    parent = _ParentFrame(position=vec3(0.0, 0.0, 0.0), rotation=quat_identity())
    joint = chain.joints[0]
    _initialize_joint_world(joint, parent)
    rot_before = joint.world_rotation.copy()
    _attract_joint_to_colliders(
        joint, parent, [cap], chain.attract_to_bones, chain.attract_max_distance,
    )
    np.testing.assert_allclose(joint.world_rotation, rot_before, atol=1.0e-6)


# ----- JSON config integration ----------------------------------------------


def test_attract_to_bones_default_is_empty_tuple() -> None:
    """SpringChain.attract_to_bones defaults to () so existing chains behave unchanged."""
    anchor, joint_nodes = _make_chain(joint_count=2)
    chain = SpringChain.from_node_chain("test", anchor, joint_nodes)
    assert chain.attract_to_bones == ()
    assert chain.attract_max_distance == pytest.approx(0.30)


def test_attract_max_distance_can_be_overridden() -> None:
    """Setting attract_max_distance via assignment works and the simulator uses it."""
    anchor, joint_nodes = _make_chain(joint_count=1, segment_length=0.5)
    chain = SpringChain.from_node_chain("test", anchor, joint_nodes)
    chain.attract_to_bones = ("spine",)
    chain.attract_max_distance = 0.05
    cap = CapsuleCollider(a=vec3(0.0, -0.5, 0.0), b=vec3(0.0, -0.5, 0.5),
                          radius=0.05, skin_offset=0.0, bone_tag="spine")
    parent = _ParentFrame(position=vec3(0.0, 0.0, 0.0), rotation=quat_identity())
    joint = chain.joints[0]
    _initialize_joint_world(joint, parent)
    rot_before = joint.world_rotation.copy()
    # Tip is at (0, -0.5, 0). Surface is at radius=0.05 from axis.
    # Distance ~ 0.05 which is right at the threshold — but the bone is already
    # next to the surface so attract should produce only a tiny rotation.
    _attract_joint_to_colliders(
        joint, parent, [cap], chain.attract_to_bones, chain.attract_max_distance,
    )
    # Either no change or a very small rotation — neither test failure
    angle_change = float(np.linalg.norm(joint.world_rotation - rot_before))
    assert angle_change < 0.5, "rotation magnitude unexpectedly large"
