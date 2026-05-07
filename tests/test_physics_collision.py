"""Tests for the rigid-rigid collision detection + impulse resolver."""
from __future__ import annotations

import numpy as np

from posecascade.physics import (
    PhysicsMode,
    PhysicsScene,
    PhysicsWorld,
    RigidBody,
    RigidShape,
)
from posecascade.physics.collision import (
    _box_box,
    _capsule_capsule,
    _quat_to_matrix3,
    _sphere_box,
    _sphere_sphere,
    find_contacts,
)
from posecascade.physics.world import _BodyState
from posecascade.utils.math3d import quat_from_euler_xyz, vec3


def _make_body(
    *,
    name: str = "b",
    shape: RigidShape = RigidShape.SPHERE,
    size: tuple[float, float, float] = (0.5, 0.0, 0.0),
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mode: PhysicsMode = PhysicsMode.DYNAMIC,
    mass: float = 1.0,
    restitution: float = 0.0,
    group: int = 0,
    non_collision_mask: int = 0,
) -> RigidBody:
    return RigidBody(
        name=name, bone_index=-1, group=group,
        non_collision_mask=non_collision_mask, shape=shape, size=size,
        position=position, rotation=rotation, mass=mass,
        linear_damping=0.0, angular_damping=0.0,
        restitution=restitution, friction=0.5, physics_mode=mode,
    )


def _state(body: RigidBody) -> _BodyState:
    """Build a ready-to-test ``_BodyState`` from a static :class:`RigidBody`."""
    return _BodyState(
        body=body,
        position=vec3(*body.position),
        orientation=quat_from_euler_xyz(*body.rotation),
        linear_velocity=vec3(0, 0, 0),
        angular_velocity=vec3(0, 0, 0),
        rest_position=vec3(*body.position),
        rest_orientation=quat_from_euler_xyz(*body.rotation),
        bone_node=None,
    )


# ----- narrowphase ------------------------------------------------------
def test_sphere_sphere_overlap_yields_contact_normal_along_centres() -> None:
    a = _state(_make_body(name="a", size=(0.5, 0, 0), position=(0, 0, 0)))
    b = _state(_make_body(name="b", size=(0.5, 0, 0), position=(0.6, 0, 0)))
    contact = _sphere_sphere(0, 1, a, b)
    assert contact is not None
    np.testing.assert_allclose(contact.normal, [1, 0, 0], atol=1e-5)
    assert abs(contact.depth - 0.4) < 1e-5


def test_sphere_sphere_separated_returns_none() -> None:
    a = _state(_make_body(name="a", size=(0.25, 0, 0), position=(0, 0, 0)))
    b = _state(_make_body(name="b", size=(0.25, 0, 0), position=(2.0, 0, 0)))
    assert _sphere_sphere(0, 1, a, b) is None


def test_sphere_box_face_contact() -> None:
    """Sphere just outside a unit box (half-extents 0.5) gets a +X normal."""
    sphere = _state(_make_body(
        name="s", shape=RigidShape.SPHERE, size=(0.3, 0, 0), position=(0.6, 0, 0),
    ))
    box = _state(_make_body(
        name="b", shape=RigidShape.BOX, size=(0.5, 0.5, 0.5), position=(0, 0, 0),
    ))
    contact = _sphere_box(0, 1, sphere, box)
    assert contact is not None
    # Normal points from sphere → box, so −X here.
    np.testing.assert_allclose(contact.normal, [-1, 0, 0], atol=1e-5)
    assert abs(contact.depth - 0.2) < 1e-5


def test_sphere_box_corner_contact() -> None:
    sphere_pos = (0.6, 0.6, 0.6)
    sphere = _state(_make_body(
        name="s", shape=RigidShape.SPHERE, size=(0.3, 0, 0), position=sphere_pos,
    ))
    box = _state(_make_body(
        name="b", shape=RigidShape.BOX, size=(0.5, 0.5, 0.5), position=(0, 0, 0),
    ))
    contact = _sphere_box(0, 1, sphere, box)
    assert contact is not None
    # Normal magnitude is 1; direction is the corner-to-sphere axis.
    assert abs(np.linalg.norm(contact.normal) - 1.0) < 1e-5
    assert contact.depth > 0.0


def test_capsule_capsule_collinear_overlap_yields_contact() -> None:
    a = _state(_make_body(
        name="a", shape=RigidShape.CAPSULE, size=(0.25, 1.0, 0),
        position=(0, 0, 0),
    ))
    b = _state(_make_body(
        name="b", shape=RigidShape.CAPSULE, size=(0.25, 1.0, 0),
        position=(0.4, 0, 0),
    ))
    contact = _capsule_capsule(0, 1, a, b)
    assert contact is not None
    np.testing.assert_allclose(contact.normal, [1, 0, 0], atol=1e-5)
    assert contact.depth > 0.0


def test_box_box_axis_aligned_overlap() -> None:
    a = _state(_make_body(
        name="a", shape=RigidShape.BOX, size=(0.5, 0.5, 0.5), position=(0, 0, 0),
    ))
    b = _state(_make_body(
        name="b", shape=RigidShape.BOX, size=(0.5, 0.5, 0.5), position=(0.8, 0, 0),
    ))
    contact = _box_box(0, 1, a, b)
    assert contact is not None
    np.testing.assert_allclose(np.abs(contact.normal), [1, 0, 0], atol=1e-5)
    assert abs(contact.depth - 0.2) < 1e-5


def test_box_box_separated_returns_none() -> None:
    a = _state(_make_body(
        name="a", shape=RigidShape.BOX, size=(0.5, 0.5, 0.5), position=(0, 0, 0),
    ))
    b = _state(_make_body(
        name="b", shape=RigidShape.BOX, size=(0.5, 0.5, 0.5), position=(2.0, 0, 0),
    ))
    assert _box_box(0, 1, a, b) is None


def test_quat_to_matrix3_identity() -> None:
    rot = _quat_to_matrix3(np.array([0, 0, 0, 1], dtype=np.float32))
    np.testing.assert_allclose(rot, np.eye(3), atol=1e-5)


# ----- broadphase + filtering ------------------------------------------
def test_broadphase_skips_non_overlapping_aabbs() -> None:
    a = _state(_make_body(name="a", position=(0, 0, 0), size=(0.25, 0, 0)))
    b = _state(_make_body(name="b", position=(10.0, 0, 0), size=(0.25, 0, 0)))
    assert find_contacts([a, b]) == []


def test_broadphase_skips_pairs_in_filtered_groups() -> None:
    """Body A's mask flag for group 1 means A↔B is filtered out."""
    a = _state(_make_body(name="a", group=0, non_collision_mask=0b10, position=(0, 0, 0)))
    b = _state(_make_body(name="b", group=1, position=(0.4, 0, 0)))
    assert find_contacts([a, b]) == []


def test_broadphase_finds_overlapping_pair() -> None:
    a = _state(_make_body(name="a", position=(0, 0, 0)))
    b = _state(_make_body(name="b", position=(0.6, 0, 0)))
    contacts = find_contacts([a, b])
    assert len(contacts) == 1


def test_two_kinematic_bodies_skipped() -> None:
    a = _state(_make_body(
        name="a", mode=PhysicsMode.KINEMATIC, position=(0, 0, 0),
    ))
    b = _state(_make_body(
        name="b", mode=PhysicsMode.KINEMATIC, position=(0.4, 0, 0),
    ))
    assert find_contacts([a, b]) == []


# ----- end-to-end through PhysicsWorld ----------------------------------
def test_dynamic_sphere_bounces_off_kinematic_floor() -> None:
    """Falling dynamic sphere hits a kinematic floor box and stops sinking."""
    sphere = _make_body(
        name="ball", shape=RigidShape.SPHERE, size=(0.5, 0, 0),
        position=(0, 1.5, 0), mode=PhysicsMode.DYNAMIC, mass=1.0, restitution=0.0,
    )
    floor = _make_body(
        name="floor", shape=RigidShape.BOX, size=(5.0, 0.25, 5.0),
        position=(0, 0, 0), mode=PhysicsMode.KINEMATIC, mass=0.0, restitution=0.0,
    )
    world = PhysicsWorld(scene=PhysicsScene(bodies=(sphere, floor)))
    elapsed = 0.0
    while elapsed < 2.0:
        world.step(0.05)
        elapsed += 0.05
    final_y = float(world.body_position(0)[1])
    # Resting on top of the floor: floor top at y=0.25, sphere centre
    # should be ≈ 0.75 (radius 0.5 above the contact). Allow ±0.1 m for
    # solver slop + Baumgarte residual.
    assert 0.6 < final_y < 0.9, f"sphere did not rest on floor: y={final_y}"


def test_two_dynamic_spheres_separate_after_overlap() -> None:
    """Two interpenetrating spheres get pushed apart by the impulse solver."""
    a = _make_body(name="a", size=(0.5, 0, 0), position=(-0.1, 0, 0), mass=1.0)
    b = _make_body(name="b", size=(0.5, 0, 0), position=(0.1, 0, 0), mass=1.0)
    world = PhysicsWorld(scene=PhysicsScene(bodies=(a, b)))
    world.set_gravity((0, 0, 0))
    world.step(0.1)
    # After one tick of correction the centre-to-centre distance should
    # have grown toward (but not necessarily reached) ``2 * radius``.
    delta_x = float(world.body_position(1)[0] - world.body_position(0)[0])
    assert delta_x > 0.2, f"spheres did not separate: dx={delta_x}"


def test_collision_emits_contacts_on_world() -> None:
    a = _make_body(name="a", size=(0.5, 0, 0), position=(0, 0, 0))
    b = _make_body(name="b", size=(0.5, 0, 0), position=(0.6, 0, 0))
    world = PhysicsWorld(scene=PhysicsScene(bodies=(a, b)))
    world.set_gravity((0, 0, 0))
    world.step(1 / 60)
    contacts = world.last_contacts()
    assert len(contacts) == 1
    assert contacts[0].a_index == 0
    assert contacts[0].b_index == 1
