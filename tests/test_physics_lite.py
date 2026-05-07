"""Tests for the script-facing :mod:`posecascade.scripting.physics_lite` API."""
from __future__ import annotations

import numpy as np
import pytest

from posecascade.animation.cloth import ClothGravity, ClothWind
from posecascade.animation.cloth_host import ClothHost
from posecascade.animation.physics_host import PhysicsHost
from posecascade.animation.spring import Gravity, PointForce, Wind
from posecascade.assets.types import ImportedScene, Mesh
from posecascade.scene.component import MeshRefComponent, SpringChainComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.scene.transform import Transform
from posecascade.scripting.physics_lite import ChainHandle, ClothPieceHandle, PhysicsLite
from posecascade.utils.math3d import vec3


def _host_with_chain(name: str = "hair_C") -> tuple[PhysicsHost, Node]:
    anchor = Node(name="head_anchor")
    parent = anchor
    joints = []
    for i in range(3):
        joint = Node(name=f"{name}_{i}", transform=Transform(translation=vec3(0.1, 0.0, 0.0)))
        parent.add_child(joint)
        joints.append(joint)
        parent = joint
    anchor.add_component(SpringChainComponent(chain_name=name, joints=tuple(joints)))
    scene = Scene(root=anchor)
    host = PhysicsHost()
    host.install_default_forces()
    host.register_scene(scene)
    return host, anchor


def test_set_gravity_replaces_default_gravity() -> None:
    host, _ = _host_with_chain()
    api = PhysicsLite(host)

    api.set_gravity((1.0, -3.0, 0.0))

    gravity_forces = [f for f in host.simulator.global_forces if isinstance(f, Gravity)]
    assert len(gravity_forces) == 1
    np.testing.assert_allclose(gravity_forces[0].force, vec3(1.0, -3.0, 0.0), atol=1.0e-6)


def test_set_gravity_installs_when_missing() -> None:
    host = PhysicsHost()  # no install_default_forces
    api = PhysicsLite(host)

    api.set_gravity((0.0, -9.8, 0.0))

    gravity_forces = [f for f in host.simulator.global_forces if isinstance(f, Gravity)]
    assert len(gravity_forces) == 1


def test_add_wind_returns_mutable_handle() -> None:
    host, _ = _host_with_chain()
    api = PhysicsLite(host)

    wind = api.add_wind(direction=(1.0, 0.0, 0.0), speed=2.0)
    assert isinstance(wind, Wind)
    assert wind.speed == 2.0

    # Caller can mutate after construction.
    wind.speed = 5.0
    assert host.simulator.global_forces[-1].speed == 5.0  # type: ignore[union-attr]


def test_add_point_force() -> None:
    host, _ = _host_with_chain()
    api = PhysicsLite(host)

    pf = api.add_point_force(source=(0.0, 0.0, 0.0), magnitude=3.0, falloff_distance=1.5)
    assert isinstance(pf, PointForce)
    assert pf.magnitude == 3.0
    assert pf.falloff_distance == 1.5


def test_get_chain_returns_handle() -> None:
    host, _ = _host_with_chain("hair_X")
    api = PhysicsLite(host)
    handle = api.get_chain("hair_X")
    assert isinstance(handle, ChainHandle)
    assert handle.name == "hair_X"


def test_get_chain_returns_none_when_missing() -> None:
    host, _ = _host_with_chain("hair_X")
    api = PhysicsLite(host)
    assert api.get_chain("does_not_exist") is None


def test_chain_handle_setters_apply() -> None:
    host, _ = _host_with_chain()
    api = PhysicsLite(host)
    handle = api.get_chain("hair_C")
    assert handle is not None

    handle.stiffness = 42.0
    handle.damping = 7.5
    handle.set_inertia(0.123)
    handle.enabled = False

    chain = host.find_chain("hair_C")
    assert chain is not None
    assert chain.stiffness == pytest.approx(42.0)
    assert chain.damping == pytest.approx(7.5)
    assert all(j.inertia == pytest.approx(0.123) for j in chain.joints)
    assert chain.enabled is False


def test_apply_impulse_changes_angular_velocity() -> None:
    host, _ = _host_with_chain()
    api = PhysicsLite(host)
    handle = api.get_chain("hair_C")
    assert handle is not None
    chain = host.find_chain("hair_C")
    assert chain is not None
    # Initialise tracked state by stepping once.
    host.tick(1.0 / 60.0)

    handle.apply_impulse(axis=(0.0, 0.0, 1.0), magnitude=2.5, joint_index=-1)

    tip = chain.joints[-1]
    assert tip.angular_velocity[2] >= 2.5 - 1.0e-3  # at least the impulse magnitude in Z


def test_apply_impulse_invalid_index_raises() -> None:
    host, _ = _host_with_chain()
    api = PhysicsLite(host)
    handle = api.get_chain("hair_C")
    assert handle is not None
    with pytest.raises(IndexError):
        handle.apply_impulse(axis=(0.0, 0.0, 1.0), magnitude=1.0, joint_index=99)


def test_chain_names_lists_registered_chains() -> None:
    host, _ = _host_with_chain("hair_A")
    api = PhysicsLite(host)
    assert "hair_A" in api.chain_names()


def test_chains_iterates_handles() -> None:
    host, _ = _host_with_chain()
    api = PhysicsLite(host)
    handles = list(api.chains())
    assert len(handles) == 1
    assert all(isinstance(h, ChainHandle) for h in handles)


def test_reset_all_returns_chains_to_rest() -> None:
    host, _ = _host_with_chain()
    api = PhysicsLite(host)
    handle = api.get_chain("hair_C")
    assert handle is not None
    chain = host.find_chain("hair_C")
    assert chain is not None
    # Disturb chain
    host.tick(1.0 / 60.0)
    handle.apply_impulse(axis=(0.0, 0.0, 1.0), magnitude=5.0, joint_index=-1)

    api.reset_all()

    for joint in chain.joints:
        np.testing.assert_allclose(joint.angular_velocity, vec3(0.0, 0.0, 0.0), atol=1.0e-6)


def test_to_vec3_rejects_wrong_shape() -> None:
    host, _ = _host_with_chain()
    api = PhysicsLite(host)
    with pytest.raises(ValueError, match="3-component"):
        api.set_gravity((1.0, 2.0))


# --- Cloth API ---------------------------------------------------------------


def _grid_mesh(rows: int = 3, cols: int = 3) -> Mesh:
    positions = np.zeros((rows * cols, 3), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            positions[r * cols + c] = [c * 0.1, -r * 0.1, 0.0]
    triangles = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v00 = r * cols + c
            v01 = r * cols + (c + 1)
            v10 = (r + 1) * cols + c
            v11 = (r + 1) * cols + (c + 1)
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
    indices = np.array(triangles, dtype=np.uint32).reshape(-1)
    return Mesh(name="cape_mesh", positions=positions, indices=indices)


def _cloth_setup() -> tuple[PhysicsHost, ClothHost, Node]:
    mesh = _grid_mesh()
    node = Node(name="cape_holder")
    node.add_component(MeshRefComponent(mesh_indices=(0,)))
    root = Node(name="root")
    root.add_child(node)
    scene = Scene(root=root)
    imported = ImportedScene(meshes=(mesh,), textures=(), skins=(), scene=scene)
    physics = PhysicsHost()
    cloth = ClothHost()
    cloth.register_imported_scene(imported)
    return physics, cloth, node


def test_add_cloth_returns_handle() -> None:
    physics, cloth, node = _cloth_setup()
    api = PhysicsLite(physics, cloth_host=cloth)
    handle = api.add_cloth(node, cloth_name="cape", anchor_fraction=0.30)
    assert isinstance(handle, ClothPieceHandle)
    assert handle.name == "cape"


def test_add_cloth_requires_node_type() -> None:
    physics, cloth, _node = _cloth_setup()
    api = PhysicsLite(physics, cloth_host=cloth)
    with pytest.raises(TypeError, match="add_cloth expected a Node"):
        api.add_cloth("not_a_node")


def test_add_cloth_without_cloth_host_raises() -> None:
    physics, _cloth, node = _cloth_setup()
    api = PhysicsLite(physics, cloth_host=None)
    with pytest.raises(RuntimeError, match="add_cloth"):
        api.add_cloth(node)


def test_cloth_handle_setters_apply() -> None:
    physics, cloth, node = _cloth_setup()
    api = PhysicsLite(physics, cloth_host=cloth)
    handle = api.add_cloth(node, cloth_name="cape", anchor_fraction=0.30)
    assert handle is not None
    handle.structural_stiffness = 0.5
    handle.bend_stiffness = 0.05
    handle.linear_damping = 0.95
    handle.set_iterations(20)
    handle.enabled = False
    piece = cloth.find_piece("cape")
    assert piece is not None
    assert piece.params.structural_stiffness == pytest.approx(0.5)
    assert piece.params.bend_stiffness == pytest.approx(0.05)
    assert piece.params.linear_damping == pytest.approx(0.95)
    assert piece.params.iterations == 20
    assert piece.enabled is False


def test_add_sphere_collider_appends_to_solver() -> None:
    physics, cloth, _node = _cloth_setup()
    api = PhysicsLite(physics, cloth_host=cloth)
    api.add_sphere_collider(center=(0.0, -0.1, 0.0), radius=0.05)
    assert len(cloth.solver.colliders) == 1


def test_add_capsule_collider_appends_to_solver() -> None:
    physics, cloth, _node = _cloth_setup()
    api = PhysicsLite(physics, cloth_host=cloth)
    api.add_capsule_collider(a=(0.0, 0.0, 0.0), b=(0.0, -0.2, 0.0), radius=0.04)
    assert len(cloth.solver.colliders) == 1


def test_set_cloth_gravity_replaces_existing() -> None:
    physics, cloth, _node = _cloth_setup()
    cloth.install_default_forces()
    api = PhysicsLite(physics, cloth_host=cloth)
    api.set_cloth_gravity((0.0, -3.0, 0.0))
    gravity_forces = [f for f in cloth.solver.forces if isinstance(f, ClothGravity)]
    assert len(gravity_forces) == 1
    np.testing.assert_allclose(gravity_forces[0].acceleration, vec3(0.0, -3.0, 0.0))


def test_add_cloth_wind_returns_handle() -> None:
    physics, cloth, _node = _cloth_setup()
    api = PhysicsLite(physics, cloth_host=cloth)
    wind = api.add_cloth_wind(direction=(1.0, 0.0, 0.0), speed=2.0)
    assert isinstance(wind, ClothWind)
    assert wind.speed == 2.0


def test_get_cloth_returns_handle() -> None:
    physics, cloth, node = _cloth_setup()
    api = PhysicsLite(physics, cloth_host=cloth)
    api.add_cloth(node, cloth_name="cape", anchor_fraction=0.30)
    handle = api.get_cloth("cape")
    assert isinstance(handle, ClothPieceHandle)
    assert api.get_cloth("missing") is None


def test_cloth_names_lists_pieces() -> None:
    physics, cloth, node = _cloth_setup()
    api = PhysicsLite(physics, cloth_host=cloth)
    api.add_cloth(node, cloth_name="cape", anchor_fraction=0.30)
    assert "cape" in api.cloth_names()


def test_cloth_handle_reset_restores_rest() -> None:
    physics, cloth, node = _cloth_setup()
    api = PhysicsLite(physics, cloth_host=cloth)
    handle = api.add_cloth(node, cloth_name="cape", anchor_fraction=0.30)
    cloth.install_default_forces()
    for _ in range(60):
        cloth.tick(1.0 / 60.0)
    piece = cloth.find_piece("cape")
    assert not np.allclose(piece.positions, piece.rest_positions)
    handle.reset()
    np.testing.assert_allclose(piece.positions, piece.rest_positions, atol=1.0e-6)
