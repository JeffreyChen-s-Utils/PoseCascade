"""Tests for :class:`posecascade.animation.physics_host.PhysicsHost`."""
from __future__ import annotations

import numpy as np

from posecascade.animation.physics_host import PhysicsHost
from posecascade.animation.spring import Wind
from posecascade.assets.types import ImportedScene
from posecascade.scene.component import SpringChainComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.scene.transform import Transform
from posecascade.utils.math3d import vec3


def _scene_with_one_chain() -> tuple[Scene, Node]:
    anchor = Node(name="head_anchor")
    parent = anchor
    joints = []
    for i in range(3):
        joint = Node(name=f"hair_C_{i}", transform=Transform(translation=vec3(0.1, 0.0, 0.0)))
        parent.add_child(joint)
        joints.append(joint)
        parent = joint
    anchor.add_component(
        SpringChainComponent(
            chain_name="hair_C",
            joints=tuple(joints),
            stiffness=10.0,
            damping=2.0,
            inertia=0.05,
        )
    )
    root = Node(name="root")
    root.add_child(anchor)
    return Scene(root=root), anchor


def test_register_scene_creates_chain() -> None:
    scene, _anchor = _scene_with_one_chain()
    host = PhysicsHost()
    host.register_scene(scene)
    assert len(host.chains()) == 1
    assert host.find_chain("hair_C") is not None


def test_register_scene_is_idempotent() -> None:
    scene, _anchor = _scene_with_one_chain()
    host = PhysicsHost()
    host.register_scene(scene)
    host.register_scene(scene)
    assert len(host.chains()) == 1


def test_tick_advances_simulator_with_force() -> None:
    scene, _anchor = _scene_with_one_chain()
    host = PhysicsHost()
    host.register_scene(scene)
    host.add_force(Wind(direction=vec3(0.0, -1.0, 0.0), speed=2.0))

    chain = host.find_chain("hair_C")
    assert chain is not None
    initial = chain.joints[-1].angular_velocity.copy()

    for _ in range(60):
        host.tick(1.0 / 60.0)

    final = chain.joints[-1].angular_velocity
    # Joint experienced force ⇒ angular velocity diverged from zero (or settled
    # to non-rest pose). Verify the joint actually moved by checking world rotation
    # changed from rest.
    rotation = chain.joints[-1].node.transform.rotation
    rest = chain.joints[-1].rest_local_rotation
    assert not np.allclose(rotation, rest, atol=1.0e-3) or not np.allclose(
        final, initial, atol=1.0e-6
    )


def test_install_default_forces_only_runs_once() -> None:
    host = PhysicsHost()
    host.install_default_forces()
    host.install_default_forces()
    host.install_default_forces()
    assert len(host.simulator.global_forces) == 1


def test_register_imported_scene_walks_skin_joints_outside_scene() -> None:
    """Some glTF authors put bones in a separate group — register_imported_scene must
    still rig chains anchored on those joints by walking ``imported.skins``."""
    # Build a chain anchored on a node that is NOT under scene.root.
    floating_anchor = Node(name="floating_head_anchor")
    parent = floating_anchor
    joints = []
    for i in range(2):
        joint = Node(name=f"orn_{i}", transform=Transform(translation=vec3(0.05, 0.0, 0.0)))
        parent.add_child(joint)
        joints.append(joint)
        parent = joint
    floating_anchor.add_component(
        SpringChainComponent(chain_name="orn", joints=tuple(joints), stiffness=20.0)
    )
    # An empty active scene + a skin holding the floating chain.
    empty_scene = Scene(root=Node(name="root"))
    fake_skin_joints = (floating_anchor, *joints)

    class _MiniSkin:
        joints = fake_skin_joints

    imported = ImportedScene(
        meshes=(),
        textures=(),
        skins=(_MiniSkin(),),  # type: ignore[arg-type]
        scene=empty_scene,
    )
    host = PhysicsHost()
    host.register_imported_scene(imported)
    assert host.find_chain("orn") is not None


def test_reset_clears_chains() -> None:
    scene, _anchor = _scene_with_one_chain()
    host = PhysicsHost()
    host.register_scene(scene)
    host.add_force(Wind(direction=vec3(1.0, 0.0, 0.0), speed=1.0))

    host.reset()

    assert host.chains() == ()
    assert host.simulator.global_forces == []


def test_tick_with_no_chains_is_noop() -> None:
    host = PhysicsHost()
    # Should not raise even with no chains registered.
    host.tick(1.0 / 60.0)
    assert host.chains() == ()


def test_remove_chains_for_subtree_drops_matching_chains() -> None:
    """Deleting a subtree should remove chains anchored on / under it."""
    scene_a, anchor_a = _scene_with_one_chain()
    host = PhysicsHost()
    host.register_scene(scene_a)
    assert len(host.chains()) == 1

    removed = host.remove_chains_for_subtree(anchor_a)

    assert removed == 1
    assert host.chains() == ()


def test_remove_chains_for_subtree_keeps_unrelated_chains() -> None:
    """Deleting one subtree must not affect chains anchored elsewhere."""
    scene_a, anchor_a = _scene_with_one_chain()
    # Add a second chain rooted on a separate anchor that's NOT under anchor_a.
    other_anchor = Node(name="other_anchor")
    parent = other_anchor
    other_joints = []
    for i in range(3):
        joint = Node(name=f"hair_X_{i}", transform=Transform(translation=vec3(0.1, 0.0, 0.0)))
        parent.add_child(joint)
        other_joints.append(joint)
        parent = joint
    other_anchor.add_component(
        SpringChainComponent(chain_name="hair_X", joints=tuple(other_joints))
    )
    scene_a.root.add_child(other_anchor)
    host = PhysicsHost()
    host.register_scene(scene_a)
    assert len(host.chains()) == 2

    removed = host.remove_chains_for_subtree(anchor_a)

    assert removed == 1
    remaining = host.chains()
    assert len(remaining) == 1
    assert remaining[0].name == "hair_X"


def test_remove_chains_for_subtree_removes_anchor_key() -> None:
    """After removal the anchor should be re-registerable (registered_anchors cleared)."""
    scene_a, anchor_a = _scene_with_one_chain()
    host = PhysicsHost()
    host.register_scene(scene_a)
    host.remove_chains_for_subtree(anchor_a)
    # Re-register: should add chain back since the anchor key was discarded.
    host.register_scene(scene_a)
    assert len(host.chains()) == 1


def test_chain_with_empty_joints_is_skipped() -> None:
    anchor = Node(name="head_anchor")
    anchor.add_component(SpringChainComponent(chain_name="empty", joints=()))
    scene = Scene(root=anchor)
    host = PhysicsHost()
    host.register_scene(scene)
    assert host.find_chain("empty") is None
