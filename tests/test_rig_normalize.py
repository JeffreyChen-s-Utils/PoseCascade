"""Tests for ``posecascade.animation.rig_normalize.collapse_wrapper_empties``.

Covers the typical nested-EMPTY chain that Sketchfab/VRoid/FBX-from-Blender
imports leave on a character (Sketchfab_model → fbx-id → RootNode →
Armature → Body...) and the few edge cases that gate the collapse:
joints stay put, multi-child nodes stay put, mesh nodes stay put.
"""
from __future__ import annotations

import numpy as np

from posecascade.animation.rig_normalize import (
    collapse_wrapper_empties,
    normalize_character_orientation,
)
from posecascade.assets.types import Skin
from posecascade.scene.component import MeshRefComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.scene.transform import Transform
from posecascade.utils.math3d import (
    quat_from_axis_angle,
    quat_identity,
    vec3,
)

_AXIS_X = vec3(1.0, 0.0, 0.0)


def _make_node(name: str, *, rotation=None, translation=None) -> Node:
    transform = Transform()
    if rotation is not None:
        transform.set_rotation(rotation)
    if translation is not None:
        transform.set_translation(translation)
    return Node(name=name, transform=transform)


def _world_transform(node: Node) -> np.ndarray:
    """Walk parents and compose 4×4 matrices, root → leaf."""
    matrix = node.transform.to_matrix()
    parent = node.parent
    while parent is not None:
        matrix = parent.transform.to_matrix() @ matrix
        parent = parent.parent
    return matrix


def _make_empty_skin() -> tuple[Skin, ...]:
    return ()


def test_top_level_wrapper_preserved() -> None:
    """A direct child of scene.root is NOT folded — scripts target it as
    the character root via ``scene.find(...)``."""
    scene = Scene(name="test")
    wrapper = _make_node("wrap", rotation=quat_from_axis_angle(_AXIS_X, -np.pi / 2))
    leaf = _make_node("leaf", translation=vec3(0.0, 0.5, 0.0))
    leaf.add_component(MeshRefComponent(mesh_indices=()))
    wrapper.add_child(leaf)
    scene.root.add_child(wrapper)

    removed = collapse_wrapper_empties(scene, _make_empty_skin())
    assert removed == 0
    assert wrapper in scene.root.children
    assert leaf in wrapper.children


def test_nested_wrappers_collapse_below_top() -> None:
    """A 4-deep wrapper chain (mirrors Sketchfab_model→fbx→RootNode→Armature):
    ``Sketchfab_model`` (top-level) stays as the character handle, the 3
    intermediate wrappers fold into the leaf. Leaf world transform is
    preserved exactly."""
    scene = Scene(name="test")
    a = _make_node("Sketchfab_model", rotation=quat_from_axis_angle(_AXIS_X, -np.pi / 2))
    b = _make_node("fbx-id", rotation=quat_from_axis_angle(_AXIS_X, +np.pi / 2))
    c = _make_node("RootNode")
    d = _make_node("Armature", rotation=quat_from_axis_angle(_AXIS_X, -np.pi / 2))
    leaf = _make_node("BodyMesh", translation=vec3(0.0, 1.0, 0.0))
    leaf.add_component(MeshRefComponent(mesh_indices=()))
    a.add_child(b)
    b.add_child(c)
    c.add_child(d)
    d.add_child(leaf)
    scene.root.add_child(a)
    leaf_world_before = _world_transform(leaf)

    removed = collapse_wrapper_empties(scene, _make_empty_skin())
    assert removed == 3
    # Sketchfab_model stays at scene root.
    assert a in scene.root.children
    # Leaf should now be a direct child of Sketchfab_model.
    assert leaf in a.children
    assert leaf.parent is a
    np.testing.assert_allclose(_world_transform(leaf), leaf_world_before, atol=1e-5)


def test_joint_does_not_collapse() -> None:
    """A node listed as a skin joint is NOT folded even if it has one
    child and no mesh — bones must keep their hierarchy intact."""
    scene = Scene(name="test")
    joint = _make_node("hip", rotation=quat_from_axis_angle(_AXIS_X, np.pi / 4))
    child = _make_node("upper_leg")
    child.add_component(MeshRefComponent(mesh_indices=()))
    joint.add_child(child)
    scene.root.add_child(joint)
    skin = Skin(
        name="armature",
        joints=(joint,),
        inverse_bind_matrices=np.eye(4, dtype=np.float32).reshape(1, 4, 4),
    )

    removed = collapse_wrapper_empties(scene, (skin,))
    assert removed == 0
    assert joint in scene.root.children


def test_multi_child_does_not_collapse() -> None:
    """A node with two children is not a wrapper — folding it would have
    to choose one child to inherit the transform."""
    scene = Scene(name="test")
    parent = _make_node("group", rotation=quat_from_axis_angle(_AXIS_X, np.pi / 6))
    leaf_a = _make_node("a")
    leaf_b = _make_node("b")
    leaf_a.add_component(MeshRefComponent(mesh_indices=()))
    leaf_b.add_component(MeshRefComponent(mesh_indices=()))
    parent.add_child(leaf_a)
    parent.add_child(leaf_b)
    scene.root.add_child(parent)

    removed = collapse_wrapper_empties(scene, _make_empty_skin())
    assert removed == 0
    assert parent in scene.root.children


def test_wrapper_translation_composes_with_child_rotation() -> None:
    """A nested wrapper with rotation + translation correctly folds, even
    when the child has its own rotation that would otherwise rotate the
    wrapper's translation in unexpected ways. (Top-level handle stays.)"""
    scene = Scene(name="test")
    top = _make_node("character_root")
    wrap_rot = quat_from_axis_angle(vec3(0.0, 0.0, 1.0), np.pi / 2)
    wrapper = _make_node("wrap", rotation=wrap_rot, translation=vec3(0.0, 0.1, 0.0))
    child_rot = quat_from_axis_angle(_AXIS_X, np.pi / 3)
    leaf = _make_node("leaf", rotation=child_rot, translation=vec3(0.5, 0.0, 0.0))
    leaf.add_component(MeshRefComponent(mesh_indices=()))
    wrapper.add_child(leaf)
    top.add_child(wrapper)
    scene.root.add_child(top)
    leaf_world_before = _world_transform(leaf)

    collapse_wrapper_empties(scene, _make_empty_skin())
    np.testing.assert_allclose(_world_transform(leaf), leaf_world_before, atol=1e-5)


def test_no_wrappers_no_op() -> None:
    """A scene of one joint-bearing node with no wrappers leaves the tree
    unchanged."""
    scene = Scene(name="test")
    leaf = _make_node("standalone")
    leaf.add_component(MeshRefComponent(mesh_indices=()))
    scene.root.add_child(leaf)

    removed = collapse_wrapper_empties(scene, _make_empty_skin())
    assert removed == 0
    assert leaf in scene.root.children


# --- normalize_character_orientation ---------------------------------------
def _make_skin(joints, name="skin") -> Skin:
    n = len(joints)
    return Skin(
        name=name,
        joints=tuple(joints),
        inverse_bind_matrices=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
    )


def test_normalize_no_op_for_upright_rig() -> None:
    """A character with head above feet at world Y already gets no rotation."""
    scene = Scene(name="test")
    top = _make_node("character_root")
    head = _make_node("head", translation=vec3(0.0, 0.18, 0.0))
    foot_l = _make_node("foot_L", translation=vec3(0.05, 0.0, 0.0))
    foot_r = _make_node("foot_R", translation=vec3(-0.05, 0.0, 0.0))
    top.add_child(head)
    top.add_child(foot_l)
    top.add_child(foot_r)
    scene.root.add_child(top)
    skin = _make_skin([head, foot_l, foot_r])

    count = normalize_character_orientation(scene, (skin,), lift_to_floor=False)
    assert count == 0
    np.testing.assert_allclose(top.transform.rotation, quat_identity(), atol=1e-4)


def test_normalize_lying_character_flips_upright() -> None:
    """A character laid out along +Z (head at high Z, feet near origin) gets
    rotated so head ends up at high Y."""
    scene = Scene(name="test")
    top = _make_node("character_root")
    # Head along +Z, feet near origin — character is lying flat.
    head = _make_node("head", translation=vec3(0.0, 0.0, 0.20))
    foot_l = _make_node("foot_L", translation=vec3(0.05, 0.0, 0.02))
    foot_r = _make_node("foot_R", translation=vec3(-0.05, 0.0, 0.02))
    top.add_child(head)
    top.add_child(foot_l)
    top.add_child(foot_r)
    scene.root.add_child(top)
    skin = _make_skin([head, foot_l, foot_r])

    count = normalize_character_orientation(scene, (skin,), lift_to_floor=False)
    assert count == 1
    # After correction, head's world Y should be greater than feet's average Y.
    head_pos = _world_transform(head)[:3, 3]
    foot_l_pos = _world_transform(foot_l)[:3, 3]
    foot_r_pos = _world_transform(foot_r)[:3, 3]
    feet_avg_y = (foot_l_pos[1] + foot_r_pos[1]) / 2
    assert head_pos[1] > feet_avg_y, (
        f"head Y {head_pos[1]:.3f} should be > feet avg Y {feet_avg_y:.3f}"
    )


def test_normalize_skips_when_landmarks_missing() -> None:
    """No 'head' joint or no 'foot' joints means we can't determine
    orientation — skip without raising."""
    scene = Scene(name="test")
    top = _make_node("character_root")
    j1 = _make_node("J_Bip_C_Hips_031")
    top.add_child(j1)
    scene.root.add_child(top)
    skin = _make_skin([j1])

    count = normalize_character_orientation(scene, (skin,))
    assert count == 0


def test_normalize_empty_skins_is_no_op() -> None:
    """An empty skins tuple is a no-op."""
    scene = Scene(name="test")
    count = normalize_character_orientation(scene, ())
    assert count == 0


# Keep imports load-bearing for IDEs.
__all__ = ["quat_identity"]
