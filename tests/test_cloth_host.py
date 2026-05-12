"""Tests for :class:`posecascade.animation.cloth_host.ClothHost`."""
from __future__ import annotations

import numpy as np

from posecascade.animation.cloth import SphereCollider
from posecascade.animation.cloth_host import ClothHost
from posecascade.assets.types import ImportedScene, Mesh
from posecascade.scene.component import ClothComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.utils.math3d import vec3


def _grid_mesh(name: str, rows: int = 3, cols: int = 3, spacing: float = 0.1) -> Mesh:
    positions = np.zeros((rows * cols, 3), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            positions[r * cols + c] = [c * spacing, -r * spacing, 0.0]
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
    return Mesh(name=name, positions=positions, indices=indices)


def _scene_with_cloth_node(mesh_index: int = 0) -> tuple[Scene, Node, Mesh]:
    mesh = _grid_mesh("cape")
    node = Node(name="cape_holder")
    node.add_component(
        ClothComponent(
            cloth_name="cape",
            mesh_index=mesh_index,
            anchor_axis=1,
            anchor_fraction=0.30,
        )
    )
    root = Node(name="root")
    root.add_child(node)
    return Scene(root=root), node, mesh


def _make_imported(scene: Scene, mesh: Mesh) -> ImportedScene:
    return ImportedScene(meshes=(mesh,), textures=(), skins=(), scene=scene)


def test_register_imported_scene_creates_binding() -> None:
    scene, _node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    assert len(host.bindings()) == 1
    assert host.find_piece("cape") is not None


def test_register_is_idempotent() -> None:
    scene, _node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    host.register_imported_scene(imported)
    assert len(host.bindings()) == 1


def test_register_skips_invalid_mesh_index() -> None:
    scene, _node, mesh = _scene_with_cloth_node(mesh_index=42)  # out of range
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    assert host.bindings() == ()


def test_install_default_forces_only_runs_once() -> None:
    host = ClothHost()
    host.install_default_forces()
    host.install_default_forces()
    assert len(host.solver.forces) == 1


def test_tick_advances_cloth_under_gravity() -> None:
    scene, _node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.install_default_forces()
    host.register_imported_scene(imported)
    initial_positions = host.find_piece("cape").positions.copy()
    for _ in range(60):
        host.tick(1.0 / 60.0)
    final = host.find_piece("cape").positions
    # Free verts should have dropped (Y decreased) under gravity.
    free = host.find_piece("cape").inverse_masses > 0.0
    assert np.mean(final[free, 1]) < np.mean(initial_positions[free, 1])


def test_iter_local_state_yields_positions_and_normals() -> None:
    scene, _node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    bindings_seen = list(host.iter_local_state())
    assert len(bindings_seen) == 1
    binding, positions_local, normals_local = bindings_seen[0]
    assert positions_local.shape == (mesh.positions.shape[0], 3)
    assert normals_local.shape == (mesh.positions.shape[0], 3)
    # Identity world matrix ⇒ local positions ≈ world positions ≈ rest positions.
    np.testing.assert_allclose(positions_local, mesh.positions, atol=1.0e-5)


def test_local_positions_round_trip_through_world_matrix() -> None:
    """Translate the cloth node; iter_local_state should still yield original local coords."""
    scene, node, mesh = _scene_with_cloth_node()
    node.transform.set_translation(vec3(5.0, 0.0, 0.0))
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    binding, positions_local, _normals = next(iter(host.iter_local_state()))
    np.testing.assert_allclose(positions_local, mesh.positions, atol=1.0e-4)


def test_local_state_tracks_parent_translation_applied_after_register() -> None:
    """Parent translated AFTER cloth registration — iter_local_state must compensate.

    The renderer reads the cloth node's CURRENT world matrix every frame.
    If iter_local_state kept using the registration-time world_to_local, the
    parent translation would double-apply: once through the stale local
    positions, once through the new model matrix. This is what made the
    bundled dance skirt drift when the declarative root yaw/translation
    started transforming Sketchfab_model.
    """
    scene, node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    # Apply the parent translation only AFTER registration.
    scene.root.transform.set_translation(vec3(2.0, -1.5, 0.75))
    _binding, positions_local, _normals = next(iter(host.iter_local_state()))
    # The cloth positions stayed at the original world coords (no tick called),
    # so local = inv(current_world) @ world_cloth = inv(translation) @ rest_pos
    # = rest_pos - translation.
    expected = mesh.positions - np.array([2.0, -1.5, 0.75], dtype=np.float32)
    np.testing.assert_allclose(positions_local, expected, atol=1.0e-4)


def test_local_state_tracks_parent_rotation_applied_after_register() -> None:
    """Parent rotated AFTER cloth registration — iter_local_state must compensate.

    The yaw the declarative animation applies to the character root is the
    common case: cloth positions remain in world coords (anchors track a
    bone, free verts simulate freely), but the renderer multiplies them by
    the model matrix the root now carries. Without a live-recomputed
    world_to_local the skirt visibly counter-rotates against the body.
    """
    from posecascade.utils.math3d import quat_from_axis_angle  # noqa: PLC0415

    scene, node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    # 30deg yaw around +Y on the root, after registration.
    yaw = float(np.pi / 6.0)
    scene.root.transform.set_rotation(quat_from_axis_angle(vec3(0.0, 1.0, 0.0), yaw))
    _binding, positions_local, _normals = next(iter(host.iter_local_state()))
    # Construct what the renderer will do: world_rendered = model_matrix @ local.
    # Since cloth positions are still at the rest (mesh) coords in world space,
    # we want world_rendered == rest_positions.
    from posecascade.animation.cloth_host import _world_matrix  # noqa: PLC0415

    model = _world_matrix(node)
    homog = np.column_stack(
        [positions_local, np.ones((positions_local.shape[0], 1), dtype=np.float32)],
    )
    rendered = (homog @ model.T)[:, :3]
    np.testing.assert_allclose(rendered, mesh.positions, atol=1.0e-4)


def test_add_collider_projects_verts() -> None:
    scene, _node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.install_default_forces()
    host.register_imported_scene(imported)
    sphere = SphereCollider(center=vec3(0.1, -0.1, 0.0), radius=0.08)
    host.add_collider(sphere)
    for _ in range(60):
        host.tick(1.0 / 60.0)
    piece = host.find_piece("cape")
    movable = piece.inverse_masses > 0.0
    deltas = piece.positions[movable] - sphere.center
    dists = np.linalg.norm(deltas, axis=1)
    assert np.all(dists >= sphere.radius - 1.0e-3)


def test_add_piece_works_for_external_construction() -> None:
    """A user script can build a ClothPiece itself and register it."""
    from posecascade.animation.cloth import anchor_by_top_axis, cloth_from_mesh  # noqa: PLC0415
    from posecascade.utils.math3d import mat4_identity  # noqa: PLC0415

    scene, _node, mesh = _scene_with_cloth_node()
    # Don't register via component — build piece directly.
    node = scene.root.children[0]
    node.components.clear()  # remove the auto-component
    piece = cloth_from_mesh(
        "manual",
        mesh.positions,
        mesh.indices,
        world_matrix=mat4_identity(),
        anchor_mask=anchor_by_top_axis(mesh.positions, fraction=0.3),
    )
    host = ClothHost()
    host.add_piece(node, piece, mesh_index=0)
    assert host.find_piece("manual") is piece
    assert len(host.bindings()) == 1


def test_reset_clears_state() -> None:
    scene, _node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.install_default_forces()
    host.register_imported_scene(imported)
    host.reset()
    assert host.bindings() == ()
    assert host.solver.pieces == []


def test_disabled_piece_skipped_in_local_state() -> None:
    scene, _node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    host.find_piece("cape").enabled = False
    assert list(host.iter_local_state()) == []


def test_remove_pieces_for_subtree_drops_matching_cloth() -> None:
    scene, node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    assert len(host.bindings()) == 1
    assert len(host.solver.pieces) == 1

    removed = host.remove_pieces_for_subtree(node)

    assert removed == 1
    assert host.bindings() == ()
    assert host.solver.pieces == []


def test_remove_pieces_for_subtree_keeps_unrelated() -> None:
    """Removing one subtree must not affect cloth on a sibling node."""
    # Two cloth holders side by side
    mesh_a = _grid_mesh("cape_a")
    mesh_b = _grid_mesh("cape_b")
    node_a = Node(name="holder_a")
    node_a.add_component(ClothComponent(cloth_name="a", mesh_index=0, anchor_fraction=0.30))
    node_b = Node(name="holder_b")
    node_b.add_component(ClothComponent(cloth_name="b", mesh_index=1, anchor_fraction=0.30))
    root = Node(name="root")
    root.add_child(node_a)
    root.add_child(node_b)
    scene = Scene(root=root)
    imported = ImportedScene(meshes=(mesh_a, mesh_b), textures=(), skins=(), scene=scene)

    host = ClothHost()
    host.register_imported_scene(imported)
    assert len(host.bindings()) == 2

    removed = host.remove_pieces_for_subtree(node_a)

    assert removed == 1
    remaining = host.bindings()
    assert len(remaining) == 1
    assert remaining[0].piece.name == "b"


def test_singular_world_matrix_rejected() -> None:
    """A node with zero-scale parent has a singular world matrix — must not crash."""
    scene, node, mesh = _scene_with_cloth_node()
    node.transform.set_scale(vec3(0.0, 0.0, 0.0))
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    # Singular matrix ⇒ binding skipped with a warning, no crash.
    assert host.bindings() == ()
