"""Tests for :class:`posecascade.animation.cloth_host.ClothHost`."""
from __future__ import annotations

import numpy as np

from posecascade.animation.cloth import SphereCollider
from posecascade.animation.cloth_host import ClothHost
from posecascade.assets.types import ImportedScene, Mesh, Skin
from posecascade.scene.component import ClothComponent, SkinRefComponent
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
    # floor_y defaults to None now; nothing to disable.
    host.install_default_forces()
    host.register_imported_scene(imported)
    initial_positions = host.find_piece("cape").positions.copy()
    for _ in range(60):
        host.tick(1.0 / 60.0)
    final = host.find_piece("cape").positions
    # Free verts should have dropped (Y decreased) under gravity.
    free = host.find_piece("cape").inverse_masses > 0.0
    assert np.mean(final[free, 1]) < np.mean(initial_positions[free, 1])


def test_floor_clamp_holds_cloth_above_floor() -> None:
    """Setting ``floor_y=0`` keeps cloth verts from dropping through the ground."""
    scene, _node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.floor_y = 0.0  # opt in for this test
    host.install_default_forces()
    host.register_imported_scene(imported)
    for _ in range(30):
        host.tick(1.0 / 60.0)
    piece = host.find_piece("cape")
    assert float(np.min(piece.positions[:, 1])) >= -1.0e-5


def test_floor_clamp_includes_small_clearance_to_avoid_z_fight() -> None:
    """Verts pushed by the clamp land STRICTLY above floor_y (not exactly on it).

    Without a clearance the clamped verts would sit at exactly the floor's
    rendered Y, producing depth-fight that the user sees as 'dress visible
    through the floor' at grazing camera angles.
    """
    from posecascade.animation.cloth import ClothParams  # noqa: PLC0415
    scene, _node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.floor_y = 0.0
    host.register_imported_scene(imported)
    piece = host.find_piece("cape")
    piece.params.passive_skin_deform = True
    # Drag every vert below floor; reclamp should lift them ABOVE the plane.
    piece.positions[:, 1] = -0.5
    host.tick(1.0 / 60.0)
    assert float(np.min(piece.positions[:, 1])) > 0.0, (
        "clamped verts must be strictly above floor to avoid z-fight"
    )
    _ = ClothParams  # silence unused-import linter


def test_reclamp_after_collider_push_keeps_passive_verts_above_floor() -> None:
    """The collider push runs AFTER the solver substep, so it can shove a
    passive-skin vert below the floor (e.g. a foot-capsule below ground
    pushes a shoe vert with it). ``tick`` must re-clamp after that pass."""
    scene, _node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.floor_y = 0.0
    host.register_imported_scene(imported)
    piece = host.find_piece("cape")
    piece.params.passive_skin_deform = True
    # Force every vert below the floor — simulating the worst case after
    # a collider push (the solver clamp ran earlier, then the collider
    # pushed everything underground).
    piece.positions[:, 1] = -0.5
    host.tick(1.0 / 60.0)
    assert float(np.min(piece.positions[:, 1])) >= -1.0e-5, (
        f"reclamp pass left {(piece.positions[:, 1] < -1e-5).sum()} verts below floor"
    )


def test_floor_y_property_forwards_to_solver_ground_y() -> None:
    """``ClothHost.floor_y`` is the public name; ``ClothSolver.ground_y`` is
    the storage. The refactor that moved the clamp inside the solver substep
    keeps them synced via a property — verify both halves of the round-trip
    so a future regression that removes the property doesn't silently lose
    the ground clamp for every animation that relies on it."""
    host = ClothHost()
    assert host.floor_y is None
    assert host.solver.ground_y is None
    host.floor_y = 1.5
    assert host.floor_y == 1.5
    assert host.solver.ground_y == 1.5
    host.floor_y = None
    assert host.floor_y is None
    assert host.solver.ground_y is None


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


def test_anchor_follower_tracks_bone_translation() -> None:
    """A registered anchor follower drags pinned verts along with a translating bone.

    Without the follower the anchor verts would stay at their init-pose
    world positions while the bone walked away (skirt detaches from
    body). With it, every :meth:`tick` snaps anchors to the bone's
    current world frame.
    """
    scene, _node, mesh = _scene_with_cloth_node()
    bone = Node(name="hip_bone")
    scene.root.add_child(bone)
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    piece = host.find_piece("cape")
    anchor_mask = piece.inverse_masses == 0.0
    anchor_xs_before = piece.positions[anchor_mask, 0].copy()
    assert host.register_anchor_follower(piece, bone) is True
    # Translate the bone 1.5 m along +X and tick once. Anchor verts must follow.
    bone.transform.set_translation(vec3(1.5, 0.0, 0.0))
    host.tick(1.0 / 60.0)
    anchor_xs_after = piece.positions[anchor_mask, 0]
    np.testing.assert_allclose(anchor_xs_after, anchor_xs_before + 1.5, atol=1.0e-4)
    # ``prev_positions`` must match ``positions`` so the Verlet step doesn't
    # see a teleport as a velocity spike.
    np.testing.assert_allclose(
        piece.prev_positions[anchor_mask], piece.positions[anchor_mask],
    )


def test_register_anchor_follower_skips_pieces_without_anchors() -> None:
    """Cloth with no pinned verts (all free) yields no follower record."""
    scene, _node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    piece = host.find_piece("cape")
    # Free every vertex so anchor_indices comes back empty.
    piece.inverse_masses[:] = 1.0
    bone = Node(name="hip_bone")
    scene.root.add_child(bone)
    assert host.register_anchor_follower(piece, bone) is False


def test_anchor_follower_drags_free_verts_with_bone_translation() -> None:
    """Free verts shift with the bone's translation delta — without this
    the body walks away while the cloth's free band gets pulled back to
    its init world position via ``rest_pull``, stretching the skirt."""
    scene, _node, mesh = _scene_with_cloth_node()
    bone = Node(name="hip_bone")
    scene.root.add_child(bone)
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    piece = host.find_piece("cape")
    free_idx = np.flatnonzero(piece.inverse_masses > 0.0)
    rest_before = piece.rest_positions[free_idx].copy()
    pos_before = piece.positions[free_idx].copy()
    host.register_anchor_follower(piece, bone)
    # Translate the bone 2 m along -Z (typical walking direction) and tick.
    bone.transform.set_translation(vec3(0.0, 0.0, -2.0))
    host.tick(1.0 / 60.0)
    # Free verts (rest + current positions) shifted by exactly the delta.
    np.testing.assert_allclose(
        piece.rest_positions[free_idx, 2], rest_before[:, 2] - 2.0, atol=1.0e-4,
    )
    # Positions also shifted before the solver step ran on top.
    # (Allow a small delta from one gravity step.)
    assert np.all(piece.positions[free_idx, 2] < pos_before[:, 2] - 1.5)


def test_remove_pieces_for_subtree_drops_anchor_followers() -> None:
    """Followers are cleaned up alongside their cloth piece when the subtree is removed."""
    scene, node, mesh = _scene_with_cloth_node()
    bone = Node(name="hip_bone")
    scene.root.add_child(bone)
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    piece = host.find_piece("cape")
    host.register_anchor_follower(piece, bone)
    assert len(host._anchor_followers) == 1
    removed = host.remove_pieces_for_subtree(node)
    assert removed == 1
    assert host._anchor_followers == []


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


def _make_skinned_imported(
    scene: Scene, node: Node, mesh: Mesh, bone: Node,
) -> ImportedScene:
    """Attach a single-bone skin to ``node`` and return an ImportedScene."""
    num_verts = mesh.positions.shape[0]
    skinned_mesh = Mesh(
        name=mesh.name,
        positions=mesh.positions,
        indices=mesh.indices,
        joints_0=np.zeros((num_verts, 4), dtype=np.uint16),
        weights_0=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_verts, 1)),
    )
    skin = Skin(
        name="skin",
        joints=(bone,),
        inverse_bind_matrices=np.eye(4, dtype=np.float32)[None, ...],
    )
    node.add_component(SkinRefComponent(skin=skin))
    return ImportedScene(meshes=(skinned_mesh,), textures=(), skins=(skin,), scene=scene)


def test_skin_target_follower_drives_rest_positions_from_bone() -> None:
    """With a single-bone skin where every vert has weight 1.0 on the bone,
    each tick should set ``rest_positions`` to the bone-translated bind verts."""
    scene, node, mesh = _scene_with_cloth_node()
    bone = Node(name="hip_bone")
    scene.root.add_child(bone)
    imported = _make_skinned_imported(scene, node, mesh, bone)
    host = ClothHost()
    host.install_default_forces()
    host.register_imported_scene(imported)
    piece = host.find_piece("cape")
    assert host.register_skin_target_follower(piece, node) is True

    # Translate the bone +X by 0.7 m and tick. Rest positions should follow.
    bone.transform.set_translation(vec3(0.7, 0.0, 0.0))
    host.tick(1.0 / 60.0)
    expected = mesh.positions + np.array([0.7, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(piece.rest_positions, expected, atol=1.0e-4)


def test_skin_target_follower_skips_node_without_skin() -> None:
    """A cloth node without a SkinRefComponent rejects the follower registration."""
    scene, node, mesh = _scene_with_cloth_node()
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    piece = host.find_piece("cape")
    assert host.register_skin_target_follower(piece, node) is False


def test_skin_target_follower_skips_unskinned_mesh() -> None:
    """A skinned node whose mesh primitive has no joints/weights is rejected."""
    scene, node, mesh = _scene_with_cloth_node()
    bone = Node(name="hip_bone")
    scene.root.add_child(bone)
    # Skin attached to the node but the mesh primitive carries no joints_0/weights_0.
    skin = Skin(
        name="skin",
        joints=(bone,),
        inverse_bind_matrices=np.eye(4, dtype=np.float32)[None, ...],
    )
    node.add_component(SkinRefComponent(skin=skin))
    imported = _make_imported(scene, mesh)
    host = ClothHost()
    host.register_imported_scene(imported)
    piece = host.find_piece("cape")
    assert host.register_skin_target_follower(piece, node) is False


def test_remove_pieces_for_subtree_drops_skin_followers() -> None:
    """Skin-target followers are cleaned up alongside their cloth piece."""
    scene, node, mesh = _scene_with_cloth_node()
    bone = Node(name="hip_bone")
    scene.root.add_child(bone)
    imported = _make_skinned_imported(scene, node, mesh, bone)
    host = ClothHost()
    host.register_imported_scene(imported)
    piece = host.find_piece("cape")
    host.register_skin_target_follower(piece, node)
    assert host._skin_followers  # noqa: SLF001

    removed = host.remove_pieces_for_subtree(node)

    assert removed == 1
    assert host._skin_followers == []  # noqa: SLF001


def test_reset_clears_skin_followers() -> None:
    """``reset()`` drops skin-target followers along with the rest of host state."""
    scene, node, mesh = _scene_with_cloth_node()
    bone = Node(name="hip_bone")
    scene.root.add_child(bone)
    imported = _make_skinned_imported(scene, node, mesh, bone)
    host = ClothHost()
    host.register_imported_scene(imported)
    piece = host.find_piece("cape")
    host.register_skin_target_follower(piece, node)
    assert host._skin_followers  # noqa: SLF001

    host.reset()

    assert host._skin_followers == []  # noqa: SLF001
