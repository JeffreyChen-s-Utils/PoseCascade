"""Tests for :mod:`posecascade.scene.anime_bedroom`."""
from __future__ import annotations

import numpy as np

from posecascade.scene.anime_bedroom import make_anime_bedroom_scene
from posecascade.scene.component import MeshRefComponent


def test_returns_imported_scene_with_pieces() -> None:
    bundle = make_anime_bedroom_scene()
    assert bundle.scene is not None
    pieces = bundle.scene.root.children
    names = {n.name for n in pieces}
    # Must contain the core room and a few obvious furniture pieces.
    assert {"floor", "ceiling", "wall_back", "wall_left", "wall_right"} <= names
    assert {"bed_frame", "bed_mattress", "bed_pillow", "bed_blanket"} <= names
    assert {"desk_top", "monitor_screen"} <= names
    assert "window" in names
    # Each piece references exactly one mesh via MeshRefComponent.
    for piece in pieces:
        components = [c for c in piece.components if isinstance(c, MeshRefComponent)]
        assert len(components) == 1, piece.name
        assert len(components[0].mesh_indices) == 1, piece.name


def test_meshes_carry_per_piece_color() -> None:
    bundle = make_anime_bedroom_scene()
    # Every mesh has a base_color tuple set; mattress and blanket must differ
    # so the renderer's per-mesh tint actually changes between pieces.
    by_name = {m.name: m for m in bundle.meshes}
    assert by_name["bed_mattress"].base_color is not None
    assert by_name["bed_blanket"].base_color is not None
    assert by_name["bed_mattress"].base_color != by_name["bed_blanket"].base_color
    assert by_name["window"].base_color != by_name["floor"].base_color


def test_each_piece_box_topology() -> None:
    bundle = make_anime_bedroom_scene()
    for mesh in bundle.meshes:
        assert mesh.positions.shape == (24, 3)
        assert mesh.indices.shape == (36,)
        assert mesh.normals is not None and mesh.normals.shape == (24, 3)
        # Normals are unit length for an axis-aligned outward-facing box.
        norms = np.linalg.norm(mesh.normals, axis=-1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1.0e-6)


def test_pieces_translated_into_room() -> None:
    bundle = make_anime_bedroom_scene(size=(6.0, 3.0, 6.0))
    by_name = {n.name: n for n in bundle.scene.root.children}
    floor_y = float(by_name["floor"].transform.translation[1])
    ceiling_y = float(by_name["ceiling"].transform.translation[1])
    assert floor_y < 0.0  # slab top sits at y=0
    assert ceiling_y > 0.0
    bed_x = float(by_name["bed_frame"].transform.translation[0])
    assert bed_x > 0.0  # bed is in the +X half of the room
