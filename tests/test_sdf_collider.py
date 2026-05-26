"""Unit tests for the SDF (signed distance field) collider — Magica
Cloth-style hair-vs-body collision using a baked voxel grid."""
from __future__ import annotations

import numpy as np
import pytest

from posecascade.animation.cloth import SDFCollider, bake_sdf_from_triangles
from posecascade.animation.spring import _project_sdf


def _unit_cube_triangles() -> np.ndarray:
    """Return the 12 triangles of a unit cube centred at the origin."""
    verts = np.array([
        [-0.5, -0.5, -0.5],
        [+0.5, -0.5, -0.5],
        [+0.5, +0.5, -0.5],
        [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5],
        [+0.5, -0.5, +0.5],
        [+0.5, +0.5, +0.5],
        [-0.5, +0.5, +0.5],
    ], dtype=np.float32)
    # 12 triangles (2 per face, 6 faces). Winding: counter-clockwise from outside.
    faces = np.array([
        # -Z face
        [0, 2, 1], [0, 3, 2],
        # +Z face
        [4, 5, 6], [4, 6, 7],
        # -Y face
        [0, 1, 5], [0, 5, 4],
        # +Y face
        [3, 7, 6], [3, 6, 2],
        # -X face
        [0, 4, 7], [0, 7, 3],
        # +X face
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.uint32)
    return verts[faces]


def test_bake_sdf_produces_valid_grid() -> None:
    triangles = _unit_cube_triangles()
    sdf = bake_sdf_from_triangles(triangles, voxel_size=0.1, padding=0.1)
    assert isinstance(sdf, SDFCollider)
    assert sdf.grid.ndim == 3
    assert sdf.grid.dtype == np.float32
    # Origin should be at AABB min - padding = (-0.6, -0.6, -0.6).
    assert sdf.grid_origin[0] == pytest.approx(-0.6, abs=1e-5)
    assert sdf.grid_origin[1] == pytest.approx(-0.6, abs=1e-5)
    assert sdf.grid_origin[2] == pytest.approx(-0.6, abs=1e-5)


def test_sdf_signs_inside_vs_outside_unit_cube() -> None:
    triangles = _unit_cube_triangles()
    sdf = bake_sdf_from_triangles(triangles, voxel_size=0.1, padding=0.2)
    # Voxel centre nearest to origin should be NEGATIVE (inside the
    # unit cube). Sample directly into the grid.
    cx = int((0.0 - sdf.grid_origin[0]) / sdf.voxel_size)
    cy = int((0.0 - sdf.grid_origin[1]) / sdf.voxel_size)
    cz = int((0.0 - sdf.grid_origin[2]) / sdf.voxel_size)
    assert sdf.grid[cx, cy, cz] < 0.0, (
        f"Expected origin inside cube to be negative, got {sdf.grid[cx, cy, cz]}"
    )
    # Voxel centre at (0.65, 0, 0) — just outside the cube (cube max is
    # 0.5, padding extends grid to 0.7) — should be POSITIVE.
    cx_out = int((0.65 - sdf.grid_origin[0]) / sdf.voxel_size)
    cy_out = int((0.0 - sdf.grid_origin[1]) / sdf.voxel_size)
    cz_out = int((0.0 - sdf.grid_origin[2]) / sdf.voxel_size)
    assert sdf.grid[cx_out, cy_out, cz_out] > 0.0


def test_project_sdf_pushes_point_inside_cube_to_surface() -> None:
    triangles = _unit_cube_triangles()
    sdf = bake_sdf_from_triangles(
        triangles, voxel_size=0.05, padding=0.1, skin_offset=0.05,
    )
    # Point clearly inside the cube — should be pushed out.
    point = np.array([0.0, 0.0, 0.4], dtype=np.float32)
    pushed, hit = _project_sdf(point, sdf)
    assert hit, "Expected point inside cube to register a hit"
    # Pushed point should be FURTHER from the origin than where it was.
    assert np.linalg.norm(pushed) > np.linalg.norm(point)


def test_project_sdf_leaves_far_point_alone() -> None:
    triangles = _unit_cube_triangles()
    sdf = bake_sdf_from_triangles(
        triangles, voxel_size=0.05, padding=0.1, skin_offset=0.02,
    )
    # Point far outside the cube — should not push.
    point = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    pushed, hit = _project_sdf(point, sdf)
    assert not hit
    assert np.allclose(pushed, point)
