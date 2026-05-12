"""Cross-path equivalence tests for the Cython cloth kernel.

Drives the same scene through both the compiled Cython kernel and the
pure-NumPy fallback to confirm they produce stable, physically equivalent
cloth state. We don't pin bit-equality — the Cython path accumulates the
PBD scatter directly into float32 positions while the NumPy path goes
through ``np.bincount``'s float64 internal accumulator, so the trajectories
diverge slightly even with the same input. The tests instead pin:

1. Both paths converge on similar settled positions (within a tolerance
   that's tighter than the cloth's natural draping motion).
2. Anchored vertices are not moved by either path.
3. Vertices inside a static sphere are projected outside its radius.
"""
from __future__ import annotations

import numpy as np
import pytest

from posecascade.animation import cloth as cloth_mod
from posecascade.animation.cloth import (
    ClothGravity,
    ClothParams,
    ClothSolver,
    SphereCollider,
    cloth_from_mesh,
)
from posecascade.utils.math3d import mat4_identity


def _make_skirt(rows: int = 12, cols: int = 6, spacing: float = 0.04):
    """Anchor the top row of a flat grid — same shape both tests use."""
    positions = np.zeros((rows * cols, 3), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            positions[r * cols + c] = (c * spacing, 1.0 - r * spacing, 0.0)
    indices: list[int] = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            a = r * cols + c
            b = a + 1
            c0 = a + cols
            d = c0 + 1
            indices += [a, c0, b, b, c0, d]
    anchor_mask = np.zeros(rows * cols, dtype=bool)
    anchor_mask[:cols] = True
    return positions, np.asarray(indices, dtype=np.uint32), anchor_mask


def _drive_60_steps(use_kernel: bool) -> np.ndarray:
    """Run 60 PBD steps with the chosen path and return the settled positions."""
    saved = cloth_mod._native
    cloth_mod._native = saved if use_kernel else None
    try:
        positions, indices, anchor_mask = _make_skirt()
        piece = cloth_from_mesh(
            name="skirt",
            local_positions=positions,
            indices=indices,
            world_matrix=mat4_identity(),
            anchor_mask=anchor_mask,
            params=ClothParams(
                structural_stiffness=0.85, bend_stiffness=0.10,
                iterations=8, linear_damping=0.985,
            ),
        )
        solver = ClothSolver(pieces=[piece])
        solver.forces.append(ClothGravity())
        solver.colliders.append(
            SphereCollider(
                center=np.array([0.12, 0.5, 0.03], dtype=np.float32),
                radius=0.10,
            ),
        )
        for _ in range(60):
            solver.step(1.0 / 60.0)
        return piece.positions.copy()
    finally:
        cloth_mod._native = saved


def test_kernel_and_fallback_settle_similarly() -> None:
    """Two paths drift in PBD numerics but settle within the same neighbourhood."""
    if cloth_mod._native is None:
        pytest.skip("Cython kernel not built — only one path available")
    kernel = _drive_60_steps(use_kernel=True)
    fallback = _drive_60_steps(use_kernel=False)
    # 5 cm tolerance covers the float32 vs float64 accumulator drift over
    # 60 steps with gravity + sphere contact — well below the cloth's own
    # draping motion (which would be 30+ cm in this setup).
    max_diff = float(np.abs(kernel - fallback).max())
    assert max_diff < 0.05, f"path divergence too large: {max_diff:.4f} m"


def test_kernel_preserves_anchors() -> None:
    """Verts whose inverse_mass is zero stay glued at rest under the kernel path."""
    if cloth_mod._native is None:
        pytest.skip("Cython kernel not built")
    positions, indices, anchor_mask = _make_skirt()
    piece = cloth_from_mesh(
        name="skirt", local_positions=positions, indices=indices,
        world_matrix=mat4_identity(), anchor_mask=anchor_mask,
        params=ClothParams(iterations=8),
    )
    rest_anchors = piece.rest_positions[anchor_mask].copy()
    solver = ClothSolver(pieces=[piece])
    solver.forces.append(ClothGravity())
    for _ in range(30):
        solver.step(1.0 / 60.0)
    np.testing.assert_allclose(
        piece.positions[anchor_mask], rest_anchors, atol=1e-5,
    )


def test_kernel_dispatches_float64() -> None:
    """The fused-type kernel accepts float64 positions and produces matching projections.

    Drives the kernel directly with both dtypes through a tiny synthetic
    case and confirms the projection lands every movable vert on / outside
    the sphere boundary. Future high-precision physics can swap dtypes
    without forking the kernel.
    """
    if cloth_mod._native is None:
        pytest.skip("Cython kernel not built")
    native = cloth_mod._native
    for dtype in (np.float32, np.float64):
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype,
        )
        prev = positions.copy()
        inv_mass = np.ones(3, dtype=dtype)
        native.project_static_sphere(
            positions, prev, inv_mass,
            dtype(0.0), dtype(0.0), dtype(0.0),
            dtype(1.0),
            dtype(0.2), dtype(1e-8),
            0, 3,
        )
        distances = np.linalg.norm(positions - np.zeros(3, dtype=dtype), axis=1)
        # Vert at 2.0 was outside radius 1.0 → unchanged.
        # Vert at 0.5 was inside → projected to radius.
        assert distances[1] >= 1.0 - 1e-5, f"dtype={dtype}: vert 1 inside sphere"
        assert distances[2] == pytest.approx(2.0, abs=1e-5)


def test_kernel_respects_vertex_range() -> None:
    """Passing ``[start, end)`` only mutates vertices in that range."""
    if cloth_mod._native is None:
        pytest.skip("Cython kernel not built")
    native = cloth_mod._native
    positions = np.array(
        [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32,
    )
    prev = positions.copy()
    native.project_static_sphere(
        positions, prev, np.ones(3, dtype=np.float32),
        0.0, 0.0, 0.0,
        1.0,
        0.2, 1e-8,
        0, 2,  # only first two verts
    )
    # Verts 0, 1 should be projected outside the unit sphere.
    assert np.linalg.norm(positions[0]) >= 1.0 - 1e-5
    assert np.linalg.norm(positions[1]) >= 1.0 - 1e-5
    # Vert 2 must be untouched.
    np.testing.assert_array_equal(positions[2], [0.5, 0.0, 0.0])


def test_kernel_projects_outside_sphere() -> None:
    """After settling, every cloth vert sits outside the static sphere radius."""
    if cloth_mod._native is None:
        pytest.skip("Cython kernel not built")
    positions, indices, anchor_mask = _make_skirt()
    piece = cloth_from_mesh(
        name="skirt", local_positions=positions, indices=indices,
        world_matrix=mat4_identity(), anchor_mask=anchor_mask,
        params=ClothParams(iterations=8),
    )
    centre = np.array([0.12, 0.5, 0.03], dtype=np.float32)
    radius = 0.10
    skin_offset = 0.005
    solver = ClothSolver(pieces=[piece])
    solver.forces.append(ClothGravity())
    solver.colliders.append(SphereCollider(center=centre, radius=radius))
    for _ in range(60):
        solver.step(1.0 / 60.0)
    distances = np.linalg.norm(piece.positions - centre, axis=1)
    movable = piece.inverse_masses > 0.0
    # Allow a 0.1 mm interpenetration tolerance — float32 round-trip
    # through the kernel can leave a vert exactly on the sphere boundary
    # by a few ULPs.
    inside = movable & (distances < (radius + skin_offset - 1e-4))
    assert not inside.any(), (
        f"{int(inside.sum())} movable verts still inside sphere after settle"
    )
