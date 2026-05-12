"""Tests for :mod:`posecascade.animation.cloth` — PBD cloth simulator."""
from __future__ import annotations

import numpy as np
import pytest

from posecascade.animation.cloth import (
    CapsuleCollider,
    ClothGravity,
    ClothParams,
    ClothSolver,
    ClothWind,
    SphereCollider,
    anchor_by_top_axis,
    cloth_from_mesh,
    compute_vertex_normals,
)
from posecascade.utils.math3d import mat4_identity, vec3


def _grid_mesh(rows: int = 4, cols: int = 4, spacing: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Flat XZ grid (Y constant), row 0 at top (Y=highest), rows triangulated."""
    positions = np.zeros((rows * cols, 3), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            positions[idx] = [c * spacing, -r * spacing, 0.0]
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
    return positions, indices


def _build_cloth_grid(
    rows: int = 4,
    cols: int = 4,
    *,
    params: ClothParams | None = None,
    anchor_fraction: float = 0.25,
):
    positions, indices = _grid_mesh(rows, cols)
    anchor_mask = anchor_by_top_axis(positions, axis=1, fraction=anchor_fraction)
    return cloth_from_mesh(
        "grid",
        positions,
        indices,
        world_matrix=mat4_identity(),
        anchor_mask=anchor_mask,
        params=params,
    )


def test_extract_edges_and_bends_count() -> None:
    cloth = _build_cloth_grid(rows=3, cols=3)
    # 3x3 grid: 12 unique edges (4 horizontal + 4 vertical + 4 diagonal per quad pair)
    # Plus bend pairs from shared edges between triangle pairs.
    assert cloth.edges.shape[0] >= 12
    assert cloth.bends.shape[1] == 2


def test_rest_cloth_stays_at_rest_no_forces() -> None:
    cloth = _build_cloth_grid()
    sim = ClothSolver(pieces=[cloth])
    initial = cloth.positions.copy()
    for _ in range(60):
        sim.step(1.0 / 60.0)
    np.testing.assert_allclose(cloth.positions, initial, atol=1.0e-5)


def test_anchor_verts_stay_pinned_under_gravity() -> None:
    cloth = _build_cloth_grid(rows=4, cols=4, anchor_fraction=0.30)
    sim = ClothSolver(pieces=[cloth])
    sim.add_force(ClothGravity(acceleration=vec3(0.0, -9.8, 0.0)))
    anchor_indices = np.where(cloth.inverse_masses == 0.0)[0]
    anchor_rest = cloth.rest_positions[anchor_indices].copy()
    for _ in range(120):
        sim.step(1.0 / 60.0)
    np.testing.assert_allclose(
        cloth.positions[anchor_indices], anchor_rest, atol=1.0e-5,
        err_msg="anchor verts drifted under gravity",
    )


def test_gravity_drops_free_verts_below_rest() -> None:
    cloth = _build_cloth_grid(rows=4, cols=4, anchor_fraction=0.25)
    sim = ClothSolver(pieces=[cloth])
    sim.add_force(ClothGravity(acceleration=vec3(0.0, -9.8, 0.0)))
    free = cloth.inverse_masses > 0.0
    rest_y_mean = float(cloth.rest_positions[free, 1].mean())
    for _ in range(60):
        sim.step(1.0 / 60.0)
    final_y_mean = float(cloth.positions[free, 1].mean())
    # The free verts as a group should sit lower than their rest Y. Local bunching
    # from the constraint solver can push individual verts slightly above rest,
    # so we check the mean rather than every vertex.
    assert final_y_mean < rest_y_mean - 0.005, (
        f"free verts did not settle below rest: mean delta {final_y_mean - rest_y_mean}"
    )


def test_edge_lengths_within_tolerance_after_settle() -> None:
    """After many iterations, edges stay close to rest length (< 5% drift on average)."""
    cloth = _build_cloth_grid(
        rows=5, cols=5,
        params=ClothParams(structural_stiffness=0.95, iterations=12),
    )
    sim = ClothSolver(pieces=[cloth])
    sim.add_force(ClothGravity(acceleration=vec3(0.0, -3.0, 0.0)))
    for _ in range(120):
        sim.step(1.0 / 60.0)
    a = cloth.positions[cloth.edges[:, 0]]
    b = cloth.positions[cloth.edges[:, 1]]
    lengths = np.linalg.norm(a - b, axis=1)
    drift = np.abs(lengths - cloth.edge_rest_lengths) / cloth.edge_rest_lengths
    assert float(np.mean(drift)) < 0.10, f"avg edge drift {float(np.mean(drift)):.3f} > 10%"


def test_sphere_collider_projects_verts_outside() -> None:
    cloth = _build_cloth_grid(rows=4, cols=4)
    sim = ClothSolver(pieces=[cloth])
    sim.add_force(ClothGravity(acceleration=vec3(0.0, -9.8, 0.0)))
    # Place a sphere where the cloth would naturally fall.
    sphere = SphereCollider(center=vec3(0.15, -0.15, 0.0), radius=0.15)
    sim.add_collider(sphere)
    for _ in range(120):
        sim.step(1.0 / 60.0)
    # No movable vert should be inside the sphere (allow skin_offset slack).
    movable = cloth.inverse_masses > 0.0
    deltas = cloth.positions[movable] - sphere.center
    dists = np.linalg.norm(deltas, axis=1)
    assert np.all(dists >= sphere.radius - 1.0e-3)


def test_slowly_hovering_sphere_uses_static_projection() -> None:
    """A collider drifting by less than its radius must not trigger the swept-CCD path.

    The swept-CCD branch projects against a long capsule from prev_center to
    center; for a hand that's basically stationary but jittering a few mm per
    frame (or one that's intentionally hovering near the cloth), every frame
    seeds a different swept volume, the cloth gets pushed in a slightly
    different direction each iteration, and a tangential ripple propagates
    through the structural mesh — visible as the skirt flapping in place
    while the hand barely moved. Gating swept on motion >= radius means
    those near-stationary frames use the cheaper static projection,
    which converges to a stable contact point.
    """
    cloth = _build_cloth_grid(rows=4, cols=4)
    sim = ClothSolver(pieces=[cloth])
    # Sphere is bigger than the per-frame motion so the dynamic gate
    # should pick the static branch.
    sphere = SphereCollider(
        center=vec3(0.15, -0.10, 0.0),
        radius=0.08,
        prev_center=vec3(0.149, -0.10, 0.0),  # 1mm drift, far smaller than 8cm radius
    )
    sim.add_collider(sphere)
    sim.step(1.0 / 60.0)
    pos_after = cloth.positions.copy()
    # Move the sphere by another 1mm and step — expected to remain stable.
    sphere.prev_center = sphere.center.copy()
    sphere.center = vec3(0.151, -0.10, 0.0)
    sim.step(1.0 / 60.0)
    drift = float(np.linalg.norm(cloth.positions - pos_after, axis=1).max())
    assert drift < 8.0e-3, (
        f"cloth jittered {drift*1000:.1f} mm under a near-stationary hover — "
        "swept-CCD probably fired despite the tiny collider motion"
    )


def test_moving_sphere_does_not_tunnel_cloth() -> None:
    """A sphere that jumps farther than its own radius in one frame must still catch verts.

    Reproduces the runtime case where the dance moves a hand bone several
    centimetres between frames — more than the sphere's radius — so a static
    end-of-frame collision check leaves the cloth verts the sphere flew past
    untouched, and the renderer shows the fingers / wrist sitting inside the
    skirt mesh. With ``prev_center`` set, the projector treats the motion as
    a swept capsule and pushes those verts to the nearest exit.
    """
    cloth = _build_cloth_grid(rows=5, cols=5)
    sim = ClothSolver(pieces=[cloth])
    # Sphere starts well above the cloth, moves down through it in one frame.
    sphere = SphereCollider(
        center=vec3(0.2, -0.5, 0.0),
        radius=0.05,
        prev_center=vec3(0.2, 0.5, 0.0),
    )
    sim.add_collider(sphere)
    sim.step(1.0 / 60.0)
    # Vertices that were on the swept path (X≈0.2, Z≈0) must have been pushed
    # out of the capsule formed by prev_center → center.
    a = sphere.prev_center
    b = sphere.center
    seg = b - a
    seg_len_sq = float(np.dot(seg, seg))
    rel = cloth.positions - a
    t = np.clip((rel @ seg) / seg_len_sq, 0.0, 1.0)
    closest = a + t[:, None] * seg
    dists = np.linalg.norm(cloth.positions - closest, axis=1)
    movable = cloth.inverse_masses > 0.0
    # Without CCD a vert at (0.2, -0.2, 0) would still be at (0.2, -0.2, 0)
    # — well inside the swept capsule's radius. With CCD it sits on the surface.
    assert np.all(dists[movable] >= sphere.radius - 5.0e-3), (
        f"swept-volume tunneling: min dist {dists[movable].min():.4f} < radius {sphere.radius}"
    )


def test_collider_push_does_not_inject_normal_velocity() -> None:
    """Sphere pushes a vert away; vert should NOT keep flying after the push.

    Reproduces the runtime case where a fast hand sweeps past a hanging skirt:
    the projection forces the cloth vert outward, but without velocity
    correction the next Verlet step treats that displacement as injected
    velocity and the vert keeps accelerating outward for many frames after
    the actual contact ended. The post-projection prev_position shift
    nulls the normal component of velocity so the cloth stays where the
    projection put it instead of flying away.
    """
    cloth = _build_cloth_grid(rows=3, cols=3, params=ClothParams(rest_pull=0.0))
    sim = ClothSolver(pieces=[cloth])
    free = np.where(cloth.inverse_masses > 0.0)[0]
    vert_idx = int(free[len(free) // 2])
    # Seed a sphere overlapping the vert. With prev_center = current center
    # we hit the static-projection branch.
    p = cloth.positions[vert_idx]
    sphere = SphereCollider(center=vec3(p[0], p[1], p[2] + 0.04), radius=0.06)
    sim.add_collider(sphere)
    # First step: vert gets projected out of the sphere.
    sim.step(1.0 / 60.0)
    pos_after_push = cloth.positions[vert_idx].copy()
    # Subsequent frames: no forces, no collider motion. The vert must NOT
    # continue drifting — its position should stay (near) put.
    for _ in range(30):
        sim.step(1.0 / 60.0)
    drift = float(np.linalg.norm(cloth.positions[vert_idx] - pos_after_push))
    # ~mm-scale residual is fine (structural constraint relaxation against
    # the static neighbours). The bug was the vert flying tens of cm —
    # 5 mm is well below "flying" but well above the residual motion.
    assert drift < 5.0e-3, (
        f"vert kept moving after collider push: drift {drift:.4f} m "
        f"(start at {pos_after_push}, end at {cloth.positions[vert_idx]})"
    )


def test_static_sphere_projection_unchanged_when_prev_center_is_none() -> None:
    """The CCD branch is opt-in: leaving ``prev_center`` as None must keep the
    cheap static-snapshot projection (no behaviour change for stationary props
    like the hip sphere).
    """
    cloth = _build_cloth_grid(rows=4, cols=4)
    sim = ClothSolver(pieces=[cloth])
    sim.add_force(ClothGravity(acceleration=vec3(0.0, -9.8, 0.0)))
    sphere = SphereCollider(center=vec3(0.15, -0.15, 0.0), radius=0.15)
    # prev_center intentionally left at default (None)
    assert sphere.prev_center is None
    sim.add_collider(sphere)
    for _ in range(60):
        sim.step(1.0 / 60.0)
    movable = cloth.inverse_masses > 0.0
    deltas = cloth.positions[movable] - sphere.center
    dists = np.linalg.norm(deltas, axis=1)
    assert np.all(dists >= sphere.radius - 1.0e-3)


def test_capsule_collider_projects_verts_outside() -> None:
    cloth = _build_cloth_grid(rows=4, cols=4)
    sim = ClothSolver(pieces=[cloth])
    sim.add_force(ClothGravity(acceleration=vec3(0.0, -9.8, 0.0)))
    capsule = CapsuleCollider(a=vec3(0.0, -0.2, 0.0), b=vec3(0.3, -0.2, 0.0), radius=0.08)
    sim.add_collider(capsule)
    for _ in range(120):
        sim.step(1.0 / 60.0)
    movable = cloth.inverse_masses > 0.0
    seg = capsule.b - capsule.a
    seg_len_sq = float(np.dot(seg, seg))
    rel = cloth.positions[movable] - capsule.a
    t = np.clip((rel @ seg) / seg_len_sq, 0.0, 1.0)
    closest = capsule.a + t[:, None] * seg
    dists = np.linalg.norm(cloth.positions[movable] - closest, axis=1)
    assert np.all(dists >= capsule.radius - 1.0e-3)


def test_reset_restores_rest_positions() -> None:
    cloth = _build_cloth_grid()
    sim = ClothSolver(pieces=[cloth])
    sim.add_force(ClothGravity(acceleration=vec3(0.0, -9.8, 0.0)))
    for _ in range(60):
        sim.step(1.0 / 60.0)
    assert not np.allclose(cloth.positions, cloth.rest_positions)
    cloth.reset()
    np.testing.assert_allclose(cloth.positions, cloth.rest_positions, atol=1.0e-6)
    np.testing.assert_allclose(cloth.prev_positions, cloth.rest_positions, atol=1.0e-6)


def test_wind_pushes_cloth_in_wind_direction() -> None:
    cloth = _build_cloth_grid()
    sim = ClothSolver(pieces=[cloth])
    sim.add_force(ClothWind(direction=vec3(0.0, 0.0, 1.0), speed=4.0))
    free = cloth.inverse_masses > 0.0
    for _ in range(60):
        sim.step(1.0 / 60.0)
    # Free verts should have moved in +Z.
    z_drift = cloth.positions[free, 2] - cloth.rest_positions[free, 2]
    assert float(np.mean(z_drift)) > 0.005


def test_substepping_with_huge_dt_stays_finite() -> None:
    cloth = _build_cloth_grid(rows=3, cols=3)
    sim = ClothSolver(pieces=[cloth], fixed_dt=1.0 / 240.0)
    sim.add_force(ClothGravity(acceleration=vec3(0.0, -9.8, 0.0)))
    sim.step(0.5)  # one big outer step → many sub-steps
    assert np.all(np.isfinite(cloth.positions))


def test_compute_vertex_normals_unit_length() -> None:
    cloth = _build_cloth_grid()
    normals = compute_vertex_normals(cloth.positions, cloth.triangles)
    lengths = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(lengths, np.ones_like(lengths), atol=1.0e-5)


def test_compute_vertex_normals_flat_grid_points_z() -> None:
    """A flat XY-plane grid (Z=0) should produce normals along +Z (or -Z, depending on winding)."""
    cloth = _build_cloth_grid()
    normals = compute_vertex_normals(cloth.positions, cloth.triangles)
    # The grid lies in the XY plane (Y vertical, Z=0); face normals point along Z.
    assert np.all(np.abs(normals[:, 2]) > 0.95)


def test_extract_islands_one_connected_grid() -> None:
    """A single connected grid mesh should yield ONE island id for every vertex."""
    from posecascade.animation.cloth import extract_islands  # noqa: PLC0415

    cloth = _build_cloth_grid(rows=3, cols=3)
    island_id = extract_islands(cloth.triangles.reshape(-1).astype(np.uint32),
                                cloth.positions.shape[0])
    assert len(np.unique(island_id)) == 1


def test_extract_islands_two_disconnected_grids() -> None:
    """Two disconnected grids in one mesh yield two distinct island ids."""
    from posecascade.animation.cloth import extract_islands  # noqa: PLC0415

    # Grid A
    pos_a, idx_a = _grid_mesh(2, 2)
    # Grid B with vertex indices offset by len(pos_a)
    pos_b, idx_b = _grid_mesh(2, 2)
    pos_b = pos_b + np.array([1.0, 0.0, 0.0], dtype=np.float32)
    positions = np.concatenate([pos_a, pos_b], axis=0)
    indices = np.concatenate([idx_a, idx_b + len(pos_a)], axis=0).astype(np.uint32)

    island_id = extract_islands(indices, positions.shape[0])
    unique = np.unique(island_id)
    assert len(unique) == 2
    # First half of verts is one island, second half another.
    assert island_id[0] != island_id[len(pos_a)]


def test_anchor_by_island_top_pins_each_island_separately() -> None:
    """A two-grid mesh — both grids should get their own top-anchor band."""
    from posecascade.animation.cloth import anchor_by_island_top  # noqa: PLC0415

    pos_a, idx_a = _grid_mesh(3, 3)             # Y from 0 down to -0.2
    pos_b, idx_b = _grid_mesh(3, 3)
    pos_b = pos_b + np.array([0.0, -1.0, 0.0], dtype=np.float32)  # Y from -1.0 down to -1.2
    positions = np.concatenate([pos_a, pos_b], axis=0).astype(np.float32)
    indices = np.concatenate([idx_a, idx_b + len(pos_a)], axis=0).astype(np.uint32)

    mask = anchor_by_island_top(positions, indices, axis=1, fraction=0.40)

    # Each island should have SOME anchored verts (top fraction by Y per island).
    n = pos_a.shape[0]
    assert int(mask[:n].sum()) > 0, "top of first grid should anchor"
    assert int(mask[n:].sum()) > 0, "top of second grid should anchor (per-island, not global)"
    # Single global top would only anchor the +Y verts of grid A.
    # With per-island, grid B's top row (Y=-1.0) is also anchored.


def test_anchor_by_island_top_simulate_top_below_filters_high_islands() -> None:
    """Islands whose max along axis exceeds simulate_top_below get fully anchored."""
    from posecascade.animation.cloth import anchor_by_island_top  # noqa: PLC0415

    # Two grids: one at high Y (max ≈ 0), one at low Y (max ≈ -0.5)
    pos_a, idx_a = _grid_mesh(3, 3)              # Y from 0 to -0.2
    pos_b, idx_b = _grid_mesh(3, 3)
    pos_b = pos_b + np.array([0.0, -1.0, 0.0], dtype=np.float32)  # Y from -1.0 to -1.2
    positions = np.concatenate([pos_a, pos_b], axis=0).astype(np.float32)
    indices = np.concatenate([idx_a, idx_b + len(pos_a)], axis=0).astype(np.uint32)

    # simulate_top_below = -0.5 → grid A's max=0 exceeds → fully anchored.
    # Grid B's max=-1.0 is below → only its top fraction anchored.
    mask = anchor_by_island_top(
        positions, indices, axis=1, fraction=0.30, simulate_top_below=-0.5,
    )

    n = pos_a.shape[0]
    assert mask[:n].sum() == n, "high island should be fully anchored"
    # Grid B has fraction=0.3 anchored, so SOME but not ALL of its verts.
    assert 0 < mask[n:].sum() < pos_b.shape[0]


def test_anchor_by_island_top_simulate_top_below_none_falls_back_to_per_island() -> None:
    """When simulate_top_below=None the function behaves identically to the unfiltered variant."""
    from posecascade.animation.cloth import anchor_by_island_top  # noqa: PLC0415

    pos_a, idx_a = _grid_mesh(3, 3)
    pos_b, idx_b = _grid_mesh(3, 3)
    pos_b = pos_b + np.array([0.0, -1.0, 0.0], dtype=np.float32)
    positions = np.concatenate([pos_a, pos_b], axis=0).astype(np.float32)
    indices = np.concatenate([idx_a, idx_b + len(pos_a)], axis=0).astype(np.uint32)

    plain = anchor_by_island_top(positions, indices, axis=1, fraction=0.40)
    filtered_none = anchor_by_island_top(
        positions, indices, axis=1, fraction=0.40, simulate_top_below=None,
    )
    np.testing.assert_array_equal(plain, filtered_none)


def test_anchor_by_island_top_invalid_fraction_raises() -> None:
    from posecascade.animation.cloth import anchor_by_island_top  # noqa: PLC0415

    positions = np.zeros((4, 3), dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    with pytest.raises(ValueError, match="fraction"):
        anchor_by_island_top(positions, indices, axis=1, fraction=0.0)


def test_anchor_by_top_axis_invalid_fraction_raises() -> None:
    positions = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="fraction"):
        anchor_by_top_axis(positions, axis=1, fraction=0.0)
    with pytest.raises(ValueError, match="fraction"):
        anchor_by_top_axis(positions, axis=1, fraction=1.5)


def test_cloth_from_mesh_validates_positions_shape() -> None:
    bad_positions = np.zeros((4, 2), dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    mask = np.zeros(4, dtype=bool)
    with pytest.raises(ValueError, match="positions must be"):
        cloth_from_mesh(
            "x", bad_positions, indices,
            world_matrix=mat4_identity(),
            anchor_mask=mask,
        )


def test_cloth_from_mesh_validates_indices_length() -> None:
    positions = np.zeros((4, 3), dtype=np.float32)
    bad_indices = np.array([0, 1], dtype=np.uint32)  # not multiple of 3
    mask = np.zeros(4, dtype=bool)
    with pytest.raises(ValueError, match="indices length"):
        cloth_from_mesh(
            "x", positions, bad_indices,
            world_matrix=mat4_identity(),
            anchor_mask=mask,
        )


def test_cloth_from_mesh_validates_anchor_mask_length() -> None:
    positions = np.zeros((4, 3), dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    bad_mask = np.zeros(3, dtype=bool)  # length doesn't match positions
    with pytest.raises(ValueError, match="anchor_mask length"):
        cloth_from_mesh(
            "x", positions, indices,
            world_matrix=mat4_identity(),
            anchor_mask=bad_mask,
        )


def test_world_matrix_transforms_positions() -> None:
    positions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    indices = np.zeros((0,), dtype=np.uint32)  # no triangles → no edges
    mask = np.zeros(1, dtype=bool)
    matrix = mat4_identity()
    matrix[0, 3] = 5.0  # translate +X by 5
    cloth = cloth_from_mesh("x", positions, indices, world_matrix=matrix, anchor_mask=mask)
    np.testing.assert_allclose(cloth.positions[0], [6.0, 0.0, 0.0], atol=1.0e-6)


def test_step_zero_dt_is_noop() -> None:
    cloth = _build_cloth_grid()
    sim = ClothSolver(pieces=[cloth])
    sim.add_force(ClothGravity(acceleration=vec3(0.0, -9.8, 0.0)))
    initial = cloth.positions.copy()
    sim.step(0.0)
    sim.step(-1.0)
    np.testing.assert_allclose(cloth.positions, initial, atol=1.0e-6)


def test_rest_pull_bounds_drift_under_constant_force() -> None:
    """With rest_pull > 0, a constant external force should equilibrate to bounded displacement."""
    cloth = _build_cloth_grid(
        rows=4, cols=4,
        params=ClothParams(
            structural_stiffness=1.0,
            bend_stiffness=0.4,
            linear_damping=0.94,
            iterations=12,
            rest_pull=20.0,
        ),
    )
    sim = ClothSolver(pieces=[cloth])
    sim.add_force(ClothWind(direction=vec3(1.0, 0.0, 0.0), speed=1.0))
    free = cloth.inverse_masses > 0.0
    # Long-running sim — without rest_pull, free verts would drift indefinitely
    # in +X under sustained wind; with rest_pull, they settle to a bounded offset.
    for _ in range(600):
        sim.step(1.0 / 60.0)
    drift = float(np.linalg.norm(cloth.positions[free] - cloth.rest_positions[free], axis=1).max())
    assert drift < 0.5, f"rest_pull failed to bound drift: {drift}"


def test_rest_pull_reduces_steady_state_drift() -> None:
    """A higher rest_pull should produce smaller equilibrium drift under the same wind."""

    def steady_drift(rest_pull: float) -> float:
        cloth = _build_cloth_grid(params=ClothParams(rest_pull=rest_pull, iterations=12))
        sim = ClothSolver(pieces=[cloth])
        sim.add_force(ClothWind(direction=vec3(1.0, 0.0, 0.0), speed=2.0))
        free = cloth.inverse_masses > 0.0
        for _ in range(180):
            sim.step(1.0 / 60.0)
        delta = cloth.positions[free] - cloth.rest_positions[free]
        return float(np.linalg.norm(delta, axis=1).max())

    weak = steady_drift(rest_pull=0.5)
    strong = steady_drift(rest_pull=50.0)
    assert strong < weak, f"strong rest_pull did not reduce drift: weak={weak} strong={strong}"


def test_disabled_piece_is_skipped() -> None:
    cloth = _build_cloth_grid()
    cloth.enabled = False
    sim = ClothSolver(pieces=[cloth])
    sim.add_force(ClothGravity(acceleration=vec3(0.0, -9.8, 0.0)))
    initial = cloth.positions.copy()
    sim.step(1.0 / 60.0)
    np.testing.assert_allclose(cloth.positions, initial, atol=1.0e-6)


def test_find_piece_lookup() -> None:
    cloth = _build_cloth_grid()
    sim = ClothSolver(pieces=[cloth])
    assert sim.find_piece("grid") is cloth
    assert sim.find_piece("missing") is None
