"""Vertex-level cloth simulation (Position-Based Dynamics, Verlet integration).

Each cloth piece is a triangle mesh whose vertices become particles. Distance
constraints along mesh edges keep the cloth from stretching; bend constraints
between vertices opposite shared edges resist creasing. Anchored vertices stay
pinned at their rest position (pinned at the seam where the cloth attaches to
the body). Sphere colliders project vertices outside body proxies so the cloth
doesn't pass through the character.

The integrator is symplectic Verlet with PBD constraint projection — fast,
stable for arbitrary stiffness up to ``stiffness_iterations``, and small in
state (positions + previous positions per vertex). All math is vectorised
through numpy so a 2k-vert cape simulates in well under one frame at 60 Hz.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from posecascade.utils.math3d import Mat4, Vec3, vec3

_NUMERIC_EPSILON = 1.0e-8
_DEFAULT_GRAVITY = (0.0, -9.8, 0.0)
_DEFAULT_LINEAR_DAMPING = 0.985
_DEFAULT_STRUCTURAL_STIFFNESS = 0.85
_DEFAULT_BEND_STIFFNESS = 0.10
_DEFAULT_ITERATIONS = 8
_DEFAULT_SUBSTEPS = 2
_DEFAULT_FIXED_DT = 1.0 / 60.0
_MAX_DELTA_PER_FRAME = 0.05      # cap a single Verlet step's force-driven displacement
_MAX_VELOCITY_PER_STEP = 0.20    # cap implicit velocity (positions - prev_positions) per substep
_VEC3_LENGTH = 3                 # 3-component points / vectors
_INDICES_PER_TRIANGLE = 3        # mesh indices come in triplets
_MIN_BEND_TRIANGLE_PAIR = 2      # an interior edge is shared by ≥ 2 triangles → contributes a bend


class ClothForce(Protocol):
    """Per-vertex world-space force; evaluated each substep."""

    def force_at(self, world_position: Vec3, time: float) -> Vec3:  # pragma: no cover
        ...


@dataclass
class ClothGravity:
    """Uniform gravity applied to every cloth vertex."""

    acceleration: Vec3 = field(default_factory=lambda: vec3(*_DEFAULT_GRAVITY))

    def force_at(self, world_position: Vec3, time: float) -> Vec3:
        del world_position, time
        return self.acceleration


@dataclass
class ClothWind:
    """Directional wind with optional sinusoidal turbulence (same model as the spring sim)."""

    direction: Vec3 = field(default_factory=lambda: vec3(1.0, 0.0, 0.0))
    speed: float = 1.0
    turbulence_amplitude: float = 0.0
    turbulence_frequency_hz: float = 1.5

    def force_at(self, world_position: Vec3, time: float) -> Vec3:
        del world_position
        base = self.direction * self.speed
        if self.turbulence_amplitude == 0.0:
            return base.astype(np.float32)
        cross = np.cross(self.direction, vec3(0.0, 1.0, 0.0)).astype(np.float32)
        cross_len = float(np.linalg.norm(cross))
        cross = vec3(1.0, 0.0, 0.0) if cross_len < _NUMERIC_EPSILON else cross / cross_len
        jitter = float(np.sin(time * self.turbulence_frequency_hz * 2.0 * np.pi))
        return (base + cross * self.turbulence_amplitude * jitter).astype(np.float32)


@dataclass
class SphereCollider:
    """Body sphere — cloth vertices that fall inside are projected to the surface."""

    center: Vec3
    radius: float
    skin_offset: float = 0.005  # extra clearance so verts don't tangent-graze the sphere


@dataclass
class CapsuleCollider:
    """Capsule between ``a`` and ``b`` of given ``radius``. Cloth verts project outside."""

    a: Vec3
    b: Vec3
    radius: float
    skin_offset: float = 0.005


@dataclass
class ClothParams:
    """Tunable per-piece parameters. Defaults give a soft, draping cape feel.

    ``rest_pull`` (1/s²) is a soft acceleration pulling every free vertex back
    toward its rest position — gives the cloth its "memory" of the artist's
    drape pose. Without it, sustained one-directional forces (wind, gravity)
    would drift the cloth away from rest indefinitely because PBD constraints
    converge too slowly to balance them at force level. Set to 0 for a true
    free-swinging cloth (where the only resistance is the boundary anchors).
    """

    structural_stiffness: float = _DEFAULT_STRUCTURAL_STIFFNESS
    bend_stiffness: float = _DEFAULT_BEND_STIFFNESS
    linear_damping: float = _DEFAULT_LINEAR_DAMPING
    iterations: int = _DEFAULT_ITERATIONS
    substeps: int = _DEFAULT_SUBSTEPS
    rest_pull: float = 4.0


@dataclass
class ClothPiece:
    """Per-cloth state + topology. ``positions`` and ``prev_positions`` are world-space.

    ``edge_valence`` and ``bend_valence`` are precomputed per-vertex constraint
    counts: each constraint's correction is divided by the affected vertex's
    valence so summing parallel (Jacobi-style) constraint corrections via
    ``np.add.at`` doesn't over-shoot. Without this scaling, a vertex shared by
    six edges receives 6× the intended correction in a single iteration —
    the integration explodes within milliseconds at any non-trivial stiffness.
    """

    name: str
    positions: NDArray[np.float32]              # (N, 3) world space
    prev_positions: NDArray[np.float32]         # (N, 3) for Verlet integration
    rest_positions: NDArray[np.float32]         # (N, 3) world space at init — for reset/anchors
    inverse_masses: NDArray[np.float32]         # (N,) — 0 for anchored verts
    edges: NDArray[np.uint32]                   # (M, 2) structural edges
    edge_rest_lengths: NDArray[np.float32]      # (M,)
    edge_valence: NDArray[np.float32]           # (N,) per-vertex structural-edge count, ≥ 1
    bends: NDArray[np.uint32]                   # (K, 2) bend pairs (may be (0, 2))
    bend_rest_lengths: NDArray[np.float32]      # (K,)
    bend_valence: NDArray[np.float32]           # (N,) per-vertex bend-pair count, ≥ 1
    triangles: NDArray[np.uint32]               # (T, 3) — for normal recomputation
    params: ClothParams = field(default_factory=ClothParams)
    enabled: bool = True

    def reset(self) -> None:
        """Snap vertices back to their rest positions and clear Verlet velocity."""
        np.copyto(self.positions, self.rest_positions)
        np.copyto(self.prev_positions, self.rest_positions)


def cloth_from_mesh(
    name: str,
    local_positions: NDArray[np.float32],
    indices: NDArray[np.uint32],
    *,
    world_matrix: Mat4,
    anchor_mask: NDArray[np.bool_],
    params: ClothParams | None = None,
) -> ClothPiece:
    """Build a :class:`ClothPiece` from raw mesh data.

    Positions are transformed into world space via ``world_matrix`` so the
    simulation operates in a single coordinate frame regardless of where the
    cloth sits in the scene hierarchy.
    """
    if local_positions.shape[1] != _VEC3_LENGTH:
        raise ValueError(f"positions must be (N, 3), got {local_positions.shape}")
    if indices.size % _INDICES_PER_TRIANGLE != 0:
        raise ValueError(f"indices length {indices.size} not divisible by 3")
    if anchor_mask.shape[0] != local_positions.shape[0]:
        raise ValueError("anchor_mask length must match vertex count")

    world_positions = _transform_points(local_positions, world_matrix)
    triangles = indices.reshape(-1, 3).astype(np.uint32, copy=False)
    edges, bends = _extract_edges_and_bends(triangles)
    edge_rest = np.linalg.norm(
        world_positions[edges[:, 0]] - world_positions[edges[:, 1]], axis=1
    ).astype(np.float32)
    bend_rest = np.linalg.norm(
        world_positions[bends[:, 0]] - world_positions[bends[:, 1]], axis=1
    ).astype(np.float32) if bends.size else np.zeros((0,), dtype=np.float32)

    inverse_masses = np.where(anchor_mask, 0.0, 1.0).astype(np.float32)
    edge_valence = _vertex_valence(edges, len(world_positions))
    bend_valence = _vertex_valence(bends, len(world_positions))
    return ClothPiece(
        name=name,
        positions=world_positions.copy(),
        prev_positions=world_positions.copy(),
        rest_positions=world_positions.copy(),
        inverse_masses=inverse_masses,
        edges=edges,
        edge_rest_lengths=edge_rest,
        edge_valence=edge_valence,
        bends=bends,
        bend_rest_lengths=bend_rest,
        bend_valence=bend_valence,
        triangles=triangles,
        params=params or ClothParams(),
    )


def _vertex_valence(constraints: NDArray[np.uint32], vertex_count: int) -> NDArray[np.float32]:
    """Count how many ``constraints`` (rows of vertex pairs) touch each vertex."""
    if constraints.size == 0:
        return np.ones(vertex_count, dtype=np.float32)
    counts = np.bincount(constraints.flatten(), minlength=vertex_count).astype(np.float32)
    return np.maximum(counts, 1.0)


def anchor_by_top_axis(
    positions: NDArray[np.float32],
    axis: int = 1,
    fraction: float = 0.15,
) -> NDArray[np.bool_]:
    """Pin verts in the top ``fraction`` of the bbox along ``axis``.

    Default ``axis=1`` (Y, world-up after the importer's Y-up coercion). With
    ``fraction=0.15``, the top 15 % of the bounding box along Y becomes
    anchored — appropriate for a cape that hangs from a waist seam. Use
    ``fraction=0.05`` for a tight pinned strip; ``fraction=0.50`` to anchor
    the whole upper half (e.g. a flag attached on one side).
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    coords = positions[:, axis]
    lo = float(coords.min())
    hi = float(coords.max())
    # ``fraction`` is the SIZE of the anchored slab as a fraction of the bbox.
    # Threshold = top boundary minus that slab thickness.
    threshold = hi - (hi - lo) * fraction
    return coords >= threshold


def extract_islands(
    indices: NDArray[np.uint32],
    vertex_count: int,
) -> NDArray[np.int32]:
    """Return per-vertex connected-component id for every vertex in the mesh.

    Two vertices share an id when an unbroken chain of triangles connects them.
    Vertices that no triangle touches get their own unique id. Used by
    :func:`anchor_by_island_top` to anchor each disconnected piece of a
    multi-component cloth (e.g. a sleeve mesh whose puffy upper part and
    long hanging flap are separate islands) at its own top edge.

    Implementation: weighted union-find with path compression. Linear in the
    number of triangle edges with near-constant inverse-Ackermann amortised
    cost per union — far below the integration cost of a single sim step.
    """
    parent = np.arange(vertex_count, dtype=np.int32)

    def find(x: int) -> int:
        # Iterative path-halving, no recursion (vertex_count can be large).
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    if indices.size:
        triangles = indices.reshape(-1, 3)
        for tri in triangles:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            union(a, b)
            union(b, c)
    return np.array([find(i) for i in range(vertex_count)], dtype=np.int32)


def anchor_by_island_top(
    positions: NDArray[np.float32],
    indices: NDArray[np.uint32],
    axis: int = 1,
    fraction: float = 0.15,
    *,
    simulate_top_below: float | None = None,
) -> NDArray[np.bool_]:
    """Pin the top ``fraction`` along ``axis`` of EACH connected mesh island.

    Where :func:`anchor_by_top_axis` finds one global top band, this finds the
    top band per disconnected piece. For a sleeve mesh whose puffy upper sleeve
    sits at Z≈0.18 and whose hanging flap sits at Z≈0.10, both pieces get their
    own anchor strip — the upper sleeve pins to the shoulder, the flap pins to
    the elbow seam, and each can sway independently from its own attach point.

    If ``simulate_top_below`` is given, any island whose maximum coordinate
    along ``axis`` exceeds that threshold is **fully anchored** instead — useful
    for filtering a multi-piece decoration mesh down to just the lower flowing
    elements (e.g. anchor the puffy upper sleeves rigidly, simulate only the
    long hanging hems below them).
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    island_id = extract_islands(indices, positions.shape[0])
    mask = np.zeros(positions.shape[0], dtype=bool)
    for unique_id in np.unique(island_id):
        members = np.where(island_id == unique_id)[0]
        if members.size == 0:
            continue
        coords = positions[members, axis]
        island_max = float(coords.max())
        if simulate_top_below is not None and island_max > simulate_top_below:
            # Above the cutoff — anchor the whole island so it stays fixed to the body.
            mask[members] = True
            continue
        lo = float(coords.min())
        threshold = island_max - (island_max - lo) * fraction
        mask[members[coords >= threshold]] = True
    return mask


@dataclass
class ClothSolver:
    """Owns multiple :class:`ClothPiece`\\s + colliders + global forces + clock."""

    pieces: list[ClothPiece] = field(default_factory=list)
    colliders: list[object] = field(default_factory=list)  # SphereCollider | CapsuleCollider
    forces: list[ClothForce] = field(default_factory=list)
    fixed_dt: float = _DEFAULT_FIXED_DT
    time: float = 0.0

    def add_piece(self, piece: ClothPiece) -> None:
        self.pieces.append(piece)

    def add_collider(self, collider: SphereCollider | CapsuleCollider) -> None:
        self.colliders.append(collider)

    def add_force(self, force: ClothForce) -> None:
        self.forces.append(force)

    def find_piece(self, name: str) -> ClothPiece | None:
        for piece in self.pieces:
            if piece.name == name:
                return piece
        return None

    def step(self, dt: float) -> None:
        """Advance every cloth piece by ``dt`` seconds."""
        if dt <= 0.0:
            return
        remaining = float(dt)
        while remaining > _NUMERIC_EPSILON:
            sub = min(self.fixed_dt, remaining)
            self._step_once(sub)
            remaining -= sub

    def _step_once(self, dt: float) -> None:
        for piece in self.pieces:
            if piece.enabled:
                _integrate_piece(piece, dt, self.time, self.forces, self.colliders)
        self.time += dt


def _integrate_piece(
    piece: ClothPiece,
    dt: float,
    time: float,
    forces: Sequence[ClothForce],
    colliders: Sequence[object],
) -> None:
    """One Verlet step + N constraint iterations + collider projection."""
    inv_mass = piece.inverse_masses[:, None]
    external_accel = _accumulate_force(piece, forces, time) * inv_mass
    # Soft rest-pull keeps the cloth's drape "memory": a free vertex experiences
    # a restoring acceleration proportional to its displacement from rest. Without
    # this, sustained one-directional wind/gravity would drift the cloth away
    # forever because Jacobi-PBD constraints converge too slowly to enforce
    # force balance — see _solve_distance_constraints for why.
    rest_offset = piece.rest_positions - piece.positions
    rest_accel = piece.params.rest_pull * rest_offset * inv_mass
    accel = external_accel + rest_accel
    raw_velocity = piece.positions - piece.prev_positions
    # Cap velocity BEFORE damping so a runaway constraint correction last frame
    # cannot keep accelerating this frame; without this, sphere/capsule projection
    # can inject an effective velocity that explodes the integrator within a few
    # substeps when many neighbouring verts share the same collider response.
    velocity = _clamp_per_vertex_displacement(raw_velocity, _MAX_VELOCITY_PER_STEP)
    velocity = velocity * piece.params.linear_damping
    delta = velocity + accel * (dt * dt)
    delta = _clamp_per_vertex_displacement(delta, _MAX_DELTA_PER_FRAME)

    next_positions = piece.positions + delta
    # Anchored verts: snap them back to rest each substep so PBD correction
    # cannot drift them. Inverse mass = 0 means constraints leave them alone,
    # but we also want zero motion from the integration step.
    anchor = piece.inverse_masses == 0.0
    if np.any(anchor):
        next_positions[anchor] = piece.rest_positions[anchor]

    piece.prev_positions = piece.positions.copy()
    piece.positions = next_positions

    for _ in range(piece.params.iterations):
        _solve_distance_constraints(
            piece.positions,
            piece.inverse_masses,
            piece.edges,
            piece.edge_rest_lengths,
            piece.edge_valence,
            piece.params.structural_stiffness,
        )
        if piece.bends.size > 0 and piece.params.bend_stiffness > 0.0:
            _solve_distance_constraints(
                piece.positions,
                piece.inverse_masses,
                piece.bends,
                piece.bend_rest_lengths,
                piece.bend_valence,
                piece.params.bend_stiffness,
            )
        _project_colliders(piece.positions, piece.inverse_masses, colliders)


def _accumulate_force(
    piece: ClothPiece,
    forces: Sequence[ClothForce],
    time: float,
) -> NDArray[np.float32]:
    """Return per-vertex acceleration from every active force.

    Treated as accelerations, not forces — the cloth has no mass parameter
    (vertices have inverse_mass = 1 or 0). ``Gravity.acceleration`` is already
    in world units/s².
    """
    if not forces:
        return np.zeros_like(piece.positions)
    total = np.zeros_like(piece.positions)
    for force in forces:
        # Forces here ignore world_position for our default set (Gravity, Wind),
        # so a single evaluation suffices. PointForce-style forces would need
        # per-vertex evaluation — wire that up when you add one.
        sample = force.force_at(piece.positions[0], time)
        total = total + sample
    return total.astype(np.float32)


def _clamp_per_vertex_displacement(
    delta: NDArray[np.float32],
    max_delta: float,
) -> NDArray[np.float32]:
    """Clamp each vertex's per-step displacement so a numeric blow-up cannot teleport."""
    norms = np.linalg.norm(delta, axis=1)
    too_big = norms > max_delta
    if not np.any(too_big):
        return delta
    out = delta.copy()
    scale = max_delta / np.maximum(norms[too_big], _NUMERIC_EPSILON)
    out[too_big] = (delta[too_big] * scale[:, None]).astype(np.float32)
    return out


def _solve_distance_constraints(
    positions: NDArray[np.float32],
    inverse_masses: NDArray[np.float32],
    edges: NDArray[np.uint32],
    rest_lengths: NDArray[np.float32],
    valence: NDArray[np.float32],
    stiffness: float,
) -> None:
    """Apply one Jacobi-style PBD distance pass (mutates ``positions``).

    Each correction is divided by the affected vertex's valence so that summed
    parallel corrections don't over-shoot — without this, high-stiffness solves
    diverge to NaN within a few substeps.

    Scatter-add uses ``np.bincount`` per-axis instead of ``np.add.at`` — the
    latter is a ~6× slower Python-level loop and dominates cloth tick cost on
    multi-thousand-vertex meshes.
    """
    if edges.shape[0] == 0:
        return
    a_idx = edges[:, 0].astype(np.intp, copy=False)
    b_idx = edges[:, 1].astype(np.intp, copy=False)
    delta = positions[a_idx] - positions[b_idx]
    lengths = np.linalg.norm(delta, axis=1)
    safe_lengths = np.maximum(lengths, _NUMERIC_EPSILON)
    direction = delta / safe_lengths[:, None]
    error = (lengths - rest_lengths) * stiffness
    wa = inverse_masses[a_idx]
    wb = inverse_masses[b_idx]
    wsum = wa + wb
    safe_wsum = np.where(wsum > 0.0, wsum, 1.0)
    factor_a = wa / safe_wsum
    factor_b = wb / safe_wsum
    scale_a = factor_a / valence[a_idx]
    scale_b = factor_b / valence[b_idx]
    correction = error[:, None] * direction
    # Per-axis bincount: each axis is one O(M) reduction over edges. Net cost
    # is 6 bincount calls per pass — much faster than 2 × np.add.at on (M, 3).
    n_verts = positions.shape[0]
    for axis in range(3):
        weight = correction[:, axis]
        scatter_a = np.bincount(a_idx, weights=weight * scale_a, minlength=n_verts)
        scatter_b = np.bincount(b_idx, weights=weight * scale_b, minlength=n_verts)
        positions[:, axis] -= scatter_a.astype(np.float32)
        positions[:, axis] += scatter_b.astype(np.float32)


def _project_colliders(
    positions: NDArray[np.float32],
    inverse_masses: NDArray[np.float32],
    colliders: Sequence[object],
) -> None:
    """Push movable vertices outside every collider (sphere or capsule)."""
    if not colliders:
        return
    movable = inverse_masses > 0.0
    for collider in colliders:
        if isinstance(collider, SphereCollider):
            _project_sphere(positions, movable, collider)
        elif isinstance(collider, CapsuleCollider):
            _project_capsule(positions, movable, collider)


def _project_sphere(
    positions: NDArray[np.float32],
    movable: NDArray[np.bool_],
    collider: SphereCollider,
) -> None:
    delta = positions - collider.center
    dist = np.linalg.norm(delta, axis=1)
    radius = collider.radius + collider.skin_offset
    inside = movable & (dist < radius)
    if not np.any(inside):
        return
    safe_dist = np.maximum(dist[inside], _NUMERIC_EPSILON)
    direction = delta[inside] / safe_dist[:, None]
    positions[inside] = (collider.center + direction * radius).astype(np.float32)


def _project_capsule(
    positions: NDArray[np.float32],
    movable: NDArray[np.bool_],
    collider: CapsuleCollider,
) -> None:
    seg = collider.b - collider.a
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq < _NUMERIC_EPSILON:
        sphere = SphereCollider(
            center=collider.a, radius=collider.radius, skin_offset=collider.skin_offset
        )
        _project_sphere(positions, movable, sphere)
        return
    rel = positions - collider.a
    t = np.clip((rel @ seg) / seg_len_sq, 0.0, 1.0)
    closest = collider.a + t[:, None] * seg
    delta = positions - closest
    dist = np.linalg.norm(delta, axis=1)
    radius = collider.radius + collider.skin_offset
    inside = movable & (dist < radius)
    if not np.any(inside):
        return
    safe_dist = np.maximum(dist[inside], _NUMERIC_EPSILON)
    direction = delta[inside] / safe_dist[:, None]
    positions[inside] = (closest[inside] + direction * radius).astype(np.float32)


def _extract_edges_and_bends(
    triangles: NDArray[np.uint32],
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """Return ``(edges, bends)``: unique mesh edges + bend pairs (verts opposite shared edges)."""
    edge_to_third: dict[tuple[int, int], list[int]] = {}
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v, third in ((a, b, c), (b, c, a), (c, a, b)):
            key = (u, v) if u < v else (v, u)
            edge_to_third.setdefault(key, []).append(third)

    edges = (
        np.array(sorted(edge_to_third.keys()), dtype=np.uint32)
        if edge_to_third
        else np.zeros((0, 2), dtype=np.uint32)
    )
    bend_pairs: list[tuple[int, int]] = []
    for thirds in edge_to_third.values():
        if len(thirds) >= _MIN_BEND_TRIANGLE_PAIR:
            # If three+ triangles meet at one edge (non-manifold), pair the first two.
            t1, t2 = thirds[0], thirds[1]
            bend_pairs.append((t1, t2) if t1 < t2 else (t2, t1))
    bends = (
        np.array(sorted(set(bend_pairs)), dtype=np.uint32)
        if bend_pairs
        else np.zeros((0, 2), dtype=np.uint32)
    )
    return edges, bends


def _transform_points(points: NDArray[np.float32], matrix: Mat4) -> NDArray[np.float32]:
    """Apply a 4x4 affine matrix to ``(N, 3)`` points and return ``(N, 3)`` result."""
    n = points.shape[0]
    homog = np.hstack([points, np.ones((n, 1), dtype=np.float32)])
    out = (homog @ matrix.T)[:, :3]
    return out.astype(np.float32, copy=False)


def compute_vertex_normals(
    positions: NDArray[np.float32],
    triangles: NDArray[np.uint32],
) -> NDArray[np.float32]:
    """Area-weighted vertex normals from triangle topology — used to refresh the normal VBO.

    Uses ``np.bincount`` for the per-vertex face-normal accumulation: ~6× faster
    than the equivalent ``np.add.at`` triple, which matters because this runs
    every frame for every cloth piece.
    """
    v0 = positions[triangles[:, 0]]
    v1 = positions[triangles[:, 1]]
    v2 = positions[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    n_verts = positions.shape[0]
    t0 = triangles[:, 0].astype(np.intp, copy=False)
    t1 = triangles[:, 1].astype(np.intp, copy=False)
    t2 = triangles[:, 2].astype(np.intp, copy=False)
    normals = np.zeros_like(positions)
    for axis in range(3):
        weight = face_normals[:, axis]
        normals[:, axis] = (
            np.bincount(t0, weights=weight, minlength=n_verts)
            + np.bincount(t1, weights=weight, minlength=n_verts)
            + np.bincount(t2, weights=weight, minlength=n_verts)
        )
    lengths = np.linalg.norm(normals, axis=1)
    safe = np.maximum(lengths, _NUMERIC_EPSILON)
    return (normals / safe[:, None]).astype(np.float32)
