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
from posecascade.utils.profiling import frame_section

# Cython kernels for the PBD inner loop + static-collider projections.
# Built by ``setup.py`` at install time; absent in pure source checkouts
# until ``python setup.py build_ext --inplace`` (or ``pip install -e .``)
# is run. The Python fallbacks below match the kernel maths bit-for-bit
# within float32 rounding so the rest of the engine is dtype-agnostic.
try:
    from posecascade.animation import _cloth_kernels as _native  # type: ignore[attr-defined]
except ImportError:                                                         # pragma: no cover
    _native = None

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
# Fraction of the tangential velocity that survives a collider contact. < 1.0
# adds dynamic friction so a vert in contact with a moving collider does not
# accelerate along the surface frame-after-frame — without this, cloth that
# the hand sweeps past gains free-sliding speed and flutters wildly long
# after the contact ends. 0.2 = 80% of tangential energy absorbed per contact;
# the cloth still slides along the surface but no swing is amplified.
_CONTACT_TANGENT_RETENTION = 0.2
# Pad added around the per-substep piece bounding box when broad-phase
# culling colliders. The piece AABB is computed once per substep, then
# the constraint loop runs 8+ iterations during which vertices can drift
# a few millimetres. The padding has to cover that drift OR a collider
# whose AABB sits just outside the unpadded piece AABB might intersect
# after iteration and be wrongly skipped. 2 cm is conservative — large
# enough to never wrongly cull, small enough to still prune the obvious
# misses (e.g. a hand-sphere across the room).
_PIECE_AABB_PADDING = 0.02
# Sub-AABB binning splits the cloth into K vertex-index bins so each
# collider's projection runs only on the bins its AABB overlaps. K=4 is
# a sweet spot for typical MMD-scale meshes (a few hundred verts per
# piece): four narrow stripes give meaningful spatial separation while
# keeping per-bin AABB compute under ~10 µs. Smaller pieces (< _MIN_BIN_
# VERTEX_COUNT) skip binning entirely — the overhead would dominate.
_DEFAULT_BIN_COUNT = 4
_MIN_BIN_VERTEX_COUNT = 32


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
    """Body sphere — cloth vertices that fall inside are projected to the surface.

    ``prev_center`` is the centre at the start of the current frame's substeps.
    When set (and distinct from ``center``), :func:`_project_sphere` treats the
    sphere as having SWEPT from ``prev_center`` to ``center`` and projects verts
    outside the resulting capsule volume — fixing the fast-hand-tunnels-into-
    skirt case where a static-snapshot check at the substep's end would leave
    a vert inside the cloth on the far side of the hand's swept path. The
    declarative bone-follow driver fills ``prev_center`` each frame; leave it
    as ``None`` for stationary colliders so the cheaper static projection runs.
    """

    center: Vec3
    radius: float
    skin_offset: float = 0.005  # extra clearance so verts don't tangent-graze the sphere
    prev_center: Vec3 | None = None


@dataclass
class CapsuleCollider:
    """Capsule between ``a`` and ``b`` of given ``radius``. Cloth verts project outside.

    ``prev_a`` / ``prev_b`` mirror :attr:`SphereCollider.prev_center` — when both
    are set, the projection treats the capsule as having swept from
    ``(prev_a, prev_b)`` to ``(a, b)``; verts inside either capsule (or the
    convex sweep linking matching endpoints) get pushed to the nearest exit.
    """

    a: Vec3
    b: Vec3
    radius: float
    skin_offset: float = 0.005
    prev_a: Vec3 | None = None
    prev_b: Vec3 | None = None


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
    # Passive skin-deform mode: when True, the solver short-circuits past
    # Verlet integration, distance / bend constraints, and external forces
    # — it ONLY snaps every vertex to its current ``rest_positions``
    # (typically refreshed each tick by a ``_SkinTargetFollower`` so it
    # equals the rigid LBS-skinned pose) and then projects against the
    # registered colliders. Use this to drive a "skinned mesh that gets
    # pushed out of body capsules" pass without paying for spring
    # dynamics — e.g. to stop dress / hair / sleeve verts from clipping
    # into the torso during animation without re-rigging the asset.
    passive_skin_deform: bool = False


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
    # Pre-converted ``np.intp`` views of the edge / bend index columns.
    # ``np.bincount`` and fancy indexing require ``intp`` weights, and the
    # implicit ``uint32 → intp`` copy used to happen inside the PBD inner
    # loop dozens of times per step. Caching them here amortises that to
    # once at build time. Filled by ``__post_init__``; never read raw.
    _edge_a_idx: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(0, dtype=np.intp), init=False, repr=False,
    )
    _edge_b_idx: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(0, dtype=np.intp), init=False, repr=False,
    )
    _bend_a_idx: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(0, dtype=np.intp), init=False, repr=False,
    )
    _bend_b_idx: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(0, dtype=np.intp), init=False, repr=False,
    )
    # Combined-side scatter buffers — concatenation of the ``a`` and ``b``
    # halves of each constraint set so the PBD pass can do one
    # ``np.bincount`` per axis instead of two (a-scatter then b-scatter).
    # ``combined_scale`` already carries the sign convention (first half
    # negated) and the mass + valence weighting that depends only on the
    # immutable piece topology, so the inner loop never has to recompute it.
    # ``scratch_weights`` is a reusable (2M,) buffer the solver writes per
    # axis through ``np.multiply(..., out=)`` so it avoids allocating a
    # fresh weight array each iteration.
    _edge_combined_idx: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(0, dtype=np.intp), init=False, repr=False,
    )
    _edge_combined_scale: NDArray[np.float32] = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), init=False, repr=False,
    )
    _edge_scratch_weights: NDArray[np.float32] = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), init=False, repr=False,
    )
    _bend_combined_idx: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(0, dtype=np.intp), init=False, repr=False,
    )
    _bend_combined_scale: NDArray[np.float32] = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), init=False, repr=False,
    )
    _bend_scratch_weights: NDArray[np.float32] = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), init=False, repr=False,
    )
    # Sub-AABB binning state: ``_bin_ranges`` is a list of ``(start, end)``
    # vertex index ranges, computed once at build time from
    # ``_DEFAULT_BIN_COUNT``. Pieces below ``_MIN_BIN_VERTEX_COUNT`` get a
    # single full-range bin so collider projection costs less than the bin
    # bookkeeping. Bin-level AABBs (``_bin_min`` / ``_bin_max``) are
    # refreshed per substep — they would otherwise drift through the PBD
    # iteration loop and lose pruning power.
    _bin_ranges: list[tuple[int, int]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._refresh_index_caches()
        self._refresh_bin_ranges()

    def _refresh_bin_ranges(self) -> None:
        """Compute the per-bin index ranges for sub-AABB broad-phase."""
        n = self.positions.shape[0]
        if n < _MIN_BIN_VERTEX_COUNT:
            self._bin_ranges = [(0, n)]
            return
        k = _DEFAULT_BIN_COUNT
        chunk = n // k
        self._bin_ranges = [
            (i * chunk, (i + 1) * chunk if i < k - 1 else n) for i in range(k)
        ]

    def _refresh_index_caches(self) -> None:
        """Re-derive intp index columns + scatter scales from immutable topology.

        Call this after any code path that swaps in different edges, bends,
        inverse masses, or valence arrays — tests that build a ClothPiece
        directly via ``replace`` do this implicitly because dataclass
        replace re-invokes ``__post_init__``.
        """
        self._edge_a_idx = self.edges[:, 0].astype(np.intp, copy=False)
        self._edge_b_idx = self.edges[:, 1].astype(np.intp, copy=False)
        (
            self._edge_combined_idx,
            self._edge_combined_scale,
            self._edge_scratch_weights,
        ) = _build_combined_scatter(
            self._edge_a_idx, self._edge_b_idx,
            self.inverse_masses, self.edge_valence,
        )
        if self.bends.size:
            self._bend_a_idx = self.bends[:, 0].astype(np.intp, copy=False)
            self._bend_b_idx = self.bends[:, 1].astype(np.intp, copy=False)
            (
                self._bend_combined_idx,
                self._bend_combined_scale,
                self._bend_scratch_weights,
            ) = _build_combined_scatter(
                self._bend_a_idx, self._bend_b_idx,
                self.inverse_masses, self.bend_valence,
            )
        else:
            self._bend_a_idx = np.empty(0, dtype=np.intp)
            self._bend_b_idx = np.empty(0, dtype=np.intp)
            self._bend_combined_idx = np.empty(0, dtype=np.intp)
            self._bend_combined_scale = np.empty(0, dtype=np.float32)
            self._bend_scratch_weights = np.empty(0, dtype=np.float32)

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
    """Owns multiple :class:`ClothPiece`\\s + colliders + global forces + clock.

    ``ground_y`` (when set) clamps every movable vertex's Y coordinate to
    stay at or above the plane after each substep. The declarative runtime
    reads ``ground: {kind: flat, y: N}`` and forwards N here, so cloth
    pieces stop sinking through the same floor the foot planter uses for
    IK — no per-pose tuning required. ``None`` disables the clamp
    (matches the default behaviour from before the engine added this
    feature, so scenes without an explicit ground keep working).
    """

    pieces: list[ClothPiece] = field(default_factory=list)
    colliders: list[object] = field(default_factory=list)  # SphereCollider | CapsuleCollider
    forces: list[ClothForce] = field(default_factory=list)
    fixed_dt: float = _DEFAULT_FIXED_DT
    time: float = 0.0
    ground_y: float | None = None

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

    def set_ground_y(self, ground_y: float | None) -> None:
        """Set or clear the ground-plane Y clamp. ``None`` disables it."""
        self.ground_y = None if ground_y is None else float(ground_y)

    def step(self, dt: float) -> None:
        """Advance every cloth piece by ``dt`` seconds."""
        if dt <= 0.0:
            return
        with frame_section("cloth.step"):
            remaining = float(dt)
            while remaining > _NUMERIC_EPSILON:
                sub = min(self.fixed_dt, remaining)
                self._step_once(sub)
                remaining -= sub

    def _step_once(self, dt: float) -> None:
        for piece in self.pieces:
            if piece.enabled:
                _integrate_piece(
                    piece, dt, self.time, self.forces, self.colliders,
                    ground_y=self.ground_y,
                )
        self.time += dt


def _integrate_piece(
    piece: ClothPiece,
    dt: float,
    time: float,
    forces: Sequence[ClothForce],
    colliders: Sequence[object],
    ground_y: float | None = None,
) -> None:
    """One Verlet step + N constraint iterations + collider projection."""
    if piece.params.passive_skin_deform:
        # Skin-only mode: snap positions to ``rest_positions`` (which a
        # SkinTargetFollower has just refreshed to the LBS-skinned pose).
        # Collider projection is NOT run here — the cloth host does it
        # in a second pass that respects per-collider bone filters
        # (so a hand-collider following Left wrist doesn't try to push
        # the hand's own skin verts out of itself).
        np.copyto(piece.prev_positions, piece.positions)
        np.copyto(piece.positions, piece.rest_positions)
        if ground_y is not None:
            _project_ground_plane(
                piece.positions, piece.prev_positions, piece.inverse_masses, ground_y,
            )
        return
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

    has_bends = piece.bends.size > 0 and piece.params.bend_stiffness > 0.0
    # Broad-phase: per-bin AABBs (the union of which is the piece AABB) +
    # per-collider AABB tuple precomputed once per substep. Re-computing
    # inside the iteration loop would cost more than it saves; the piece-
    # AABB padding (``_PIECE_AABB_PADDING``) absorbs the tens of milli-
    # metres a vert can move across 8 PBD passes. The collider AABBs are
    # static across a substep (the cloth host swaps them between solver
    # steps, never inside an iteration), so caching is safe. Tuples of
    # plain floats keep the per-iteration overlap test out of NumPy.
    bin_aabbs = _compute_bin_aabbs(piece)
    collider_aabbs = [_collider_aabb_tuple(c) for c in colliders] if colliders else None
    for _ in range(piece.params.iterations):
        _solve_distance_constraints(
            piece.positions,
            piece._edge_a_idx,
            piece._edge_b_idx,
            piece.edge_rest_lengths,
            piece._edge_combined_idx,
            piece._edge_combined_scale,
            piece._edge_scratch_weights,
            piece.params.structural_stiffness,
        )
        if has_bends:
            _solve_distance_constraints(
                piece.positions,
                piece._bend_a_idx,
                piece._bend_b_idx,
                piece.bend_rest_lengths,
                piece._bend_combined_idx,
                piece._bend_combined_scale,
                piece._bend_scratch_weights,
                piece.params.bend_stiffness,
            )
        _project_colliders(
            piece.positions, piece.prev_positions, piece.inverse_masses,
            colliders,
            bin_ranges=piece._bin_ranges,
            bin_aabbs=bin_aabbs,
            collider_aabbs=collider_aabbs,
        )
        if ground_y is not None:
            _project_ground_plane(
                piece.positions, piece.prev_positions, piece.inverse_masses, ground_y,
            )


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
    # ``zeros_like`` already returned float32 (positions are float32) and every
    # accumulation kept the dtype, so ``astype`` here would just duplicate the
    # buffer. ``copy=False`` returns ``total`` itself when the dtype matches.
    return total.astype(np.float32, copy=False)


def _row_norms(v: NDArray[np.float32]) -> NDArray[np.float32]:
    """``np.linalg.norm(v, axis=1)`` for (N, 3) vectors, measured ~2× faster.

    ``einsum('ij,ij->i', v, v)`` lets NumPy fuse the per-row dot product into
    a single tight loop and skip the dispatch overhead ``linalg.norm`` carries
    for its general N-dim case. The PBD constraint pass and every collider
    projection runs this dozens of times per substep, so the win compounds.
    """
    return np.sqrt(np.einsum("ij,ij->i", v, v))


def _build_combined_scatter(
    a_idx: NDArray[np.intp],
    b_idx: NDArray[np.intp],
    inverse_masses: NDArray[np.float32],
    valence: NDArray[np.float32],
) -> tuple[NDArray[np.intp], NDArray[np.float32], NDArray[np.float32]]:
    """Precompute the immutable scatter inputs for :func:`_solve_distance_constraints`.

    Returns ``(combined_idx, combined_scale, scratch_weights)`` where
    ``combined_idx`` is ``[a_idx, b_idx]`` so a single bincount visits every
    affected vertex, and ``combined_scale`` carries the per-constraint
    inverse-mass / valence weighting already signed: the first half is
    ``-factor_a / valence[a]`` and the second is ``+factor_b / valence[b]``.
    Folding the sign into the scale lets the solver use ``positions +=
    scatter`` uniformly instead of separate ``-=``/``+=`` passes.

    ``scratch_weights`` is an empty (2M,) float32 buffer the solver fills
    via ``np.multiply(..., out=...)`` each axis — avoids allocating a fresh
    weight buffer inside the PBD inner loop.
    """
    m = a_idx.shape[0]
    if m == 0:
        return (
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    wa = inverse_masses[a_idx]
    wb = inverse_masses[b_idx]
    wsum = wa + wb
    # Match the original solver: vertex pairs where both are anchored
    # (wsum == 0) get a denominator of 1.0 — their factors evaluate to 0
    # because wa and wb are both 0, so the constraint produces no scatter.
    safe_wsum = np.where(wsum > 0.0, wsum, 1.0)
    factor_a = wa / safe_wsum
    factor_b = wb / safe_wsum
    scale_a_neg = (-factor_a / valence[a_idx]).astype(np.float32, copy=False)
    scale_b = (factor_b / valence[b_idx]).astype(np.float32, copy=False)
    combined_idx = np.concatenate([a_idx, b_idx])
    combined_scale = np.concatenate([scale_a_neg, scale_b]).astype(np.float32, copy=False)
    scratch_weights = np.empty(2 * m, dtype=np.float32)
    return combined_idx, combined_scale, scratch_weights


def _clamp_per_vertex_displacement(
    delta: NDArray[np.float32],
    max_delta: float,
) -> NDArray[np.float32]:
    """Clamp each vertex's per-step displacement so a numeric blow-up cannot teleport."""
    norms = _row_norms(delta)
    too_big = norms > max_delta
    if not np.any(too_big):
        return delta
    out = delta.copy()
    scale = max_delta / np.maximum(norms[too_big], _NUMERIC_EPSILON)
    # delta and scale are both float32; copy=False keeps the multiplication
    # result in place rather than duplicating a fresh buffer per substep.
    out[too_big] = (delta[too_big] * scale[:, None]).astype(np.float32, copy=False)
    return out


def _solve_distance_constraints(
    positions: NDArray[np.float32],
    a_idx: NDArray[np.intp],
    b_idx: NDArray[np.intp],
    rest_lengths: NDArray[np.float32],
    combined_idx: NDArray[np.intp],
    combined_scale: NDArray[np.float32],
    scratch_weights: NDArray[np.float32],
    stiffness: float,
) -> None:
    """Apply one Jacobi-style PBD distance pass (mutates ``positions``).

    Dispatches to the Cython kernel when the compiled extension is
    available; the pure-Python NumPy implementation below is the
    bit-for-bit equivalent fallback used in source checkouts that haven't
    run the build step. ``scratch_weights`` is only consulted in the
    fallback path — the kernel writes directly into ``positions`` via
    typed memory views so it doesn't need the scratch buffer.
    """
    if a_idx.shape[0] == 0:
        return
    if _native is not None:
        _native.solve_distance_constraints(
            positions, a_idx, b_idx, rest_lengths,
            combined_idx, combined_scale,
            stiffness, _NUMERIC_EPSILON,
        )
        return
    _solve_distance_constraints_numpy(
        positions, a_idx, b_idx, rest_lengths,
        combined_idx, combined_scale, scratch_weights,
        stiffness,
    )


def _solve_distance_constraints_numpy(
    positions: NDArray[np.float32],
    a_idx: NDArray[np.intp],
    b_idx: NDArray[np.intp],
    rest_lengths: NDArray[np.float32],
    combined_idx: NDArray[np.intp],
    combined_scale: NDArray[np.float32],
    scratch_weights: NDArray[np.float32],
    stiffness: float,
) -> None:
    """NumPy fallback for :func:`_solve_distance_constraints`.

    Each correction is divided by the affected vertex's valence so that summed
    parallel corrections don't over-shoot — without this, high-stiffness solves
    diverge to NaN within a few substeps. The ``a`` and ``b`` halves of each
    constraint are concatenated into ``combined_idx`` (with the ``a`` side's
    sign baked into ``combined_scale``) so each axis needs ONE ``bincount``
    instead of two — measured 12% faster than the split-scatter version.
    """
    m = a_idx.shape[0]
    delta = positions[a_idx] - positions[b_idx]
    lengths = _row_norms(delta)
    safe_lengths = np.maximum(lengths, _NUMERIC_EPSILON)
    direction = delta / safe_lengths[:, None]
    error = (lengths - rest_lengths) * stiffness
    correction = error[:, None] * direction
    n_verts = positions.shape[0]
    scale_neg = combined_scale[:m]
    scale_pos = combined_scale[m:]
    weights_neg = scratch_weights[:m]
    weights_pos = scratch_weights[m:]
    for axis in range(3):
        column = correction[:, axis]
        np.multiply(column, scale_neg, out=weights_neg)
        np.multiply(column, scale_pos, out=weights_pos)
        scatter = np.bincount(combined_idx, weights=scratch_weights, minlength=n_verts)
        positions[:, axis] += scatter.astype(np.float32, copy=False)


_AABBTuple = tuple[float, float, float, float, float, float]


def _compute_bin_aabbs(piece: ClothPiece) -> list[_AABBTuple]:
    """Return the per-bin padded AABB tuples for the current piece pose.

    The bin index ranges are fixed at piece-build time; this just walks
    them, takes the min/max over the current ``positions`` slice, and
    pads by ``_PIECE_AABB_PADDING`` to absorb PBD iteration drift. A
    single-bin piece falls through to one whole-mesh AABB.
    """
    bins: list[_AABBTuple] = []
    positions = piece.positions
    for start, end in piece._bin_ranges:                                # noqa: SLF001
        chunk = positions[start:end]
        mn = chunk.min(axis=0)
        mx = chunk.max(axis=0)
        bins.append((
            float(mn[0]) - _PIECE_AABB_PADDING,
            float(mn[1]) - _PIECE_AABB_PADDING,
            float(mn[2]) - _PIECE_AABB_PADDING,
            float(mx[0]) + _PIECE_AABB_PADDING,
            float(mx[1]) + _PIECE_AABB_PADDING,
            float(mx[2]) + _PIECE_AABB_PADDING,
        ))
    return bins


def _collider_aabb_tuple(collider: object) -> _AABBTuple | None:
    """Return ``(min_x, min_y, min_z, max_x, max_y, max_z)`` for ``collider``.

    Returns a flat tuple of Python floats — the overlap test runs in the
    iteration hot path and the numpy-array form allocated enough per call
    to wipe out the broad-phase gain. The capsule sweep through
    ``prev_a/b`` (sphere through ``prev_center``) is unioned in so a fast-
    moving collider's sweep volume is still considered.

    Returns ``None`` for collider types we don't know how to bound — those
    fall through to the full projection without culling.
    """
    if isinstance(collider, SphereCollider):
        r = collider.radius + collider.skin_offset
        cx = float(collider.center[0])
        cy = float(collider.center[1])
        cz = float(collider.center[2])
        if collider.prev_center is not None:
            px = float(collider.prev_center[0])
            py = float(collider.prev_center[1])
            pz = float(collider.prev_center[2])
            return (
                min(cx, px) - r, min(cy, py) - r, min(cz, pz) - r,
                max(cx, px) + r, max(cy, py) + r, max(cz, pz) + r,
            )
        return (cx - r, cy - r, cz - r, cx + r, cy + r, cz + r)
    if isinstance(collider, CapsuleCollider):
        r = collider.radius + collider.skin_offset
        ax = float(collider.a[0])
        ay = float(collider.a[1])
        az = float(collider.a[2])
        bx = float(collider.b[0])
        by = float(collider.b[1])
        bz = float(collider.b[2])
        mnx = min(ax, bx)
        mny = min(ay, by)
        mnz = min(az, bz)
        mxx = max(ax, bx)
        mxy = max(ay, by)
        mxz = max(az, bz)
        if collider.prev_a is not None and collider.prev_b is not None:
            pax = float(collider.prev_a[0])
            pay = float(collider.prev_a[1])
            paz = float(collider.prev_a[2])
            pbx = float(collider.prev_b[0])
            pby = float(collider.prev_b[1])
            pbz = float(collider.prev_b[2])
            mnx = min(mnx, pax, pbx)
            mny = min(mny, pay, pby)
            mnz = min(mnz, paz, pbz)
            mxx = max(mxx, pax, pbx)
            mxy = max(mxy, pay, pby)
            mxz = max(mxz, paz, pbz)
        return (mnx - r, mny - r, mnz - r, mxx + r, mxy + r, mxz + r)
    return None


def _project_ground_plane(
    positions: NDArray[np.float32],
    prev_positions: NDArray[np.float32],
    inverse_masses: NDArray[np.float32],
    ground_y: float,
) -> None:
    """Clamp movable vertex Y to stay at or above ``ground_y``.

    Mirrors the projection into ``prev_positions`` so the next Verlet step
    doesn't read the lift as downward velocity — same pattern as the
    sphere / capsule projection. Anchored verts (inverse_mass == 0) are
    left alone; they're typically waistband / collar verts pinned to the
    rig and the rig is what should be moved if the anchors are below the
    floor, not the cloth's own correction pass.
    """
    movable = inverse_masses > 0.0
    if not np.any(movable):
        return
    y = positions[:, 1]
    below = movable & (y < ground_y)
    if not np.any(below):
        return
    lift = ground_y - y[below]
    positions[below, 1] = ground_y
    # Carry the lift into prev_positions so velocity = pos - prev keeps the
    # tangential component but zeroes the normal (vertical) component.
    prev_positions[below, 1] = prev_positions[below, 1] + lift


def _project_colliders(
    positions: NDArray[np.float32],
    prev_positions: NDArray[np.float32],
    inverse_masses: NDArray[np.float32],
    colliders: Sequence[object],
    bin_ranges: Sequence[tuple[int, int]] | None = None,
    bin_aabbs: Sequence[_AABBTuple] | None = None,
    collider_aabbs: Sequence[_AABBTuple | None] | None = None,
) -> None:
    """Push movable vertices outside every collider (sphere or capsule).

    ``prev_positions`` is updated in lockstep with the position correction so
    the next Verlet step doesn't read the projection displacement as velocity.
    Without this, a fast-moving collider pushes a vert several cm in one step,
    the integrator then interprets that gap as a high velocity, and the vert
    keeps flying for many frames after the actual contact ended. The
    correction nulls the NORMAL-component velocity only — tangential velocity
    is preserved so cloth still slides along a contact surface.

    ``inverse_masses`` flows down rather than a precomputed ``movable`` bool
    mask so the Cython projection kernels can read it directly — converting
    bool → float32 every substep would land back in the alloc hot path.
    """
    if not colliders:
        return
    # Broad-phase: per-bin AABBs (and matching vertex-index ranges) +
    # per-collider AABB tuple precomputed once per substep. For each
    # collider, walk every bin and dispatch the projection kernel only on
    # the vertex range whose bin actually overlaps — a hand-sphere typically
    # touches one bin of a four-bin skirt, so the kernel scans ~120 verts
    # instead of all 480. Six float comparisons per (bin, collider) pair
    # is dirt-cheap compared to a missed kernel scan.
    if bin_ranges is None or bin_aabbs is None:
        bin_ranges = ((0, positions.shape[0]),)
        bin_aabbs = (None,)
    boxes = collider_aabbs
    for index, collider in enumerate(colliders):
        cbox = boxes[index] if boxes is not None else None
        for bin_index, (bstart, bend) in enumerate(bin_ranges):
            bbox = bin_aabbs[bin_index]
            if cbox is not None and bbox is not None and (
                bbox[3] < cbox[0] or bbox[0] > cbox[3]
                or bbox[4] < cbox[1] or bbox[1] > cbox[4]
                or bbox[5] < cbox[2] or bbox[2] > cbox[5]
            ):
                continue
            if isinstance(collider, SphereCollider):
                _project_sphere(
                    positions, prev_positions, inverse_masses, collider,
                    start=bstart, end=bend,
                )
            elif isinstance(collider, CapsuleCollider):
                _project_capsule(
                    positions, prev_positions, inverse_masses, collider,
                    start=bstart, end=bend,
                )


def _project_sphere(
    positions: NDArray[np.float32],
    prev_positions: NDArray[np.float32],
    inverse_masses: NDArray[np.float32],
    collider: SphereCollider,
    start: int = 0,
    end: int = -1,
) -> None:
    """Push verts outside the sphere — uses swept-capsule volume when prev_center is set.

    Without CCD a fast bone (hand, foot) can move farther than its own radius in
    one substep; a vert behind the swept path is checked at the substep's END
    position, finds nothing intersecting, and gets stranded on the wrong side
    of the cloth. The swept capsule from ``prev_center`` to ``center`` catches
    those verts and projects them out the nearest face.

    Hot-path optimisation: when the collider barely moved (motion < radius),
    a static projection at the current position covers the contact area
    anyway. Skipping the swept-capsule fallback there cuts down the number
    of verts the projection touches per substep — important because every
    affected vert sends a tangential ripple through the structural mesh
    that takes many frames to damp out, and the cloth visibly flaps near
    a hovering hand even when the hand has effectively stopped moving.
    """
    radius = collider.radius + collider.skin_offset
    if collider.prev_center is None:
        _project_static_sphere(
            positions, prev_positions, inverse_masses, collider.center, radius, start, end,
        )
        return
    motion = collider.center - collider.prev_center
    motion_sq = float(np.dot(motion, motion))
    if motion_sq < radius * radius:
        _project_static_sphere(
            positions, prev_positions, inverse_masses, collider.center, radius, start, end,
        )
        return
    _project_swept(
        positions, prev_positions, inverse_masses,
        collider.prev_center, collider.center, radius, start, end,
    )


def _project_capsule(
    positions: NDArray[np.float32],
    prev_positions: NDArray[np.float32],
    inverse_masses: NDArray[np.float32],
    collider: CapsuleCollider,
    start: int = 0,
    end: int = -1,
) -> None:
    """Push verts outside the capsule — sweeps the midpoint when prev_a/prev_b are set.

    Approximation of the full 4D minkowski-sweep volume by two projections:
    the current capsule pose, plus a diagonal capsule from the previous
    midpoint to the current midpoint. The midpoint-sweep catches most
    tunneling cases (a forearm or shin rigidly translating between frames)
    without the 4× cost of the full four-segment projection — that variant
    was measurably laggy at iterations≥12 with 6 capsule colliders.
    """
    radius = collider.radius + collider.skin_offset
    if collider.prev_a is None or collider.prev_b is None:
        _project_static_capsule(
            positions, prev_positions, inverse_masses,
            collider.a, collider.b, radius, start, end,
        )
        return
    motion_a = collider.a - collider.prev_a
    motion_b = collider.b - collider.prev_b
    motion_a_sq = float(np.dot(motion_a, motion_a))
    motion_b_sq = float(np.dot(motion_b, motion_b))
    radius_sq = radius * radius
    if motion_a_sq < radius_sq and motion_b_sq < radius_sq:
        # Either still or moving less than the capsule radius — static
        # projection at the current pose already covers the contact band,
        # and adding the midpoint sweep here just spreads the projection
        # over more verts and seeds the structural ripples the user sees
        # as the skirt fluttering.
        _project_static_capsule(
            positions, prev_positions, inverse_masses,
            collider.a, collider.b, radius, start, end,
        )
        return
    # Current capsule pose first — biggest single-pose volume.
    _project_static_capsule(
        positions, prev_positions, inverse_masses,
        collider.a, collider.b, radius, start, end,
    )
    # Midpoint sweep: capsule joining the previous and current midpoints.
    # Cheap second pass that catches a vert the static projection missed
    # because the collider had translated past it.
    prev_mid = (collider.prev_a + collider.prev_b) * 0.5
    curr_mid = (collider.a + collider.b) * 0.5
    _project_static_capsule(
        positions, prev_positions, inverse_masses,
        prev_mid, curr_mid, radius, start, end,
    )


def _project_static_sphere(
    positions: NDArray[np.float32],
    prev_positions: NDArray[np.float32],
    inverse_masses: NDArray[np.float32],
    center: Vec3,
    radius: float,
    start: int = 0,
    end: int = -1,
) -> None:
    n = positions.shape[0]
    stop = n if end < 0 else end
    if _native is not None:
        # The kernel inlines the ``inverse_masses == 0`` skip — no bool mask
        # allocation per substep. ``start`` / ``stop`` lets sub-AABB binning
        # restrict the scan to a vertex range.
        _native.project_static_sphere(
            positions, prev_positions, inverse_masses,
            float(center[0]), float(center[1]), float(center[2]),
            float(radius),
            _CONTACT_TANGENT_RETENTION, _NUMERIC_EPSILON,
            start, stop,
        )
        return
    movable = inverse_masses > 0.0
    if start != 0 or stop != n:
        range_mask = np.zeros(n, dtype=bool)
        range_mask[start:stop] = True
        movable = movable & range_mask
    delta = positions - center
    dist = _row_norms(delta)
    inside = movable & (dist < radius)
    if not np.any(inside):
        return
    safe_dist = np.maximum(dist[inside], _NUMERIC_EPSILON)
    normal = delta[inside] / safe_dist[:, None]
    # All inputs are float32; ``copy=False`` skips the duplicate buffer.
    new_positions = (center + normal * radius).astype(np.float32, copy=False)
    _apply_projection_with_velocity_correction(
        positions, prev_positions, inside, new_positions, normal,
    )


def _project_static_capsule(
    positions: NDArray[np.float32],
    prev_positions: NDArray[np.float32],
    inverse_masses: NDArray[np.float32],
    a: Vec3,
    b: Vec3,
    radius: float,
    start: int = 0,
    end: int = -1,
) -> None:
    n = positions.shape[0]
    stop = n if end < 0 else end
    seg = b - a
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq < _NUMERIC_EPSILON:
        _project_static_sphere(
            positions, prev_positions, inverse_masses, a, radius, start, stop,
        )
        return
    if _native is not None:
        _native.project_static_capsule(
            positions, prev_positions, inverse_masses,
            float(a[0]), float(a[1]), float(a[2]),
            float(b[0]), float(b[1]), float(b[2]),
            float(radius),
            _CONTACT_TANGENT_RETENTION, _NUMERIC_EPSILON,
            start, stop,
        )
        return
    movable = inverse_masses > 0.0
    if start != 0 or stop != n:
        range_mask = np.zeros(n, dtype=bool)
        range_mask[start:stop] = True
        movable = movable & range_mask
    rel = positions - a
    t = np.clip((rel @ seg) / seg_len_sq, 0.0, 1.0)
    closest = a + t[:, None] * seg
    delta = positions - closest
    dist = _row_norms(delta)
    inside = movable & (dist < radius)
    if not np.any(inside):
        return
    safe_dist = np.maximum(dist[inside], _NUMERIC_EPSILON)
    normal = delta[inside] / safe_dist[:, None]
    # All inputs are float32; ``copy=False`` skips the duplicate buffer.
    new_positions = (closest[inside] + normal * radius).astype(np.float32, copy=False)
    _apply_projection_with_velocity_correction(
        positions, prev_positions, inside, new_positions, normal,
    )


def _project_swept(
    positions: NDArray[np.float32],
    prev_positions: NDArray[np.float32],
    inverse_masses: NDArray[np.float32],
    prev_center: Vec3,
    center: Vec3,
    radius: float,
    start: int = 0,
    end: int = -1,
) -> None:
    """Project verts outside the capsule swept by a moving sphere."""
    _project_static_capsule(
        positions, prev_positions, inverse_masses,
        prev_center, center, radius, start, end,
    )


def _apply_projection_with_velocity_correction(
    positions: NDArray[np.float32],
    prev_positions: NDArray[np.float32],
    inside_mask: NDArray[np.bool_],
    new_positions: NDArray[np.float32],
    contact_normal: NDArray[np.float32],
) -> None:
    """Commit ``new_positions`` and dampen velocity at the contact.

    Two corrections in one shift of ``prev_positions``:

    1. **Null the normal component.** A vert that ended up inside the collider
       was projected back out; the implicit Verlet velocity ``pos - prev``
       would otherwise contain the projection displacement and the cloth
       would bounce off the surface or fly perpendicular to it. Removing
       the normal component absorbs the impact.
    2. **Scale the tangential component by ``_CONTACT_TANGENT_RETENTION``.**
       The cloth still slides along the contact but with friction — without
       this, a swept collider drags the cloth and seeds a tangential ripple
       that takes many frames to damp through ``linear_damping`` alone.
       Fast hand passes through the skirt's vicinity were leaving the cloth
       flapping for nearly a second after the actual contact ended.

    Net effect: ``pos - prev`` after this call equals ``retention * tangent``,
    so the next Verlet step reads only damped tangential velocity at the
    contact point.
    """
    old_positions = positions[inside_mask]
    old_prev = prev_positions[inside_mask]
    velocity = old_positions - old_prev
    normal_component = np.einsum("ij,ij->i", velocity, contact_normal)
    tangential = (velocity - normal_component[:, None] * contact_normal) * (
        _CONTACT_TANGENT_RETENTION
    )
    positions[inside_mask] = new_positions
    prev_positions[inside_mask] = new_positions - tangential


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
    lengths = _row_norms(normals.astype(np.float32, copy=False))
    safe = np.maximum(lengths, _NUMERIC_EPSILON)
    return (normals / safe[:, None]).astype(np.float32)
