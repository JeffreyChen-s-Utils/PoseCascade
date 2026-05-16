"""Engine-side driver that builds cloth pieces from scene components and steps each frame.

Mirrors :class:`~posecascade.animation.physics_host.PhysicsHost` for the cloth
solver: scans the scene for :class:`~posecascade.scene.component.ClothComponent`
markers, snapshots each tagged mesh into a :class:`ClothPiece`, and drives
:meth:`ClothSolver.step` once per frame. The renderer pulls per-frame position
+ normal updates by iterating :meth:`ClothHost.iter_dirty_bindings`.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from posecascade.animation.cloth import (
    CapsuleCollider,
    ClothForce,
    ClothGravity,
    ClothParams,
    ClothPiece,
    ClothSolver,
    SphereCollider,
    anchor_by_island_top,
    anchor_by_top_axis,
    cloth_from_mesh,
    compute_vertex_normals,
)
from posecascade.assets.types import ImportedScene
from posecascade.scene.component import ClothComponent, SkinRefComponent
from posecascade.scene.node import Node
from posecascade.utils.logging import get_logger
from posecascade.utils.math3d import Mat4, mat4_identity, vec3

_log = get_logger(__name__)
_DEFAULT_GRAVITY = (0.0, -2.5, 0.0)


@dataclass
class ClothBinding:
    """Pairs a scene Node + mesh primitive with the cloth piece simulating it.

    The renderer uses this to locate the GPU buffers it needs to refresh: given
    a :class:`Node`, look up its GL meshes and overwrite the position + normal
    VBOs from ``piece.positions``.
    """

    node: Node
    mesh_index: int
    piece: ClothPiece
    # Captured at registration. iter_local_state recomputes inv(world) each
    # frame; this only stays as the fallback when the live world matrix is
    # singular.
    world_to_local: Mat4


@dataclass
class _SkinTargetFollower:
    """Drives a cloth piece's per-vert ``rest_positions`` from skeletal skinning.

    The cloth solver's existing ``rest_pull`` term applies a soft acceleration
    pulling every free vertex toward its ``rest_positions``. By overwriting
    ``rest_positions`` each tick with the bone-skinned target — i.e. where
    the vertex would sit if rigid LBS skinning were applied — the cloth
    follows the skeleton (legs swinging under a skirt, arms moving inside a
    sleeve) while still allowing dynamic drape, wind, and gravity to deflect
    it off the rigid target. With ``rest_pull`` high the cloth tracks the
    skinned pose closely; with it low, the cloth wobbles freely around it.

    Without this follower, a skirt cloth anchored to ``Root_M_01`` stays at
    its waistband-rest position while the legs swing through it — the cloth
    has no idea where the legs are. Adding this follower lets each cloth vert
    inherit its mesh skinning weights, so verts with high ``Hip_L_02`` weight
    move with the left leg while waistband verts stay glued to the root.
    """

    piece: ClothPiece
    # ``Skin`` reference (typed ``object`` to avoid a circular import on
    # :mod:`posecascade.assets.types`). Used to key the renderer's
    # bone-matrix-cache hand-off so the host's already-walked parent
    # chains feed the toon / outline / shadow passes instead of being
    # re-computed there.
    skin: object
    skin_joints: tuple[Node, ...]
    inverse_bind: NDArray[np.float32]
    bind_positions: NDArray[np.float32]
    joints_per_vert: NDArray[np.int32]
    weights_per_vert: NDArray[np.float32]
    # Optional per-vert bind-pose normals — when set, the LBS pass
    # additionally produces skinned normals stored in
    # ``piece.cached_skinned_normals`` so ``iter_local_state`` can
    # skip the ~3 ms per-frame ``compute_vertex_normals`` recompute
    # (the topology is fixed; the LBS already rotates the bind normals
    # by the per-vert weighted bone matrix's 3x3 sub-block).
    bind_normals: NDArray[np.float32] | None = None
    # Reusable scratch buffers — preallocated at registration so the
    # per-tick LBS pass never re-allocates large per-vert arrays.
    # Numbers below are for a 30k-vert body mesh; ``_combined`` (the
    # 4x4 weighted matrix per vert) is the largest at 1.9 MB.
    _bind_h: NDArray[np.float32] = field(init=False, repr=False)
    _combined: NDArray[np.float32] = field(init=False, repr=False)
    _gather: NDArray[np.float32] = field(init=False, repr=False)
    _skinned_h: NDArray[np.float32] = field(init=False, repr=False)
    _skinned_n: NDArray[np.float32] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = self.bind_positions.shape[0]
        self._bind_h = np.concatenate(
            [self.bind_positions, np.ones((n, 1), dtype=np.float32)], axis=1,
        )
        self._combined = np.empty((n, 4, 4), dtype=np.float32)
        self._gather = np.empty((n, 4, 4), dtype=np.float32)
        self._skinned_h = np.empty((n, 4, 1), dtype=np.float32)
        self._skinned_n = np.empty((n, 3, 1), dtype=np.float32)


@dataclass
class _AnchorFollower:
    """Track a cloth piece against a moving bone's FULL world frame each tick.

    Each tick the host computes the bone's RELATIVE transform since the
    previous tick (``M_current @ inv(M_last)``) and applies that transform
    to every cloth vert (anchors AND free verts) and to all three
    position arrays (``positions``, ``prev_positions``, ``rest_positions``).
    This effectively runs the cloth simulation in the bone's moving frame
    while preserving its world-frame velocity = positions - prev_positions
    (rotated through the same transform).

    Why full transform (not just translation): when the rig yaws 180 deg
    between gait phases, applying only the translation to free verts
    would leave them at the original orientation while the anchors snap
    to the rotated bone — the cloth visibly tears in half right at the
    waistband (anchors flipped 180 deg, free verts un-rotated, structural
    edges spanning impossible distances). Tracking the full transform
    keeps the cloth as a coherent piece through any rig motion.

    After the relative transform, the anchor verts are snapped to their
    exact ``anchor_offsets_local`` * current_bone_world position, which
    recovers any floating-point drift accumulated across many ticks.
    """

    piece: ClothPiece
    bone_node: Node
    anchor_indices: NDArray[np.int64]
    anchor_offsets_local: NDArray[np.float32]
    last_bone_world: NDArray[np.float32]


@dataclass
class ClothHost:
    """Owns a :class:`ClothSolver` and binds it to scene nodes/meshes."""

    _solver: ClothSolver = field(default_factory=ClothSolver)
    _bindings: list[ClothBinding] = field(default_factory=list)
    _registered_nodes: set[int] = field(default_factory=set)
    _gravity_installed: bool = False
    _anchor_followers: list[_AnchorFollower] = field(default_factory=list)
    _skin_followers: list[_SkinTargetFollower] = field(default_factory=list)
    # Maps id(collider) → frozenset of joint indices to EXCLUDE from
    # that collider's push during ``passive_skin_deform`` projection.
    # Populated by ``register_collider_bone_filter`` — a hand-sphere
    # collider following ``Left wrist`` registers the wrist joint plus
    # every finger descendant, so the wrist-collider doesn't try to
    # push the hand's own skin verts out of itself.
    _collider_bone_filters: dict[int, frozenset[int]] = field(default_factory=dict)
    # Maps id(piece) → (N,) int array of dominant joint index per
    # vertex. Built lazily when the first filter-aware projection runs;
    # used together with ``_collider_bone_filters`` to mask out verts
    # that should NOT be pushed by a given collider.
    _piece_dominant_joint: dict[int, NDArray[np.int32]] = field(default_factory=dict)
    # Per (piece, collider) cache of the eligible-vert mask + flat index
    # array. Skinning weights are static so this set NEVER changes once
    # registered — caching avoids the per-frame ``np.isin`` on 30k+
    # verts inside the hot push-out loop. Invalidated only on
    # ``reset`` / ``remove_pieces_for_subtree``.
    _eligible_cache: dict[
        int, dict[int, tuple[NDArray[np.bool_], NDArray[np.int64]]],
    ] = field(default_factory=dict)
    # Per-tick ``id(skin) → joint_matrices (J, 4, 4)`` cache, populated by
    # ``_update_skin_targets``. The renderer reads from this in
    # ``apply_cloth_state`` so it doesn't re-walk the parent chains a
    # cloth-tracked rig has already walked this frame — see the
    # ``renderer._adopt_cloth_bone_matrix_cache`` shim.
    _last_joint_matrix_cache: dict[int, NDArray[np.float32]] = field(
        default_factory=dict,
    )
    # Per-piece cached world-space skinned normals, refreshed each tick
    # by the LBS pass. ``iter_local_state`` uses these instead of running
    # the ~3 ms ``compute_vertex_normals`` pass on the deformed positions
    # — the per-vert weighted bone-matrix rotation already gives correct
    # normals, and topology is fixed.
    _piece_skinned_normals: dict[int, NDArray[np.float32]] = field(
        default_factory=dict,
    )
    # Set of ``id(piece)`` whose passive-skin LBS + collider push is
    # handled on the GPU via the compute-shader dispatcher. The renderer
    # populates this once after registering the piece with the dispatcher
    # (see ``renderer.prepare_gpu_passive_skin``). When set, the host's
    # ``_update_skin_targets`` skips the per-vert LBS gather + matmul for
    # that piece (still computes joint matrices — the renderer's bone
    # matrix cache needs them); ``_project_passive_pieces`` skips the
    # collider push; ``iter_local_state`` skips the local-state yield so
    # the renderer doesn't try to re-upload positions the GPU wrote.
    _gpu_managed_pieces: set[int] = field(default_factory=set)
    # Track the most recent imported scene so user scripts (via physics_lite) can
    # register cloth pieces by node reference without the script needing direct
    # access to the importer's flat mesh tuple.
    _last_imported: ImportedScene | None = None
    # World-Y plane below which cloth verts get clamped each tick. When set,
    # verts below ``floor_y`` are snapped onto it and stay there until the
    # solver lifts them — but a cloth whose equilibrium sits below the floor
    # ends up with its TOP verts near the rest position and its BOTTOM verts
    # pinned to the plane, visually a stretched mesh smear on the floor.
    # Default ``None`` (no clamp); callers who need it set it explicitly,
    # and only when ``rest_pull`` is tuned strongly enough that the cloth
    # won't simply collapse onto the plane.
    floor_y: float | None = None

    @property
    def solver(self) -> ClothSolver:
        return self._solver

    def install_default_forces(self) -> None:
        """Install default world-space gravity (idempotent)."""
        if self._gravity_installed:
            return
        self._solver.add_force(ClothGravity(acceleration=vec3(*_DEFAULT_GRAVITY)))
        self._gravity_installed = True

    def register_imported_scene(self, imported: ImportedScene) -> None:
        """Scan ``imported.scene`` for :class:`ClothComponent` markers and rig each one.

        Caches ``imported`` so :meth:`add_cloth_for_node` can resolve mesh data
        from a Node reference alone — needed by sandboxed scripts which never
        see the importer's flat mesh tuple directly.
        """
        self._last_imported = imported
        if imported.scene is None:
            return
        for node in imported.scene.root.traverse():
            for component in node.components:
                if isinstance(component, ClothComponent):
                    self._register_one(node, component, imported)

    @property
    def last_imported(self) -> ImportedScene | None:
        return self._last_imported

    def add_cloth_for_node(
        self,
        node: Node,
        *,
        mesh_index: int | None = None,
        cloth_name: str | None = None,
        anchor_axis: int = 1,
        anchor_fraction: float = 0.15,
        anchor_mode: str = "top_axis",
        simulate_top_below: float | None = None,
        structural_stiffness: float = 0.85,
        bend_stiffness: float = 0.10,
        linear_damping: float = 0.985,
        iterations: int = 8,
        substeps: int = 2,
        rest_pull: float = 4.0,
    ) -> ClothPiece | None:
        """Build a cloth piece from a Node's mesh and register it.

        ``mesh_index`` defaults to the first mesh on the node's
        :class:`~posecascade.scene.component.MeshRefComponent`. Returns the
        :class:`ClothPiece` so the caller can mutate its parameters; ``None``
        if no scene has been registered yet or the node has no mesh.
        """
        if self._last_imported is None:
            _log.warning("add_cloth_for_node called before register_imported_scene")
            return None
        resolved = self._resolve_mesh_index(node, mesh_index)
        if resolved is None:
            return None
        component = ClothComponent(
            cloth_name=cloth_name or f"cloth_{node.name}",
            mesh_index=resolved,
            anchor_axis=anchor_axis,
            anchor_fraction=anchor_fraction,
            anchor_mode=anchor_mode,
            simulate_top_below=simulate_top_below,
            structural_stiffness=structural_stiffness,
            bend_stiffness=bend_stiffness,
            linear_damping=linear_damping,
            iterations=iterations,
            substeps=substeps,
            rest_pull=rest_pull,
        )
        # Attach the component to the node so the editor's inspector (and any
        # later register_imported_scene scan) can discover the cloth via
        # node.components instead of having to query the host directly.
        if not any(
            isinstance(c, ClothComponent) and c.cloth_name == component.cloth_name
            for c in node.components
        ):
            node.add_component(component)
        self._register_one(node, component, self._last_imported)
        return self.find_piece(component.cloth_name)

    @staticmethod
    def _resolve_mesh_index(node: Node, override: int | None) -> int | None:
        """Pick a mesh_index for ``node`` — explicit override else first MeshRefComponent."""
        if override is not None:
            return override
        from posecascade.scene.component import MeshRefComponent  # noqa: PLC0415

        for component in node.components:
            if isinstance(component, MeshRefComponent) and component.mesh_indices:
                return int(component.mesh_indices[0])
        _log.warning("node %r has no MeshRefComponent — cannot register cloth", node.name)
        return None

    def add_piece(
        self,
        node: Node,
        piece: ClothPiece,
        mesh_index: int,
        world_to_local: Mat4 | None = None,
    ) -> None:
        """Register an externally-built :class:`ClothPiece` (e.g. from a user script)."""
        binding = ClothBinding(
            node=node,
            mesh_index=mesh_index,
            piece=piece,
            world_to_local=world_to_local if world_to_local is not None else mat4_identity(),
        )
        self._bindings.append(binding)
        self._solver.add_piece(piece)
        self._registered_nodes.add(id(node) ^ hash(piece.name))

    def add_force(self, force: ClothForce) -> None:
        self._solver.add_force(force)

    def add_collider(self, collider: SphereCollider | CapsuleCollider) -> None:
        self._solver.add_collider(collider)

    def find_piece(self, name: str) -> ClothPiece | None:
        return self._solver.find_piece(name)

    def bindings(self) -> tuple[ClothBinding, ...]:
        return tuple(self._bindings)

    def iter_local_state(self) -> Iterable[tuple[ClothBinding, NDArray, NDArray]]:
        """Yield ``(binding, positions_local, normals_local)`` for every active piece.

        The renderer consumes this to refresh dynamic VBOs each frame. Positions
        and normals are returned in the cloth node's LOCAL space — the renderer
        applies the model matrix downstream as for any other mesh.

        ``world_to_local`` is recomputed from the node's CURRENT world matrix
        each frame, not the one captured at registration. Otherwise a parent
        transform applied after registration (declarative root yaw/lean/
        translation, IK on an intermediate joint, …) double-applies in the
        renderer: once via the now-stale local positions, once via the new
        model matrix the renderer reads off the same node.
        """
        for binding in self._bindings:
            if not binding.piece.enabled:
                continue
            if id(binding.piece) in self._gpu_managed_pieces:
                # GPU compute already wrote the position + normal VBOs in
                # ``Renderer.apply_cloth_state`` — no CPU readback needed.
                continue
            current_world = _world_matrix(binding.node)
            try:
                world_to_local = np.linalg.inv(current_world).astype(np.float32, copy=False)
            except np.linalg.LinAlgError:
                world_to_local = binding.world_to_local
            local_positions = _transform_points(binding.piece.positions, world_to_local)
            cached_normals = self._piece_skinned_normals.get(id(binding.piece))
            if cached_normals is not None:
                # Normals are already in world space (LBS'd through the
                # per-vert weighted bone matrix). Rotate them to local
                # using ``world_to_local`` — translation row stays
                # implicit because normals have no translation.
                local_normals = (
                    cached_normals @ world_to_local[:3, :3].T
                ).astype(np.float32, copy=False)
            else:
                local_normals = compute_vertex_normals(
                    local_positions, binding.piece.triangles,
                )
            yield binding, local_positions, local_normals

    def tick(self, dt: float) -> None:
        self._update_anchor_followers()
        self._update_skin_targets()
        self._solver.step(dt)
        self._project_passive_pieces()
        self._apply_floor_clamp()

    def _apply_floor_clamp(self) -> None:
        """Snap any cloth verts that fell below :attr:`floor_y` back onto it.

        Updates ``prev_positions`` in lockstep so the next Verlet step
        doesn't read a teleport-sized velocity spike off the snap. No-op
        when ``floor_y`` is None or no piece has any vert below the plane.
        """
        if self.floor_y is None:
            return
        floor = float(self.floor_y)
        for piece in self._solver.pieces:
            if not piece.enabled:
                continue
            below = piece.positions[:, 1] < floor
            if not np.any(below):
                continue
            piece.positions[below, 1] = floor
            piece.prev_positions[below, 1] = floor

    def register_anchor_follower(self, piece: ClothPiece, bone_node: Node) -> bool:
        """Track ``piece``'s full cloth frame to ``bone_node`` each tick.

        After this call every :meth:`tick` applies the bone's RELATIVE
        transform (since the previous tick) to every vert in
        ``positions`` / ``prev_positions`` / ``rest_positions``, then
        snaps the anchor verts to their precise per-tick world position.
        Cloth velocity (positions - prev_positions) gets rotated through
        the same transform, so the simulation continues consistently in
        the bone's moving frame.

        Returns ``True`` when a follower was registered, ``False`` when
        the piece has no pinned verts (no anchors → nothing to track).
        """
        inv_masses = np.asarray(piece.inverse_masses, dtype=np.float32)
        anchor_indices = np.flatnonzero(inv_masses == 0.0).astype(np.int64, copy=False)
        if anchor_indices.size == 0:
            return False
        bone_world = _world_matrix(bone_node).astype(np.float32, copy=False)
        bone_world_inv = np.linalg.inv(bone_world).astype(np.float32, copy=False)
        anchor_world = np.asarray(piece.positions, dtype=np.float32)[anchor_indices]
        homog = np.column_stack(
            [anchor_world, np.ones((anchor_world.shape[0], 1), dtype=np.float32)],
        )
        anchor_offsets_local = (homog @ bone_world_inv.T)[:, :3].astype(
            np.float32, copy=False,
        )
        self._anchor_followers.append(
            _AnchorFollower(
                piece=piece,
                bone_node=bone_node,
                anchor_indices=anchor_indices,
                anchor_offsets_local=anchor_offsets_local,
                last_bone_world=bone_world.copy(),
            ),
        )
        return True

    def register_skin_target_follower(
        self,
        piece: ClothPiece,
        mesh_node: Node,
    ) -> bool:
        """Drive ``piece.rest_positions`` from the mesh node's skin each tick.

        Reads the bind positions, joint indices, and weights off the mesh
        primitive the cloth was built from, plus the skin's inverse-bind
        matrices off the node's :class:`SkinRefComponent`. Returns ``False``
        if either the node has no skin or the mesh has no skin attributes
        (in which case the caller falls back to the static anchor follower).
        """
        skin = None
        for component in mesh_node.components:
            if isinstance(component, SkinRefComponent) and component.skin is not None:
                skin = component.skin
                break
        if skin is None:
            return False
        imported = self._last_imported
        if imported is None:
            return False
        # Locate the mesh primitive the piece was built from. The binding
        # records its mesh_index; find ours by matching piece identity.
        binding = next((b for b in self._bindings if b.piece is piece), None)
        if binding is None:
            return False
        mesh = imported.meshes[binding.mesh_index]
        if mesh.joints_0 is None or mesh.weights_0 is None:
            return False
        inv_bind = np.asarray(skin.inverse_bind_matrices, dtype=np.float32)
        bind_positions = np.asarray(mesh.positions, dtype=np.float32)
        joints_per_vert = np.asarray(mesh.joints_0, dtype=np.int32)
        weights_per_vert = np.asarray(mesh.weights_0, dtype=np.float32)
        bind_normals = (
            np.asarray(mesh.normals, dtype=np.float32)
            if mesh.normals is not None else None
        )
        self._skin_followers.append(
            _SkinTargetFollower(
                piece=piece,
                skin=skin,
                skin_joints=tuple(skin.joints),
                inverse_bind=inv_bind,
                bind_positions=bind_positions,
                joints_per_vert=joints_per_vert,
                weights_per_vert=weights_per_vert,
                bind_normals=bind_normals,
            ),
        )
        # Cache the dominant joint per vert (highest-weight slot). The
        # filter pass uses this to decide which verts a given collider
        # should be allowed to push — e.g. a hand-sphere skips verts
        # already weighted to the hand chain.
        dominant_slot = np.argmax(weights_per_vert, axis=1)
        dominant_joint = joints_per_vert[
            np.arange(joints_per_vert.shape[0]), dominant_slot,
        ].astype(np.int32, copy=False)
        self._piece_dominant_joint[id(piece)] = dominant_joint
        return True

    def register_collider_bone_filter(
        self,
        collider: object,
        exclude_root: Node,
        skin_joints: tuple[Node, ...] | None = None,
    ) -> None:
        """Record which skin joints a collider should NOT push verts away from.

        Builds the set of joint indices spanning ``exclude_root`` and every
        descendant scene node, intersected with the bound mesh's skin joint
        list. Stored under ``id(collider)`` so the per-frame filter pass can
        skip verts whose dominant bone falls in the excluded set — i.e. the
        collider only pushes "foreign" verts, never verts already weighted
        to itself or to its child bones.

        If ``skin_joints`` is None the most recently registered skin's
        joint tuple is used. No-ops when the host has no skin info yet.
        """
        excluded_nodes = {id(node) for node in exclude_root.traverse()}
        if skin_joints is None:
            if not self._skin_followers:
                return
            skin_joints = self._skin_followers[-1].skin_joints
        excluded_indices: set[int] = set()
        for idx, joint in enumerate(skin_joints):
            if id(joint) in excluded_nodes:
                excluded_indices.add(idx)
        if not excluded_indices:
            return
        existing = self._collider_bone_filters.get(id(collider), frozenset())
        self._collider_bone_filters[id(collider)] = frozenset(
            existing | excluded_indices,
        )

    def _project_passive_pieces(self) -> None:
        """Push passive-skin-deform piece verts out of every registered collider.

        For each piece flagged ``passive_skin_deform``:
          1. Iterate every active solver collider.
          2. Read its ``_collider_bone_filters`` entry (defaulting to an
             empty exclude set — push all verts).
          3. Mask out verts whose ``_piece_dominant_joint`` is in the
             excluded set; these belong to the collider's own bone
             subtree and must not be pushed away from themselves.
          4. Project the eligible verts against the collider in-place.

        Numpy-vectorised per collider; cost on a 30k-vert body mesh with
        ~7 colliders is ~1 ms / frame. Sphere and capsule are the only
        collider kinds handled — extend the per-kind branch when new
        primitives land.
        """
        if not self._solver.colliders:
            return
        for piece in self._solver.pieces:
            if not piece.enabled or not piece.params.passive_skin_deform:
                continue
            if id(piece) in self._gpu_managed_pieces:
                # Collider push runs in the compute shader — skip CPU path.
                continue
            dominant = self._piece_dominant_joint.get(id(piece))
            if dominant is None:
                continue
            piece_cache = self._eligible_cache.setdefault(id(piece), {})
            for collider in self._solver.colliders:
                entry = piece_cache.get(id(collider))
                if entry is None:
                    excluded = self._collider_bone_filters.get(
                        id(collider), frozenset(),
                    )
                    if excluded:
                        excluded_arr = np.fromiter(
                            excluded, dtype=np.int32, count=len(excluded),
                        )
                        eligible = ~np.isin(dominant, excluded_arr)
                    else:
                        eligible = np.ones(dominant.shape[0], dtype=bool)
                    eligible_idx = np.flatnonzero(eligible).astype(
                        np.int64, copy=False,
                    )
                    entry = (eligible, eligible_idx)
                    piece_cache[id(collider)] = entry
                _, eligible_idx = entry
                if eligible_idx.size == 0:
                    continue
                _push_out_collider(piece.positions, eligible_idx, collider)

    def _update_skin_targets(self) -> None:
        """Refresh each piece's ``rest_positions`` from current bone matrices.

        Hot path on passive-skin-deform meshes (30k+ verts per tick). The
        per-frame cost was dominated by two things on the naive version:
        (a) ``_world_matrix(bone_node)`` re-walking each bone's parent
        chain on every call — 354 bones × depth ~8 = ~3000 matrix mults
        per frame; (b) Python-level loop over joints with one matmul per
        joint. Both are fixed below: world matrices are memoised inside
        a per-tick cache (~354 mults, one per bone), the joint matrix
        stack is computed with a single batched matmul, and the per-vert
        homogeneous bind buffer + skinned output buffer are reused
        across ticks instead of reallocated.
        """
        if not self._skin_followers:
            return
        self._last_joint_matrix_cache.clear()
        world_cache: dict[int, NDArray[np.float32]] = {}

        def _cached_world(node: Node) -> NDArray[np.float32]:
            cached = world_cache.get(id(node))
            if cached is not None:
                return cached
            local = node.transform.to_matrix().astype(np.float32, copy=False)
            parent = node.parent
            result = local if parent is None else _cached_world(parent) @ local
            world_cache[id(node)] = result
            return result

        for follower in self._skin_followers:
            num_joints = len(follower.skin_joints)
            # Stack all bone world matrices then batch-multiply by the
            # inverse-bind matrices — a single (J, 4, 4) @ (J, 4, 4) call
            # replaces the per-joint Python loop.
            bone_worlds = np.empty((num_joints, 4, 4), dtype=np.float32)
            for j, bone_node in enumerate(follower.skin_joints):
                bone_worlds[j] = _cached_world(bone_node)
            joint_matrices = np.matmul(bone_worlds, follower.inverse_bind)
            # Stash for the renderer's per-frame bone matrix cache —
            # saves the toon / outline / shadow passes from re-walking
            # this same skin's parent chains. The GPU dispatcher also
            # reads from here, so the matrices must be computed even
            # when the per-vert LBS path is skipped below.
            self._last_joint_matrix_cache[id(follower.skin)] = joint_matrices
            if id(follower.piece) in self._gpu_managed_pieces:
                # Compute shader does the per-vert LBS + normal LBS;
                # skip the ~5–6 ms numpy gather + matmul on a 30 k vert
                # mesh.
                continue

            bind_h = follower._bind_h               # noqa: SLF001 — same module
            combined = follower._combined           # noqa: SLF001
            gather = follower._gather               # noqa: SLF001
            skinned_h = follower._skinned_h         # noqa: SLF001
            num_slots = follower.joints_per_vert.shape[1]
            # Per-slot gather + weighted accumulate into ``combined``,
            # then one (N, 4, 4) @ (N, 4, 1) matmul. einsum
            # (``ns,nsij->nij``) was tried as an alternative but ran
            # ~15 % slower on 30k verts — the per-slot loop with
            # in-place ops hits BLAS's small-matmul fast paths better.
            np.take(
                joint_matrices, follower.joints_per_vert[:, 0], axis=0,
                out=gather,
            )
            np.multiply(
                gather, follower.weights_per_vert[:, 0, None, None],
                out=combined,
            )
            for slot in range(1, num_slots):
                np.take(
                    joint_matrices, follower.joints_per_vert[:, slot], axis=0,
                    out=gather,
                )
                gather *= follower.weights_per_vert[:, slot, None, None]
                combined += gather
            np.matmul(combined, bind_h[:, :, None], out=skinned_h)
            follower.piece.rest_positions[:] = skinned_h[:, :3, 0]
            # LBS the bind-pose normals through the SAME combined matrix
            # (3x3 sub-block only — translation column doesn't apply to
            # normals). Saves the ~3 ms per-frame
            # ``compute_vertex_normals`` recompute in ``iter_local_state``
            # AND gives more accurate normals (proper per-vert rotation
            # vs. recomputed-from-positions area-weighted average).
            if follower.bind_normals is not None:
                skinned_n = follower._skinned_n     # noqa: SLF001
                np.matmul(
                    combined[:, :3, :3],
                    follower.bind_normals[:, :, None],
                    out=skinned_n,
                )
                # Normalise in-place (skinning preserves direction but
                # weighted-blend can shorten the vector through joint
                # collapse on LBS).
                norms = skinned_n[:, :, 0]
                lengths = np.sqrt(np.einsum("ij,ij->i", norms, norms))
                safe = np.maximum(lengths, 1.0e-6, dtype=np.float32)   # noqa: PLR2004
                self._piece_skinned_normals[id(follower.piece)] = (
                    norms / safe[:, None]
                ).astype(np.float32, copy=False)

    def _update_anchor_followers(self) -> None:
        if not self._anchor_followers:
            return
        for follower in self._anchor_followers:
            current_bone_world = _world_matrix(follower.bone_node).astype(
                np.float32, copy=False,
            )
            # Relative transform from previous tick to this tick. Identity
            # when the bone hasn't moved → skip the work in that case.
            last_inv = np.linalg.inv(follower.last_bone_world).astype(
                np.float32, copy=False,
            )
            relative = (current_bone_world @ last_inv).astype(np.float32, copy=False)
            if not _is_identity_matrix(relative):
                for arr in (
                    follower.piece.positions,
                    follower.piece.prev_positions,
                    follower.piece.rest_positions,
                ):
                    _transform_in_place(arr, relative)
            # Snap anchor verts to their precise bone-relative position
            # (recovers any drift accumulated across many transformed ticks).
            offsets = follower.anchor_offsets_local
            homog = np.column_stack(
                [offsets, np.ones((offsets.shape[0], 1), dtype=np.float32)],
            )
            new_world = (homog @ current_bone_world.T)[:, :3].astype(
                np.float32, copy=False,
            )
            follower.piece.positions[follower.anchor_indices] = new_world
            follower.piece.prev_positions[follower.anchor_indices] = new_world
            follower.piece.rest_positions[follower.anchor_indices] = new_world
            follower.last_bone_world = current_bone_world.copy()

    def reset(self) -> None:
        """Drop all bindings, forces, and colliders. Lets a host be re-used across scenes."""
        self._solver = ClothSolver()
        self._bindings.clear()
        self._registered_nodes.clear()
        self._anchor_followers.clear()
        self._skin_followers.clear()
        self._collider_bone_filters.clear()
        self._piece_dominant_joint.clear()
        self._eligible_cache.clear()
        self._last_joint_matrix_cache.clear()
        self._piece_skinned_normals.clear()
        self._gpu_managed_pieces.clear()
        self._gravity_installed = False

    def mark_gpu_managed(self, piece: ClothPiece) -> None:
        """Mark ``piece`` as GPU-dispatched — host bypasses CPU LBS for it.

        Called by the renderer once it has successfully registered the
        piece's SSBOs with the compute-shader dispatcher. The host keeps
        computing joint matrices (the renderer's cache reads them) but
        skips the per-vert LBS, the collider push, and the local-state
        yield for this piece.
        """
        self._gpu_managed_pieces.add(id(piece))

    def is_gpu_managed(self, piece: ClothPiece) -> bool:
        """True when ``piece``'s LBS + push are handled by the GPU dispatcher."""
        return id(piece) in self._gpu_managed_pieces

    def collider_filter_for(self, collider: object) -> frozenset[int]:
        """Return the per-collider excluded-joint set (empty if none registered)."""
        return self._collider_bone_filters.get(id(collider), frozenset())

    def colliders(self) -> tuple[object, ...]:
        """Return the active collider tuple — convenience for the GPU dispatcher path."""
        return tuple(self._solver.colliders)

    def follower_for_piece(self, piece: ClothPiece) -> _SkinTargetFollower | None:
        """Return the skin-target follower bound to ``piece``, or ``None``."""
        for follower in self._skin_followers:
            if follower.piece is piece:
                return follower
        return None

    def joint_matrices_for_skin(self, skin: object) -> NDArray[np.float32] | None:
        """Return the most recently computed joint matrices for ``skin``."""
        return self._last_joint_matrix_cache.get(id(skin))

    def remove_pieces_for_subtree(self, root_node: Node) -> int:
        """Drop every cloth binding whose node sits in ``root_node``'s subtree.

        Used when the editor deletes a subtree — without it, the solver keeps
        running the cloth and the renderer keeps trying to upload its positions
        to a GL mesh that's no longer in the scene. Returns the count removed.
        """
        subtree_ids = {id(n) for n in root_node.traverse()}
        removed_pieces: set[int] = set()
        kept_bindings: list[ClothBinding] = []
        for binding in self._bindings:
            if id(binding.node) in subtree_ids:
                self._registered_nodes.discard(id(binding.node) ^ hash(binding.piece.name))
                removed_pieces.add(id(binding.piece))
            else:
                kept_bindings.append(binding)
        self._bindings = kept_bindings
        self._solver.pieces = [p for p in self._solver.pieces if id(p) not in removed_pieces]
        self._anchor_followers = [
            f for f in self._anchor_followers if id(f.piece) not in removed_pieces
        ]
        self._skin_followers = [
            f for f in self._skin_followers if id(f.piece) not in removed_pieces
        ]
        for piece_id in removed_pieces:
            self._piece_dominant_joint.pop(piece_id, None)
            self._eligible_cache.pop(piece_id, None)
            self._piece_skinned_normals.pop(piece_id, None)
            self._gpu_managed_pieces.discard(piece_id)
        return len(removed_pieces)

    def _register_one(
        self,
        node: Node,
        component: ClothComponent,
        imported: ImportedScene,
    ) -> None:
        key = id(node) ^ hash(component.cloth_name)
        if key in self._registered_nodes:
            return
        if component.mesh_index < 0 or component.mesh_index >= len(imported.meshes):
            _log.warning(
                "cloth %r: mesh_index %d out of range",
                component.cloth_name, component.mesh_index,
            )
            return
        mesh = imported.meshes[component.mesh_index]
        node_world_matrix = _world_matrix(node)
        # For SKINNED meshes the bind-pose rendered position is
        # ``mesh_local`` directly — the inverse_bind matrix in the skin
        # data cancels every parent transform on the way back out, so
        # applying the mesh node's world matrix here would put the cloth
        # 0.27 m below where the skinned mesh actually renders (on
        # March 7th: node scale = 0.73 → skirt top initialised at Y=0.53
        # while the skinned skirt top sits at Y=0.726). Use identity for
        # skinned meshes; the mesh node's matrix only for static meshes.
        has_skin = any(
            isinstance(c, SkinRefComponent) for c in node.components
        )
        init_world_matrix = (
            mat4_identity().astype(np.float32, copy=False)
            if has_skin
            else node_world_matrix
        )
        try:
            world_to_local = np.linalg.inv(node_world_matrix).astype(
                np.float32, copy=False,
            )
        except np.linalg.LinAlgError:
            _log.warning(
                "cloth %r: node world matrix not invertible — skipping",
                component.cloth_name,
            )
            return
        anchor_mask = _build_anchor_mask(
            mesh.positions, mesh.indices, init_world_matrix, component,
        )
        params = ClothParams(
            structural_stiffness=component.structural_stiffness,
            bend_stiffness=component.bend_stiffness,
            linear_damping=component.linear_damping,
            iterations=component.iterations,
            substeps=component.substeps,
            rest_pull=component.rest_pull,
        )
        piece = cloth_from_mesh(
            component.cloth_name or f"cloth_{component.mesh_index}",
            mesh.positions,
            mesh.indices,
            world_matrix=init_world_matrix,
            anchor_mask=anchor_mask,
            params=params,
        )
        binding = ClothBinding(
            node=node,
            mesh_index=component.mesh_index,
            piece=piece,
            world_to_local=world_to_local,
        )
        self._bindings.append(binding)
        self._solver.add_piece(piece)
        self._registered_nodes.add(key)


def _build_anchor_mask(
    local_positions: NDArray,
    indices: NDArray,
    world_matrix: Mat4,
    component: ClothComponent,
) -> NDArray:
    """Pick anchor verts using the world-space coordinate after applying ``world_matrix``.

    Switches between :func:`anchor_by_top_axis` (single global slab) and
    :func:`anchor_by_island_top` (one slab per connected mesh component) based
    on ``component.anchor_mode``. Falls back to the global mode for any unknown
    mode string with a warning so a typo doesn't silently leave the cloth
    completely free-floating. ``anchor_fraction <= 0`` returns an all-false
    mask — used by passive-skin-deform cloth pieces where every vert is
    driven by the skinning each tick and no vert should be pinned.
    """
    if component.anchor_fraction <= 0.0:
        return np.zeros(local_positions.shape[0], dtype=bool)
    world_positions = _transform_points(local_positions, world_matrix)
    if component.anchor_mode == "per_island_top":
        return anchor_by_island_top(
            world_positions,
            indices,
            axis=component.anchor_axis,
            fraction=component.anchor_fraction,
            simulate_top_below=component.simulate_top_below,
        )
    if component.anchor_mode != "top_axis":
        _log.warning(
            "cloth %r: unknown anchor_mode %r — falling back to 'top_axis'",
            component.cloth_name, component.anchor_mode,
        )
    return anchor_by_top_axis(
        world_positions,
        axis=component.anchor_axis,
        fraction=component.anchor_fraction,
    )


def _transform_points(points: NDArray, matrix: Mat4) -> NDArray:
    """Apply a 4x4 affine matrix to ``(N, 3)`` points."""
    n = points.shape[0]
    homog = np.hstack([points, np.ones((n, 1), dtype=np.float32)])
    out = (homog @ matrix.T)[:, :3]
    return out.astype(np.float32, copy=False)


def _world_matrix(node: Node) -> Mat4:
    """Compose ``node``'s world matrix by walking the parent chain."""
    matrix = node.transform.to_matrix()
    parent = node.parent
    while parent is not None:
        matrix = parent.transform.to_matrix() @ matrix
        parent = parent.parent
    return matrix.astype(np.float32, copy=False)


def _is_identity_matrix(m: NDArray, eps: float = 1.0e-6) -> bool:
    """True if ``m`` is within ``eps`` of the 4x4 identity matrix."""
    return bool(np.max(np.abs(m - np.eye(4, dtype=m.dtype))) <= eps)


def _transform_in_place(points: NDArray, matrix: Mat4) -> None:
    """Apply a 4x4 affine matrix to ``(N, 3)`` points, writing back in place."""
    n = points.shape[0]
    if n == 0:
        return
    homog = np.hstack([points, np.ones((n, 1), dtype=points.dtype)])
    points[:] = (homog @ matrix.T)[:, :3]


def _push_out_collider(
    positions: NDArray[np.float32],
    eligible_idx: NDArray[np.int64],
    collider: object,
) -> None:
    """Project the ``eligible_idx`` rows of ``positions`` outside ``collider``, in place.

    Standalone projection for the passive-skin-deform path — doesn't need
    the cloth solver's bin / velocity machinery because the verts are
    snapped fresh from the LBS-skinned pose each tick, so prev_positions
    equals current positions and there's no velocity to preserve.
    ``eligible_idx`` is a precomputed flat int array of vertex indices to
    consider — caching it avoids the per-frame ``np.isin`` on the full
    vertex set. Supports :class:`SphereCollider` and :class:`CapsuleCollider`.
    """
    if eligible_idx.size == 0:
        return
    if isinstance(collider, SphereCollider):
        _push_sphere(positions, eligible_idx, collider)
        return
    if isinstance(collider, CapsuleCollider):
        _push_capsule(positions, eligible_idx, collider)


def _aabb_filter(
    pts: NDArray[np.float32], lo: NDArray[np.float32], hi: NDArray[np.float32],
) -> NDArray[np.bool_]:
    """Per-vertex bool mask: ``True`` where ``pts`` falls inside the AABB."""
    return (
        (pts[:, 0] > lo[0]) & (pts[:, 0] < hi[0])
        & (pts[:, 1] > lo[1]) & (pts[:, 1] < hi[1])
        & (pts[:, 2] > lo[2]) & (pts[:, 2] < hi[2])
    )


def _safe_normalize(
    delta: NDArray[np.float32], dist_sq: NDArray[np.float32],
    fallback: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return ``(unit_norms, distances)`` with NaN-safe handling for zero distance."""
    d = np.sqrt(dist_sq)
    safe = np.where(d > 1.0e-6, d, np.float32(1.0))                       # noqa: PLR2004
    norms = delta / safe[:, None]
    zero_mask = d <= 1.0e-6                                               # noqa: PLR2004
    if np.any(zero_mask):
        norms[zero_mask] = fallback
    return norms, d


def _push_sphere(
    positions: NDArray[np.float32],
    eligible_idx: NDArray[np.int64],
    collider: SphereCollider,
) -> None:
    """Project ``eligible_idx`` rows of ``positions`` outside ``collider``."""
    center = np.asarray(collider.center, dtype=np.float32)
    r = float(collider.radius + collider.skin_offset)
    pts_all = positions[eligible_idx]
    in_aabb = _aabb_filter(pts_all, center - r, center + r)
    if not np.any(in_aabb):
        return
    near_idx = eligible_idx[in_aabb]
    delta = pts_all[in_aabb] - center
    dist_sq = np.einsum("ij,ij->i", delta, delta)
    inside = dist_sq < (r * r)
    if not np.any(inside):
        return
    norms, _ = _safe_normalize(
        delta[inside], dist_sq[inside],
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    positions[near_idx[inside]] = center + norms * r


def _push_capsule(
    positions: NDArray[np.float32],
    eligible_idx: NDArray[np.int64],
    collider: CapsuleCollider,
) -> None:
    """Project ``eligible_idx`` rows of ``positions`` outside ``collider``."""
    a = np.asarray(collider.a, dtype=np.float32)
    b = np.asarray(collider.b, dtype=np.float32)
    r = float(collider.radius + collider.skin_offset)
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq <= 1.0e-12:                                              # noqa: PLR2004
        _push_sphere(
            positions, eligible_idx,
            SphereCollider(
                center=collider.a, radius=collider.radius,
                skin_offset=collider.skin_offset,
            ),
        )
        return
    pts_all = positions[eligible_idx]
    in_aabb = _aabb_filter(pts_all, np.minimum(a, b) - r, np.maximum(a, b) + r)
    if not np.any(in_aabb):
        return
    near_idx = eligible_idx[in_aabb]
    pts = pts_all[in_aabb]
    t = np.clip(np.einsum("ij,j->i", pts - a, ab) / ab_len_sq, 0.0, 1.0)
    closest = a + t[:, None] * ab
    delta = pts - closest
    dist_sq = np.einsum("ij,ij->i", delta, delta)
    inside = dist_sq < (r * r)
    if not np.any(inside):
        return
    # Fallback direction for degenerate (vert exactly on axis) cases.
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(ab[1])) > 0.99:                                          # noqa: PLR2004
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    perp = np.cross(ab / np.sqrt(ab_len_sq), up).astype(np.float32)
    perp_norm = perp / max(float(np.linalg.norm(perp)), 1.0e-6)           # noqa: PLR2004
    norms, _ = _safe_normalize(delta[inside], dist_sq[inside], perp_norm)
    positions[near_idx[inside]] = closest[inside] + norms * r


# Re-export so callers can register cloth pieces / colliders / forces without
# touching the cloth module directly.
__all__ = [
    "CapsuleCollider",
    "ClothBinding",
    "ClothForce",
    "ClothGravity",
    "ClothHost",
    "ClothPiece",
    "SphereCollider",
]
