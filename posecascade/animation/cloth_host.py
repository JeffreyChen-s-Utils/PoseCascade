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
from posecascade.scene.component import ClothComponent
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
class ClothHost:
    """Owns a :class:`ClothSolver` and binds it to scene nodes/meshes."""

    _solver: ClothSolver = field(default_factory=ClothSolver)
    _bindings: list[ClothBinding] = field(default_factory=list)
    _registered_nodes: set[int] = field(default_factory=set)
    _gravity_installed: bool = False
    # Track the most recent imported scene so user scripts (via physics_lite) can
    # register cloth pieces by node reference without the script needing direct
    # access to the importer's flat mesh tuple.
    _last_imported: ImportedScene | None = None

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
            current_world = _world_matrix(binding.node)
            try:
                world_to_local = np.linalg.inv(current_world).astype(np.float32, copy=False)
            except np.linalg.LinAlgError:
                world_to_local = binding.world_to_local
            local_positions = _transform_points(binding.piece.positions, world_to_local)
            local_normals = compute_vertex_normals(local_positions, binding.piece.triangles)
            yield binding, local_positions, local_normals

    def tick(self, dt: float) -> None:
        self._solver.step(dt)

    def reset(self) -> None:
        """Drop all bindings, forces, and colliders. Lets a host be re-used across scenes."""
        self._solver = ClothSolver()
        self._bindings.clear()
        self._registered_nodes.clear()
        self._gravity_installed = False

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
        world_matrix = _world_matrix(node)
        try:
            world_to_local = np.linalg.inv(world_matrix).astype(np.float32, copy=False)
        except np.linalg.LinAlgError:
            _log.warning(
                "cloth %r: node world matrix not invertible — skipping",
                component.cloth_name,
            )
            return
        anchor_mask = _build_anchor_mask(mesh.positions, mesh.indices, world_matrix, component)
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
            world_matrix=world_matrix,
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
    completely free-floating.
    """
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
