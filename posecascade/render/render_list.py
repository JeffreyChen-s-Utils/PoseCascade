"""Immutable render list produced by the cull/sort pass.

The renderer consumes :class:`RenderList` items in order; it never reads scene
graph internals directly. This decouples scene mutation (any thread, gated by
the command queue) from rendering (always the GL-owning thread).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from posecascade.utils.math3d import Mat4


@dataclass(frozen=True)
class RenderItem:
    """One draw call: a mesh id, material id, and world matrix."""

    mesh_id: int
    material_id: int
    world_matrix: Mat4
    skin_id: int | None = None


@dataclass(frozen=True)
class RenderList:
    """A sorted, immutable batch of :class:`RenderItem` for a single frame."""

    items: tuple[RenderItem, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.items)
