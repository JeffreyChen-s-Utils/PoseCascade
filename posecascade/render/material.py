"""Engine-side MMD-style material description.

This module is the shared contract between the PMX/PMD importer (which
fills in the per-material values) and the renderer (which binds them as
uniforms / textures during the toon pass). Keeping it in
``posecascade/render`` rather than under ``importers/pmx`` keeps the
renderer free of any importer-side imports — the layering rule from
``CLAUDE.md`` applies.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class SphereMode(IntEnum):
    """How the sphere texture composites with the diffuse term."""

    DISABLED = 0
    MULTIPLY = 1   # final.rgb *= sphere.rgb
    ADD = 2        # final.rgb += sphere.rgb
    SUB_TEXTURE = 3  # treat sphere ref as a sub-texture overlay (rare; not yet rendered)


# ----- material flag bits — mirror PMX semantics so the importer can pass
# the raw ``flags`` byte through unchanged. The renderer only looks at the
# subset it can act on; everything else is forwarded to later phases.
MAT_FLAG_DOUBLE_SIDED = 1 << 0
MAT_FLAG_GROUND_SHADOW = 1 << 1
MAT_FLAG_CAST_SHADOW = 1 << 2
MAT_FLAG_RECEIVE_SHADOW = 1 << 3
MAT_FLAG_HAS_EDGE = 1 << 4


@dataclass(frozen=True)
class MMDMaterial:
    """Per-mesh MMD shading parameters.

    A :class:`~posecascade.assets.types.Mesh` whose ``mmd_material`` is not
    ``None`` is rendered through the toon pass; otherwise the renderer
    uses its existing forward-PBR/Lambert path. Texture indices reference
    the owning :class:`~posecascade.assets.types.ImportedScene.textures`
    tuple (``None`` means the renderer's white fallback / a no-op).
    """

    diffuse: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    specular: tuple[float, float, float] = (0.0, 0.0, 0.0)
    specular_power: float = 0.0
    ambient: tuple[float, float, float] = (0.5, 0.5, 0.5)
    edge_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    edge_size: float = 1.0
    sphere_texture_index: int | None = None
    sphere_mode: SphereMode = SphereMode.DISABLED
    toon_texture_index: int | None = None
    flags: int = 0

    @property
    def is_double_sided(self) -> bool:
        return bool(self.flags & MAT_FLAG_DOUBLE_SIDED)

    @property
    def has_edge(self) -> bool:
        return bool(self.flags & MAT_FLAG_HAS_EDGE)
