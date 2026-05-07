"""Engine-side MMD morph types.

A :class:`MorphAsset` is what the importer hands back: every PMX morph
expressed as a typed dataclass plus a ``by_name`` lookup the player uses
to resolve VMD morph keyframes against. Group / Flip morphs hold child
references as indices into ``by_index`` so the accumulator can recurse
without a separate name lookup.

Phase 4 acts on Vertex / Bone / UV (channel 0) / Material / Group / Flip
morphs. The remaining types (UV1..4, Impulse) are still parsed and stored
so the player can ignore them gracefully — and so later phases can pick
them up without re-touching the importer adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class MorphType(IntEnum):
    """Mirrors PMX morph type bytes verbatim."""

    GROUP = 0
    VERTEX = 1
    BONE = 2
    UV = 3
    UV1 = 4
    UV2 = 5
    UV3 = 6
    UV4 = 7
    MATERIAL = 8
    FLIP = 9         # PMX 2.1
    IMPULSE = 10     # PMX 2.1


class MorphPanel(IntEnum):
    """MMD's UI panel grouping. Engine-side just metadata for the inspector."""

    HIDDEN = 0
    EYE = 1
    LIP = 2
    BROW = 3
    OTHER = 4


class MaterialMorphOp(IntEnum):
    MULTIPLY = 0
    ADD = 1


# ----- per-type payloads ------------------------------------------------
@dataclass(frozen=True)
class VertexMorphOffset:
    vertex_index: int
    offset: tuple[float, float, float]


@dataclass(frozen=True)
class BoneMorphOffset:
    bone_index: int
    translation: tuple[float, float, float]
    # Quat xyzw — identity ``(0, 0, 0, 1)`` when the bone is only translated.
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True)
class UvMorphOffset:
    vertex_index: int
    offset: tuple[float, float, float, float]   # vec4 — only xy used for UV0; full vec4 for UV1..4


@dataclass(frozen=True)
class MaterialMorphTarget:
    """A single material affected by a material morph.

    ``material_index = -1`` is the PMX wildcard meaning "apply to every
    material"; the accumulator preserves that semantics by fanning the
    target out at apply time.
    """

    material_index: int
    op: MaterialMorphOp
    diffuse: tuple[float, float, float, float]
    specular: tuple[float, float, float]
    specular_power: float
    ambient: tuple[float, float, float]
    edge_color: tuple[float, float, float, float]
    edge_size: float
    texture_coef: tuple[float, float, float, float]
    sphere_coef: tuple[float, float, float, float]
    toon_coef: tuple[float, float, float, float]


@dataclass(frozen=True)
class GroupMorphChild:
    """A child of a group / flip morph: an index into ``MorphAsset.by_index``."""

    morph_index: int
    weight: float


# ----- top-level morph dataclasses --------------------------------------
@dataclass(frozen=True)
class _MorphBase:
    """Common metadata fields shared across every concrete morph type."""

    name: str
    panel: MorphPanel = MorphPanel.OTHER


@dataclass(frozen=True)
class VertexMorph(_MorphBase):
    offsets: tuple[VertexMorphOffset, ...] = ()


@dataclass(frozen=True)
class BoneMorph(_MorphBase):
    offsets: tuple[BoneMorphOffset, ...] = ()


@dataclass(frozen=True)
class UvMorph(_MorphBase):
    channel: int = 0   # 0 → primary UV (texcoords_0); 1..4 → additional UVs
    offsets: tuple[UvMorphOffset, ...] = ()


@dataclass(frozen=True)
class MaterialMorph(_MorphBase):
    targets: tuple[MaterialMorphTarget, ...] = ()


@dataclass(frozen=True)
class GroupMorph(_MorphBase):
    children: tuple[GroupMorphChild, ...] = ()


@dataclass(frozen=True)
class FlipMorph(_MorphBase):
    children: tuple[GroupMorphChild, ...] = ()


@dataclass(frozen=True)
class ImpulseMorphOffset:
    rigid_body_index: int
    is_local: bool
    velocity: tuple[float, float, float]
    torque: tuple[float, float, float]


@dataclass(frozen=True)
class ImpulseMorph(_MorphBase):
    """Stored verbatim — the physics phase will read it. Phase 4 ignores it."""

    offsets: tuple[ImpulseMorphOffset, ...] = ()


Morph = (
    VertexMorph
    | BoneMorph
    | UvMorph
    | MaterialMorph
    | GroupMorph
    | FlipMorph
    | ImpulseMorph
)


@dataclass(frozen=True)
class MorphAsset:
    """All morphs belonging to one imported model.

    ``by_index`` is the PMX ordering — Group / Flip child references point
    here. ``by_name`` is the lookup the player uses to resolve VMD morph
    keyframes against; later-defined morphs win on collision (matches
    MMD's "the editor can't have two morphs with the same name; if you
    do, the last wins" behaviour).
    """

    by_index: tuple[Morph, ...] = field(default_factory=tuple)
    by_name: dict[str, int] = field(default_factory=dict)

    def lookup(self, name: str) -> Morph | None:
        index = self.by_name.get(name)
        if index is None:
            return None
        return self.by_index[index]


def build_morph_asset(morphs: tuple[Morph, ...]) -> MorphAsset:
    """Wrap a flat morph tuple into a :class:`MorphAsset` with a name lookup."""
    by_name: dict[str, int] = {}
    for index, morph in enumerate(morphs):
        by_name[morph.name] = index
    return MorphAsset(by_index=morphs, by_name=by_name)
