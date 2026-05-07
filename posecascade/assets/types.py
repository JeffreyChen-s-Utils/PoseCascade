"""CPU-side asset dataclasses.

These are the format-independent shapes importers produce and the renderer
consumes. They are immutable after creation; mutating an asset requires
producing a new one and bumping the version counter on the cache entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from posecascade.render.material import MMDMaterial

if TYPE_CHECKING:
    from posecascade.animation.bone_resolver import BoneResolverRules
    from posecascade.animation.display_frames import DisplayFrameGroup
    from posecascade.animation.ik import IkChain
    from posecascade.animation.morph import MorphAsset
    from posecascade.physics.types import PhysicsScene


def _empty_physics_scene() -> PhysicsScene:
    """Lazy default for :attr:`ImportedScene.physics_scene`.

    Mirrors the deferred-import pattern used by :func:`_empty_morph_asset`
    — the physics module imports types from animation at runtime.
    """
    from posecascade.physics.types import PhysicsScene  # noqa: PLC0415
    return PhysicsScene()


def _empty_bone_resolver_rules() -> BoneResolverRules:
    """Lazy default for :attr:`ImportedScene.bone_resolver_rules`.

    Same circular-import dance as :func:`_empty_morph_asset` — defer the
    ``posecascade.animation`` import until first construction.
    """
    from posecascade.animation.bone_resolver import BoneResolverRules  # noqa: PLC0415
    return BoneResolverRules()


def _empty_morph_asset() -> MorphAsset:
    """Lazy default for :attr:`ImportedScene.morphs`.

    Defining the factory inside this module would force an import of
    :mod:`posecascade.animation.morph` at module-load time — which in turn
    pulls in ``posecascade.animation/__init__.py`` and creates a circular
    dependency through ``cloth_host``'s reverse import of this module.
    The lazy lookup defers the import to the first construction.
    """
    from posecascade.animation.morph import MorphAsset  # noqa: PLC0415
    return MorphAsset()


@dataclass(frozen=True)
class Mesh:
    """Vertex / index arrays plus optional skinning channels, tint, and texture link."""

    name: str
    positions: NDArray[np.float32]              # (N, 3)
    indices: NDArray[np.uint32]                 # (M,)
    normals: NDArray[np.float32] | None = None  # (N, 3)
    tangents: NDArray[np.float32] | None = None # (N, 4) — bitangent sign in w
    texcoords_0: NDArray[np.float32] | None = None  # (N, 2)
    joints_0: NDArray[np.uint16] | None = None       # (N, 4)
    weights_0: NDArray[np.float32] | None = None     # (N, 4)
    base_color: tuple[float, float, float, float] | None = None  # RGBA tint, 0–1
    # Index into the owning ImportedScene.textures of the base-colour map, or
    # None if the mesh has no albedo texture (renderer falls back to white).
    base_color_texture_index: int | None = None
    # When non-None, the renderer routes this mesh through the MMD toon pass
    # (toon ramp + sphere texture + inverted-hull outline). Non-MMD imports
    # leave this as None and continue to use the existing forward path.
    mmd_material: MMDMaterial | None = None


@dataclass(frozen=True)
class Texture:
    """Decoded texture pixel data plus sampler hints."""

    name: str
    pixels: NDArray[np.uint8]                  # (H, W, C)
    srgb: bool = False
    wrap_s: str = "repeat"
    wrap_t: str = "repeat"
    mag_filter: str = "linear"
    min_filter: str = "linear_mipmap_linear"


@dataclass(frozen=True)
class Skin:
    """A skeleton: ordered joint nodes plus their inverse-bind matrices.

    ``joints`` are :class:`~posecascade.scene.node.Node` references — typed
    as ``object`` here to avoid an import cycle with :mod:`posecascade.scene`.
    Each :class:`Skin` is created by the glTF importer and held in
    :attr:`ImportedScene.skins`; nodes that render as skinned geometry
    carry a :class:`~posecascade.scene.component.SkinRefComponent` pointing
    at one of these.
    """

    name: str
    joints: tuple[object, ...]
    inverse_bind_matrices: NDArray[np.float32]  # (J, 4, 4) row-major


@dataclass(frozen=True)
class ImportedScene:
    """The bundle an importer hands back: meshes, textures, skins, and a scene graph."""

    meshes: tuple[Mesh, ...] = field(default_factory=tuple)
    textures: tuple[Texture, ...] = field(default_factory=tuple)
    skins: tuple[Skin, ...] = field(default_factory=tuple)
    # The importer creates plain :class:`~posecascade.scene.scene.Scene` instances —
    # ``object`` here keeps this module free of a scene import cycle.
    scene: object | None = None
    # PMX/PMD importers populate this with the model's morph definitions; non-MMD
    # importers leave it as the empty default. The animation player consumes it
    # through :mod:`posecascade.animation.morph_accumulator`.
    morphs: MorphAsset = field(default_factory=_empty_morph_asset)
    # PMX-only IK definitions. The animation player reads these and runs a
    # CCD pass each frame; non-MMD imports leave them empty.
    ik_chains: tuple[IkChain, ...] = field(default_factory=tuple)
    # PMX-only bone append (付与) and fixed-axis rules. The animation
    # player runs them after the IK pass; non-MMD imports leave them empty.
    bone_resolver_rules: BoneResolverRules = field(
        default_factory=_empty_bone_resolver_rules,
    )
    # PMX-only rigid bodies + 6DOF spring joints. The animation player
    # spins up a :class:`PhysicsWorld` when this scene is non-empty.
    physics_scene: PhysicsScene = field(default_factory=_empty_physics_scene)
    # PMX-only display-frame panels (the "face / body / arms" groups
    # used by the timeline editor for track grouping). Non-PMX imports
    # leave this empty; the editor falls back to flat per-track listing.
    display_frame_groups: tuple[DisplayFrameGroup, ...] = field(default_factory=tuple)
