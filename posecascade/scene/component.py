"""Base :class:`Component` for scene-graph composition.

Nodes hold a list of components. Renderers, animators, and scripts read
components by type rather than by class hierarchy, so behaviour can be
recombined without inheritance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from posecascade.scene.node import Node


@dataclass
class Component:
    """Marker base for scene-graph components.

    Subclass to attach typed data (``MeshRefComponent``, ``LightComponent``,
    ``ScriptComponent``). Components are plain dataclasses; they MUST NOT
    contain GL resources — those live in the asset cache, keyed by id.
    """

    enabled: bool = True


@dataclass
class MeshRefComponent(Component):
    """References meshes in the owning :class:`~posecascade.assets.types.ImportedScene`.

    A glTF node with a multi-primitive mesh maps to several entries in our
    flat ``ImportedScene.meshes`` tuple; this component records every primitive
    index that should render under the node's world transform.
    """

    mesh_indices: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class SkinRefComponent(Component):
    """Marks the owning node's meshes as skinned, pointing at a :class:`Skin`.

    The renderer uses the referenced :class:`~posecascade.assets.types.Skin`
    to compute per-frame bone matrices (joint world × inverse bind) and
    upload them to the skinning vertex shader. ``skin`` is typed as ``object``
    to keep this module free of an import cycle with :mod:`posecascade.assets`.
    """

    skin: object | None = None


@dataclass
class SpringChainComponent(Component):
    """Marks the owning node as the anchor of a spring-physics chain.

    Stores the ordered ``joints`` (root → tip) plus tuned per-chain physics
    parameters as plain floats. The animation layer (PhysicsHost / SpringSimulator)
    consumes this component to build a :class:`SpringChain`. The component itself
    has no animation imports — keeping :mod:`posecascade.scene` free of dependencies
    on :mod:`posecascade.animation`.
    """

    chain_name: str = ""
    joints: tuple[Node, ...] = field(default_factory=tuple)
    stiffness: float = 8.0
    damping: float = 1.5
    inertia: float = 1.0


@dataclass
class ClothComponent(Component):
    """Marks the owning node's mesh primitive as a PBD-simulated cloth.

    ``mesh_index`` indexes into the owning :class:`~posecascade.assets.types.ImportedScene`
    ``meshes`` tuple — the renderer needs this to find the GPU buffers to stream
    new positions into each frame. ``anchor_axis`` / ``anchor_fraction`` pick a
    band of vertices to pin (default: top 15% of the bounding box along world Y),
    which keeps the cloth attached to the body where the seam lives.

    Stores tuning parameters as plain floats so this module stays free of
    :mod:`posecascade.animation` imports.
    """

    cloth_name: str = ""
    mesh_index: int = -1
    anchor_axis: int = 1               # 0=X, 1=Y, 2=Z; 1 (Y-up) suits world-up gravity scenes
    anchor_fraction: float = 0.15      # top 15% of the bbox along anchor_axis ⇒ pinned
    # ``anchor_mode``:
    #   "top_axis"  — anchor the top fraction of the WHOLE mesh's bbox.
    #   "per_island_top" — anchor the top fraction of EACH connected mesh island
    #                      separately (use this when one mesh contains several
    #                      visually disconnected pieces — e.g. sleeves whose
    #                      shoulder cap and elbow flap are different islands).
    anchor_mode: str = "top_axis"
    structural_stiffness: float = 0.85
    bend_stiffness: float = 0.10
    linear_damping: float = 0.985
    iterations: int = 8
    substeps: int = 2
    rest_pull: float = 4.0
    # Used only with ``anchor_mode="per_island_top"``. If set, any mesh island
    # whose maximum coord along ``anchor_axis`` exceeds this value is fully
    # anchored — keeps the upper-body decorations on a multi-piece sleeve mesh
    # rigid while only the lower hanging flaps actually swing.
    simulate_top_below: float | None = None
