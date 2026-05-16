"""Top-level scene container."""
from __future__ import annotations

from dataclasses import dataclass, field

from posecascade.scene.node import Node


@dataclass
class Scene:
    """A scene owns a root :class:`Node` and global ambient parameters.

    The ``bone_aliases`` map lets user scripts target rig joints by
    canonical anatomical name (``"head"``, ``"upper_arm_L"``, …) regardless
    of which naming convention the asset shipped with (VRoid ``J_Bip_*``,
    MMD Japanese ``頭``, March 7th FBX ``Head_M_055``). Populated by
    :func:`posecascade.animation.bone_aliasing.detect_humanoid_aliases`
    when the importer runs; :meth:`find` consults it before falling back
    to a literal name search so existing ``scene.find("Head_M_055")``
    calls continue to work.
    """

    name: str = "scene"
    root: Node = field(default_factory=lambda: Node(name="root"))
    bone_aliases: dict[str, Node] = field(default_factory=dict)

    def find(self, node_name: str) -> Node | None:
        """Convenience: depth-first search from the root.

        Aliases take precedence — a script asking for ``"head"`` gets
        the rig's canonical head bone (e.g. ``Head_M_055`` on March 7th)
        rather than a missing node named literally ``head``. If
        ``node_name`` isn't an alias, falls back to a depth-first name
        search from the scene root so any explicit rig bone or scene
        handle (``"Sketchfab_model"``, ``"J_Bip_C_Hips"``) still resolves.
        """
        aliased = self.bone_aliases.get(node_name)
        if aliased is not None:
            return aliased
        return self.root.find_first(node_name)
