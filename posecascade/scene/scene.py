"""Top-level scene container."""
from __future__ import annotations

from dataclasses import dataclass, field

from posecascade.scene.node import Node


@dataclass
class Scene:
    """A scene owns a root :class:`Node` and global ambient parameters."""

    name: str = "scene"
    root: Node = field(default_factory=lambda: Node(name="root"))

    def find(self, node_name: str) -> Node | None:
        """Convenience: depth-first search from the root."""
        return self.root.find_first(node_name)
