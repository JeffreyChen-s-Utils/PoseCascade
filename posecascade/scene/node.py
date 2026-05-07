"""Scene-graph node — Composite of a transform and a list of components."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from posecascade.errors import SceneError
from posecascade.scene.component import Component
from posecascade.scene.transform import Transform


@dataclass
class Node:
    """A named, transformable point in the scene graph with optional components."""

    name: str = ""
    transform: Transform = field(default_factory=Transform)
    components: list[Component] = field(default_factory=list)
    children: list[Node] = field(default_factory=list)
    parent: Node | None = field(default=None, repr=False, compare=False)

    def add_child(self, child: Node) -> None:
        """Attach ``child`` under this node. Raises if ``child`` is already parented."""
        if child is self:
            raise SceneError("a node cannot be its own child")
        if child.parent is not None:
            raise SceneError(f"node {child.name!r} already has a parent")
        if self._would_create_cycle(child):
            raise SceneError(f"adding {child.name!r} would create a cycle")
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: Node) -> None:
        """Detach ``child`` from this node. No-op if it isn't a child."""
        try:
            self.children.remove(child)
        except ValueError as err:
            raise SceneError(f"{child.name!r} is not a child of {self.name!r}") from err
        child.parent = None

    def add_component(self, component: Component) -> None:
        """Attach a component to this node."""
        self.components.append(component)

    def find_first(self, name: str) -> Node | None:
        """Depth-first search for a descendant (or self) with the given name."""
        for node in self.traverse():
            if node.name == name:
                return node
        return None

    def traverse(self) -> Iterator[Node]:
        """Yield this node and all descendants in pre-order."""
        yield self
        for child in self.children:
            yield from child.traverse()

    def _would_create_cycle(self, candidate: Node) -> bool:
        """Return True if ``candidate`` is an ancestor of this node."""
        current: Node | None = self
        while current is not None:
            if current is candidate:
                return True
            current = current.parent
        return False
