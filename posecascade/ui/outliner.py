"""Scene-graph outliner — left dock that mirrors the active :class:`Scene` as a tree.

Built on ``QTreeWidget`` so PySide6 handles selection, highlighting, and key
navigation without a custom model. The user-facing wiring (selection → inspector,
selection → renderer pick) is done by :class:`MainWindow`; this widget just
publishes ``node_selected`` once the user clicks a row.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
)

from posecascade.scene.node import Node
from posecascade.scene.scene import Scene


class OutlinerDock(QDockWidget):
    """Left dock showing the scene hierarchy. Emits ``node_selected`` on click."""

    node_selected = Signal(object)  # Node — typed as object for Qt's metatype tolerance
    # Emitted with the deleted Node so MainWindow can clean up associated cloth /
    # spring chain bindings before the Node is dropped from the scene tree.
    node_deleted = Signal(object)

    def __init__(self, parent: object = None) -> None:
        super().__init__("Outliner", parent)  # type: ignore[arg-type]
        self.setObjectName("OutlinerDock")
        self._scene: Scene | None = None
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setToolTip(
            "Scene hierarchy. Click a node to inspect / edit it on the right. "
            "Right-click for actions. Press Delete to remove the selected "
            "node and its descendants.",
        )
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        # Right-click menu for delete.
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        # Delete key as a shortcut on the tree widget. QAction with the tree as
        # parent so the shortcut only fires while the tree has focus.
        self._delete_action = QAction("Delete Node", self._tree)
        self._delete_action.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        self._delete_action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        self._delete_action.triggered.connect(self._delete_selected)
        self._tree.addAction(self._delete_action)
        # We need to know which Node a row represents but storing a reference on
        # QTreeWidgetItem via setData() pickles through Qt — keep our own dict
        # keyed by id(item) so the Node itself is held by the dict (Python ref).
        self._node_for_item: dict[int, Node] = {}
        self.setWidget(self._tree)

    def set_scene(self, scene: Scene | None) -> None:
        """Repopulate the tree from ``scene``. Call when the active scene changes."""
        self._scene = scene
        self._tree.clear()
        self._node_for_item.clear()
        if scene is None:
            return
        self._add_node(scene.root, parent_item=None)
        self._tree.expandToDepth(2)

    def select_node(self, node: Node | None) -> None:
        """Programmatically highlight the row representing ``node``."""
        if node is None:
            self._tree.clearSelection()
            return
        for item_id, candidate in self._node_for_item.items():
            if candidate is node:
                item = self._find_item(item_id)
                if item is not None:
                    self._tree.setCurrentItem(item)
                return

    def _add_node(self, node: Node, parent_item: QTreeWidgetItem | None) -> None:
        label = node.name or "<unnamed>"
        if node.components:
            label = f"{label}  ({len(node.components)})"
        item = QTreeWidgetItem([label])
        if parent_item is None:
            self._tree.addTopLevelItem(item)
        else:
            parent_item.addChild(item)
        self._node_for_item[id(item)] = node
        for child in node.children:
            self._add_node(child, item)

    def _find_item(self, item_id: int) -> QTreeWidgetItem | None:
        # QTreeWidget has no global lookup by id, so iterate. Trees are small
        # enough that O(N) is fine; this only fires on programmatic selection.
        iterator = QTreeWidgetItemIterator(self._tree)
        while iterator.value() is not None:
            if id(iterator.value()) == item_id:
                return iterator.value()
            iterator += 1
        return None

    def _on_selection_changed(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            self.node_selected.emit(None)
            return
        node = self._node_for_item.get(id(items[0]))
        self.node_selected.emit(node)

    def _on_context_menu(self, pos: object) -> None:
        item = self._tree.itemAt(pos)  # type: ignore[arg-type]
        if item is None:
            return
        node = self._node_for_item.get(id(item))
        if node is None or node.parent is None:
            # Root node — no delete (would orphan the whole scene).
            return
        menu = QMenu(self._tree)
        action = menu.addAction("Delete")
        action.triggered.connect(self._delete_selected)
        menu.exec(self._tree.mapToGlobal(pos))  # type: ignore[arg-type]

    def _delete_selected(self) -> None:
        """Detach every selected node from its parent and emit ``node_deleted`` for each."""
        items = list(self._tree.selectedItems())
        deleted: list[Node] = []
        for item in items:
            node = self._node_for_item.get(id(item))
            if node is None or node.parent is None:
                continue
            node.parent.remove_child(node)
            deleted.append(node)
        for node in deleted:
            self.node_deleted.emit(node)
        if deleted and self._scene is not None:
            # Re-render the tree from the now-mutated scene. Cheap because
            # node counts are small in typical editor sessions.
            self.set_scene(self._scene)
