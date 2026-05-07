"""Qt smoke tests for :class:`posecascade.ui.outliner.OutlinerDock`.

Uses the shared ``qapp`` fixture from ``tests/conftest.py`` so a singleton
``QApplication`` is constructed once per test session.
"""
from __future__ import annotations

from typing import Any

from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.ui.outliner import OutlinerDock


def _build_scene() -> Scene:
    root = Node(name="root")
    child_a = Node(name="child_a")
    child_b = Node(name="child_b")
    grand = Node(name="grand")
    root.add_child(child_a)
    root.add_child(child_b)
    child_a.add_child(grand)
    return Scene(root=root)


def test_outliner_set_scene_populates_tree(qapp: Any) -> None:
    del qapp  # fixture only ensures QApplication exists
    dock = OutlinerDock()
    dock.set_scene(_build_scene())
    assert dock.widget().topLevelItemCount() == 1
    root_item = dock.widget().topLevelItem(0)
    assert root_item.text(0).startswith("root")
    assert root_item.childCount() == 2


def test_outliner_set_scene_clears_when_set_again(qapp: Any) -> None:
    del qapp
    dock = OutlinerDock()
    dock.set_scene(_build_scene())
    dock.set_scene(None)
    assert dock.widget().topLevelItemCount() == 0


def test_outliner_emits_signal_on_selection(qapp: Any) -> None:
    del qapp
    dock = OutlinerDock()
    dock.set_scene(_build_scene())
    received: list[Node | None] = []
    dock.node_selected.connect(received.append)

    # Programmatically select the second child of root.
    root_item = dock.widget().topLevelItem(0)
    second_child_item = root_item.child(1)
    dock.widget().setCurrentItem(second_child_item)

    assert received, "no node_selected signal fired"
    assert received[-1] is not None
    assert received[-1].name == "child_b"


def test_outliner_select_node_highlights_correct_row(qapp: Any) -> None:
    del qapp
    dock = OutlinerDock()
    scene = _build_scene()
    dock.set_scene(scene)
    grand = scene.root.children[0].children[0]

    dock.select_node(grand)

    selected = dock.widget().selectedItems()
    assert len(selected) == 1
    assert selected[0].text(0).startswith("grand")


def test_outliner_label_shows_component_count(qapp: Any) -> None:
    del qapp
    from posecascade.scene.component import MeshRefComponent  # noqa: PLC0415

    root = Node(name="root")
    decorated = Node(name="decorated")
    decorated.add_component(MeshRefComponent(mesh_indices=(0,)))
    decorated.add_component(MeshRefComponent(mesh_indices=(1,)))
    root.add_child(decorated)
    scene = Scene(root=root)

    dock = OutlinerDock()
    dock.set_scene(scene)
    decorated_item = dock.widget().topLevelItem(0).child(0)
    assert "(2)" in decorated_item.text(0)


def test_outliner_select_none_clears_highlight(qapp: Any) -> None:
    del qapp
    dock = OutlinerDock()
    dock.set_scene(_build_scene())
    dock.widget().setCurrentItem(dock.widget().topLevelItem(0))
    assert dock.widget().selectedItems()
    dock.select_node(None)
    assert not dock.widget().selectedItems()


def test_outliner_delete_removes_node_from_scene(qapp: Any) -> None:
    """Deleting a non-root node detaches it from its parent and emits node_deleted."""
    del qapp
    dock = OutlinerDock()
    scene = _build_scene()
    dock.set_scene(scene)

    received: list[Node] = []
    dock.node_deleted.connect(received.append)

    # Select child_a (root → child_a) and delete via the action
    root_item = dock.widget().topLevelItem(0)
    child_a_item = root_item.child(0)
    dock.widget().setCurrentItem(child_a_item)
    dock._delete_selected()  # noqa: SLF001 — tests the deletion path directly

    # Scene mutated.
    assert len(scene.root.children) == 1
    assert scene.root.children[0].name == "child_b"
    # Signal fired with the removed node.
    assert len(received) == 1
    assert received[0].name == "child_a"
    # Tree refreshed without child_a.
    refreshed_root = dock.widget().topLevelItem(0)
    assert refreshed_root.childCount() == 1


def test_outliner_delete_root_is_noop(qapp: Any) -> None:
    """Deleting the root would orphan the whole scene — must be silently rejected."""
    del qapp
    dock = OutlinerDock()
    scene = _build_scene()
    dock.set_scene(scene)
    dock.widget().setCurrentItem(dock.widget().topLevelItem(0))

    received: list[Node] = []
    dock.node_deleted.connect(received.append)
    dock._delete_selected()  # noqa: SLF001

    # Scene untouched.
    assert scene.root is not None
    assert len(scene.root.children) == 2
    assert received == []
