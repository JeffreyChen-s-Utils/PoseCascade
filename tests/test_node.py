"""Tests for :class:`posecascade.scene.node.Node` Composite behaviour."""
from __future__ import annotations

import pytest

from posecascade.errors import SceneError
from posecascade.scene.node import Node


def test_add_child_sets_parent() -> None:
    parent = Node(name="parent")
    child = Node(name="child")
    parent.add_child(child)
    assert child.parent is parent
    assert parent.children == [child]


def test_remove_child_clears_parent() -> None:
    parent = Node(name="parent")
    child = Node(name="child")
    parent.add_child(child)
    parent.remove_child(child)
    assert child.parent is None
    assert parent.children == []


def test_add_child_rejects_self() -> None:
    node = Node(name="self_loop")
    with pytest.raises(SceneError):
        node.add_child(node)


def test_add_child_rejects_already_parented() -> None:
    a = Node(name="a")
    b = Node(name="b")
    c = Node(name="c")
    a.add_child(b)
    with pytest.raises(SceneError):
        c.add_child(b)


def test_add_child_rejects_cycle() -> None:
    a = Node(name="a")
    b = Node(name="b")
    a.add_child(b)
    with pytest.raises(SceneError):
        b.add_child(a)


def test_remove_child_rejects_non_child() -> None:
    a = Node(name="a")
    stranger = Node(name="stranger")
    with pytest.raises(SceneError):
        a.remove_child(stranger)


def test_traverse_pre_order() -> None:
    root = Node(name="root")
    left = Node(name="left")
    right = Node(name="right")
    leaf = Node(name="leaf")
    root.add_child(left)
    root.add_child(right)
    left.add_child(leaf)
    names = [n.name for n in root.traverse()]
    assert names == ["root", "left", "leaf", "right"]


def test_find_first_returns_descendant() -> None:
    root = Node(name="root")
    target = Node(name="target")
    root.add_child(Node(name="other"))
    root.add_child(target)
    assert root.find_first("target") is target


def test_find_first_returns_none_when_missing() -> None:
    root = Node(name="root")
    assert root.find_first("ghost") is None
