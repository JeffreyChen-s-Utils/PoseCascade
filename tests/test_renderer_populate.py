"""Tests for :meth:`posecascade.render.renderer.Renderer.populate_from_scene`.

Verifies the meshes-per-node mapping is built correctly without touching GL —
the GL path is exercised by tests/test_render_smoke.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from posecascade.assets.types import ImportedScene, Mesh
from posecascade.render.renderer import Renderer
from posecascade.scene.component import MeshRefComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene


@pytest.fixture
def fake_renderer(monkeypatch: pytest.MonkeyPatch) -> Renderer:
    """Renderer where mesh upload returns a sentinel — no real GL calls."""
    from posecascade.render import renderer as renderer_module  # noqa: PLC0415
    upload_calls: list[Mesh] = []

    def fake_upload(mesh: Mesh) -> object:
        upload_calls.append(mesh)
        return ("gl_mesh", len(upload_calls) - 1)

    monkeypatch.setattr(renderer_module, "upload_mesh", fake_upload)
    r = Renderer(shaders_root=MagicMock())
    r._upload_calls = upload_calls  # type: ignore[attr-defined]
    return r


def _mesh(name: str) -> Mesh:
    return Mesh(
        name=name,
        positions=np.zeros((3, 3), dtype=np.float32),
        indices=np.zeros((3,), dtype=np.uint32),
    )


def test_populate_skips_when_scene_is_none(fake_renderer: Renderer) -> None:
    fake_renderer.populate_from_scene(ImportedScene(meshes=(_mesh("a"),), scene=None))
    assert fake_renderer._upload_calls == []  # type: ignore[attr-defined]


def test_populate_uploads_each_referenced_mesh(fake_renderer: Renderer) -> None:
    meshes = (_mesh("a"), _mesh("b"))
    scene = Scene()
    node = Node(name="object")
    node.add_component(MeshRefComponent(mesh_indices=(0, 1)))
    scene.root.add_child(node)
    fake_renderer.populate_from_scene(ImportedScene(meshes=meshes, scene=scene))
    uploaded = fake_renderer._upload_calls  # type: ignore[attr-defined]
    assert [m.name for m in uploaded] == ["a", "b"]
    bindings = fake_renderer._node_to_mesh[id(node)]  # noqa: SLF001
    assert len(bindings) == 2


def test_populate_dedupes_shared_mesh_index(fake_renderer: Renderer) -> None:
    meshes = (_mesh("shared"),)
    scene = Scene()
    a = Node(name="a")
    b = Node(name="b")
    a.add_component(MeshRefComponent(mesh_indices=(0,)))
    b.add_component(MeshRefComponent(mesh_indices=(0,)))
    scene.root.add_child(a)
    scene.root.add_child(b)
    fake_renderer.populate_from_scene(ImportedScene(meshes=meshes, scene=scene))
    # Mesh uploaded once, referenced from two nodes.
    assert len(fake_renderer._upload_calls) == 1  # type: ignore[attr-defined]
    assert id(a) in fake_renderer._node_to_mesh  # noqa: SLF001
    assert id(b) in fake_renderer._node_to_mesh  # noqa: SLF001


def test_populate_ignores_out_of_range_indices(fake_renderer: Renderer) -> None:
    scene = Scene()
    node = Node(name="bad")
    node.add_component(MeshRefComponent(mesh_indices=(5, 17)))
    scene.root.add_child(node)
    fake_renderer.populate_from_scene(ImportedScene(meshes=(_mesh("only"),), scene=scene))
    assert fake_renderer._upload_calls == []  # type: ignore[attr-defined]
    assert id(node) not in fake_renderer._node_to_mesh  # noqa: SLF001


def test_populate_walks_descendants(fake_renderer: Renderer) -> None:
    scene = Scene()
    parent = Node(name="parent")
    child = Node(name="child")
    child.add_component(MeshRefComponent(mesh_indices=(0,)))
    parent.add_child(child)
    scene.root.add_child(parent)
    fake_renderer.populate_from_scene(ImportedScene(meshes=(_mesh("c"),), scene=scene))
    assert len(fake_renderer._upload_calls) == 1  # type: ignore[attr-defined]
    assert id(child) in fake_renderer._node_to_mesh  # noqa: SLF001
