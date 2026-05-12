"""Tests for the selection-overlay outline pass.

Drives :meth:`Renderer.set_selected_holder` +
:meth:`Renderer._draw_selection_overlay` with the underlying outline shader
stubbed out, so the assertions stay GL-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np

from posecascade.render.renderer import (
    _SELECTION_EDGE_COLOR,
    _SELECTION_EDGE_SIZE,
    Renderer,
)
from posecascade.scene.node import Node


@dataclass
class _Recorded:
    """One captured ``_draw_outline_with`` call (no GL ran)."""

    node: Node
    mesh_id: int
    edge_size: float
    edge_color: tuple[float, float, float, float]


@dataclass
class _StubRenderer:
    """Subset of Renderer state the overlay walk reads."""

    renderer: Renderer
    calls: list[_Recorded] = field(default_factory=list)


def _make_renderer_with_meshes(holder_children: list[Node]) -> _StubRenderer:
    """Build a Renderer wired with one fake mesh id per child node.

    The fake outline implementation records every call without invoking GL.
    """
    r = Renderer(shaders_root=MagicMock())
    stub = _StubRenderer(renderer=r)
    for i, node in enumerate(holder_children):
        r._meshes[i] = MagicMock(name=f"gl_mesh_{i}")          # noqa: SLF001
        r._node_to_mesh[id(node)] = [i]                         # noqa: SLF001

    def fake_draw_outline_with(
        node: Node,
        gl_mesh: object,
        skin: object,
        edge_size: float,
        edge_color: tuple[float, float, float, float],
        view: object,
        proj: object,
    ) -> None:
        del gl_mesh, skin, view, proj
        mesh_id = next(
            mid for mid, m in r._meshes.items()                 # noqa: SLF001
            if m is r._meshes.get(r._node_to_mesh[id(node)][0]) # noqa: SLF001
        )
        stub.calls.append(
            _Recorded(node=node, mesh_id=mesh_id,
                      edge_size=edge_size, edge_color=tuple(edge_color)),
        )

    r._draw_outline_with = fake_draw_outline_with               # type: ignore[method-assign]
    return stub


def test_selection_overlay_visits_every_mesh_under_holder() -> None:
    """Every node under the selected holder that owns a mesh gets re-outlined."""
    holder = Node(name="char")
    child_a = Node(name="hair")
    child_b = Node(name="skirt")
    holder.add_child(child_a)
    holder.add_child(child_b)
    stub = _make_renderer_with_meshes([child_a, child_b])
    stub.renderer.set_selected_holder(holder)

    view = np.eye(4, dtype=np.float32)
    proj = np.eye(4, dtype=np.float32)
    stub.renderer._draw_selection_overlay(view, proj)            # noqa: SLF001

    visited = {call.node.name for call in stub.calls}
    assert visited == {"hair", "skirt"}
    for call in stub.calls:
        assert call.edge_size == _SELECTION_EDGE_SIZE
        assert call.edge_color == _SELECTION_EDGE_COLOR


def test_selection_overlay_skips_when_nothing_selected() -> None:
    """Without a selection, the overlay walk is a no-op — zero outline calls."""
    holder = Node(name="char")
    child = Node(name="body")
    holder.add_child(child)
    stub = _make_renderer_with_meshes([child])

    view = np.eye(4, dtype=np.float32)
    proj = np.eye(4, dtype=np.float32)
    stub.renderer._draw_selection_overlay(view, proj)            # noqa: SLF001

    assert stub.calls == []


def test_set_selected_holder_clears_with_none() -> None:
    """Passing ``None`` to ``set_selected_holder`` cancels a prior selection."""
    holder = Node(name="char")
    r = Renderer(shaders_root=MagicMock())
    r.set_selected_holder(holder)
    assert r._selected_holder is holder                          # noqa: SLF001
    r.set_selected_holder(None)
    assert r._selected_holder is None                            # noqa: SLF001


def test_selection_overlay_handles_holder_outside_mesh_map() -> None:
    """A selected holder with no mesh-bearing descendants is harmless."""
    holder = Node(name="empty")
    r = Renderer(shaders_root=MagicMock())
    r.set_selected_holder(holder)
    calls: list[object] = []
    r._draw_outline_with = lambda *a, **kw: calls.append(a)      # type: ignore[method-assign]
    r._draw_selection_overlay(                                   # noqa: SLF001
        np.eye(4, dtype=np.float32),
        np.eye(4, dtype=np.float32),
    )
    assert calls == []
