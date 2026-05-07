"""Tests for :meth:`Renderer.populate_from_slots` data plumbing.

We don't need an actual GL context for the structural assertions —
calling ``populate_from_scene`` requires uploads, but ``populate_from_slots``
does the scene-graph composition first. Here we verify the composition
(slot transform parenting) without uploading any meshes by mocking
out the upload step.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pmx.importer import PmxImporter

from posecascade.render.renderer import Renderer
from posecascade.scene.model_slot import ModelSlot, SceneSlots
from posecascade.utils.math3d import vec3

_TINY_PMX = Path(__file__).resolve().parent / "fixtures" / "mmd" / "tiny.pmx"


def _slot_with_imported(name: str, *, visible: bool = True) -> ModelSlot:
    return ModelSlot(
        name=name,
        imported=PmxImporter().load(_TINY_PMX),
        visible=visible,
    )


def _renderer_no_gl(monkeypatch) -> Renderer:                  # noqa: ANN001
    """Build a renderer that mocks out the GL upload helpers so no GL
    context is needed; we only care about the scene-composition logic."""
    renderer = Renderer(shaders_root=Path(__file__).resolve().parent.parent / "shaders")
    monkeypatch.setattr(
        "posecascade.render.renderer.upload_mesh",
        lambda mesh: type("FakeGLMesh", (), {
            "vbos": (), "vao": 0, "ebo": 0, "index_count": 0,
            "aabb_min": (0, 0, 0), "aabb_max": (0, 0, 0),
            "base_color": None, "base_color_texture_id": None,
            "mmd_material": mesh.mmd_material,
            "sphere_texture_id": None, "toon_texture_id": None,
        })(),
    )
    monkeypatch.setattr(renderer, "_upload_textures", lambda imported: {})
    return renderer


def test_populate_from_slots_creates_one_root_per_slot(monkeypatch) -> None:    # noqa: ANN001
    """Two slots → composite scene root has two synthetic slot roots."""
    slots = SceneSlots()
    slots.add(_slot_with_imported("a"))
    slots.add(_slot_with_imported("b"))
    renderer = _renderer_no_gl(monkeypatch)
    composite = renderer.populate_from_slots(slots)
    top_names = [child.name for child in composite.root.children]
    assert top_names == ["slot_a", "slot_b"]


def test_populate_from_slots_skips_invisible_slot(monkeypatch) -> None:        # noqa: ANN001
    slots = SceneSlots()
    slots.add(_slot_with_imported("visible"))
    slots.add(_slot_with_imported("hidden", visible=False))
    renderer = _renderer_no_gl(monkeypatch)
    composite = renderer.populate_from_slots(slots)
    top_names = [child.name for child in composite.root.children]
    assert "slot_visible" in top_names
    assert "slot_hidden" not in top_names


def test_slot_transform_carried_via_synthetic_root(monkeypatch) -> None:        # noqa: ANN001
    """Setting the slot's transform must show up on the synthetic root
    Node added to the composite scene."""
    slot = _slot_with_imported("character")
    slot.transform.set_translation(vec3(2.0, 1.5, -1.0))
    slots = SceneSlots()
    slots.add(slot)
    renderer = _renderer_no_gl(monkeypatch)
    composite = renderer.populate_from_slots(slots)
    slot_root = composite.root.children[0]
    np.testing.assert_allclose(
        slot_root.transform.translation, [2.0, 1.5, -1.0],
    )


def test_imported_scene_root_emptied_after_populate(monkeypatch) -> None:        # noqa: ANN001
    """Re-parenting moves children out of the original imported scene's
    root; this is the documented one-way behaviour."""
    slot = _slot_with_imported("once")
    slots = SceneSlots()
    slots.add(slot)
    renderer = _renderer_no_gl(monkeypatch)
    renderer.populate_from_slots(slots)
    assert slot.imported.scene.root.children == []


def test_set_light_writes_state(monkeypatch) -> None:                             # noqa: ANN001
    """``set_light`` updates the renderer's stored direction + colour."""
    renderer = _renderer_no_gl(monkeypatch)
    renderer.set_light(direction=(0.5, -1.0, 0.5), color=(0.8, 0.6, 0.4))
    assert renderer._light_direction == (0.5, -1.0, 0.5)            # noqa: SLF001
    assert renderer._light_color == (0.8, 0.6, 0.4)                  # noqa: SLF001
