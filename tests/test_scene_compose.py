"""Tests for the example scene composer.

Exercises the pure scene-building part — no Qt event loop, no GL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Examples is not a package — load the module directly.
sys.path.insert(0, str(PROJECT_ROOT / "examples"))


@pytest.fixture
def services() -> object:
    from posecascade.app.registry import build_services  # noqa: PLC0415
    return build_services(project_root=PROJECT_ROOT)


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "character": Path("/__missing_character__/x.glb"),
        "dog": Path("/__missing_dog__/x.glb"),
        "room": Path("/__missing_room__/x.glb"),
        "script": Path("/__missing__/x.py"),
        "character_scale": 1.0,
        "dog_scale": 1.0,
        "room_scale": 1.0,
        "character_up": "y",
        "dog_up": "y",
        "room_up": "y",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_procedural_room_when_no_assets(services: object) -> None:
    from scene_compose import build_composite_scene  # noqa: PLC0415

    scene, imports = build_composite_scene(services, _args())
    # The procedural anime bedroom is returned as an ImportedScene, so it
    # shows up in the imports list and goes through populate_from_scene.
    assert len(imports) == 1
    # Composite has a single child: the room holder.
    assert [child.name for child in scene.root.children] == ["room"]
    # Bedroom pieces appear somewhere under that holder (now nested under
    # imported.scene.root rather than direct children — see _wrap_under_holder).
    room_holder = scene.root.children[0]
    piece_names = {node.name for node in room_holder.traverse()}
    assert {"floor", "ceiling", "bed_mattress", "desk_top", "window"} <= piece_names


def test_holders_named_for_provided_assets(
    services: object, tmp_path: Path,
) -> None:
    """When character/dog .glb files are loadable, holders should appear by name."""
    pytest.importorskip("pygltflib")
    from gltf.importer import GltfImporter  # noqa: PLC0415
    from scene_compose import build_composite_scene  # noqa: PLC0415

    placeholder_glb = _make_minimal_glb(tmp_path)
    args = _args(character=placeholder_glb, dog=placeholder_glb)
    GltfImporter().load(placeholder_glb)  # sanity check

    scene, imports = build_composite_scene(services, args)
    names = [child.name for child in scene.root.children]
    assert "character" in names
    assert "dog" in names
    assert "room" in names  # procedural bedroom still added
    # 2 real imports + 1 procedural bedroom = 3.
    assert len(imports) == 3


def _make_minimal_glb(tmp_path: Path) -> Path:
    """Write a 1-triangle .glb the project's GltfImporter accepts."""
    import base64  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    from pygltflib import (  # noqa: PLC0415
        GLTF2,
        Accessor,
        Attributes,
        Buffer,
        BufferView,
        Mesh,
        Node,
        Primitive,
        Scene,
    )
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    ).tobytes()
    indices = np.array([0, 1, 2], dtype=np.uint32).tobytes()
    blob = positions + indices
    data_uri = "data:application/octet-stream;base64," + base64.b64encode(blob).decode()
    g = GLTF2()
    g.buffers = [Buffer(byteLength=len(blob), uri=data_uri)]
    g.bufferViews = [
        BufferView(buffer=0, byteOffset=0, byteLength=len(positions)),
        BufferView(buffer=0, byteOffset=len(positions), byteLength=len(indices)),
    ]
    g.accessors = [
        Accessor(
            bufferView=0, componentType=5126, count=3, type="VEC3",
            max=[1.0, 1.0, 0.0], min=[0.0, 0.0, 0.0],
        ),
        Accessor(bufferView=1, componentType=5125, count=3, type="SCALAR"),
    ]
    primitive = Primitive(attributes=Attributes(POSITION=0), indices=1)
    g.meshes = [Mesh(name="tri", primitives=[primitive])]
    g.nodes = [Node(name="root", mesh=0)]
    g.scenes = [Scene(nodes=[0])]
    g.scene = 0
    out = tmp_path / "tri.gltf"
    g.save(str(out))
    return out
