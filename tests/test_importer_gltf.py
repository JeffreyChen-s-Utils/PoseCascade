"""Tests for the glTF importer.

Builds a tiny single-triangle glTF in-memory using ``pygltflib`` and verifies
the importer's accessor decoding, node hierarchy, and embedded buffer caps.
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest
from gltf.importer import GltfImporter
from pygltflib import (
    GLTF2,
    Accessor,
    Attributes,
    Buffer,
    BufferView,
    Primitive,
)
from pygltflib import (
    Mesh as GltfMesh,
)
from pygltflib import (
    Node as GltfNode,
)
from pygltflib import (
    Scene as GltfScene,
)

from posecascade.errors import MalformedAssetError


def _triangle_blob() -> tuple[bytes, int, int]:
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    ).tobytes()
    indices = np.array([0, 1, 2], dtype=np.uint32).tobytes()
    return positions + indices, len(positions), len(indices)


def _build_triangle_gltf(blob_uri: str) -> GLTF2:
    pos_bytes, _ = (12 * 3, 4 * 3)
    gltf = GLTF2()
    gltf.buffers = [Buffer(byteLength=pos_bytes + (4 * 3), uri=blob_uri)]
    gltf.bufferViews = [
        BufferView(buffer=0, byteOffset=0, byteLength=pos_bytes),
        BufferView(buffer=0, byteOffset=pos_bytes, byteLength=4 * 3),
    ]
    gltf.accessors = [
        Accessor(bufferView=0, componentType=5126, count=3, type="VEC3",
                 max=[1.0, 1.0, 0.0], min=[0.0, 0.0, 0.0]),
        Accessor(bufferView=1, componentType=5125, count=3, type="SCALAR"),
    ]
    primitive = Primitive(attributes=Attributes(POSITION=0), indices=1)
    gltf.meshes = [GltfMesh(name="tri", primitives=[primitive])]
    gltf.nodes = [GltfNode(name="root", mesh=0, translation=[0.5, 0.0, 0.0])]
    gltf.scenes = [GltfScene(nodes=[0])]
    gltf.scene = 0
    return gltf


def test_loads_triangle_from_external_bin(tmp_path: Path) -> None:
    blob, _, _ = _triangle_blob()
    bin_path = tmp_path / "tri.bin"
    bin_path.write_bytes(blob)
    gltf = _build_triangle_gltf(blob_uri="tri.bin")
    gltf_path = tmp_path / "tri.gltf"
    gltf.save(str(gltf_path))

    scene = GltfImporter().load(gltf_path)
    assert len(scene.meshes) == 1
    mesh = scene.meshes[0]
    assert mesh.positions.shape == (3, 3)
    np.testing.assert_allclose(
        mesh.positions, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    assert mesh.indices.tolist() == [0, 1, 2]


def test_loads_triangle_from_data_uri(tmp_path: Path) -> None:
    blob, _, _ = _triangle_blob()
    data_uri = "data:application/octet-stream;base64," + base64.b64encode(blob).decode()
    gltf = _build_triangle_gltf(blob_uri=data_uri)
    gltf_path = tmp_path / "tri.gltf"
    gltf.save(str(gltf_path))

    scene = GltfImporter().load(gltf_path)
    assert scene.meshes[0].positions.shape == (3, 3)


def test_node_hierarchy_walks(tmp_path: Path) -> None:
    blob, _, _ = _triangle_blob()
    bin_path = tmp_path / "tri.bin"
    bin_path.write_bytes(blob)
    gltf = _build_triangle_gltf(blob_uri="tri.bin")
    gltf.nodes.append(GltfNode(name="child", mesh=0))
    gltf.nodes[0].children = [1]
    gltf.save(str(tmp_path / "tri.gltf"))

    scene = GltfImporter().load(tmp_path / "tri.gltf")
    root_children = scene.scene.root.children
    assert len(root_children) == 1
    assert root_children[0].name == "root"
    assert len(root_children[0].children) == 1
    assert root_children[0].children[0].name == "child"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MalformedAssetError):
        GltfImporter().load(tmp_path / "missing.gltf")


def test_oversize_data_uri_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # `importers/` is a namespace package, so the importer manager loads
    # `gltf.importer` (the same module path used by the rest of the engine).
    # Patch THAT module — `importers.gltf.importer` is a different object.
    import gltf.importer as importer_module  # noqa: PLC0415
    monkeypatch.setattr(importer_module, "MAX_EMBEDDED_BUFFER_BYTES", 8, raising=False)

    blob, _, _ = _triangle_blob()
    data_uri = "data:application/octet-stream;base64," + base64.b64encode(blob).decode()
    gltf = _build_triangle_gltf(blob_uri=data_uri)
    gltf_path = tmp_path / "tri.gltf"
    gltf.save(str(gltf_path))
    with pytest.raises(MalformedAssetError):
        GltfImporter().load(gltf_path)


def test_unsupported_data_uri_scheme(tmp_path: Path) -> None:
    gltf = _build_triangle_gltf(blob_uri="data:image/png;base64,AAAA")
    gltf_path = tmp_path / "tri.gltf"
    gltf.save(str(gltf_path))
    with pytest.raises(MalformedAssetError):
        GltfImporter().load(gltf_path)
