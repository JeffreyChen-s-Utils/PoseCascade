"""Tests for the DirectX .x text importer.

Covers a hand-written tiny .x file (so the test does not depend on any
upstream sample), the malformed-header guard, and registry integration
through :class:`ImporterManager`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from x.importer import XImporter

from posecascade.assets.importer_manager import ImporterManager
from posecascade.errors import MalformedAssetError, UnsupportedFormatError
from posecascade.scene.component import MeshRefComponent

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_IMPORTERS_ROOT = _PROJECT_ROOT / "importers"


def _tiny_cube_x() -> str:
    """Return a hand-rolled .x cube — 8 vertices, 12 triangles, one Material."""
    return """xof 0303txt 0032

Mesh Cube {
 8;
 -1.0;-1.0;-1.0;,
 1.0;-1.0;-1.0;,
 1.0;1.0;-1.0;,
 -1.0;1.0;-1.0;,
 -1.0;-1.0;1.0;,
 1.0;-1.0;1.0;,
 1.0;1.0;1.0;,
 -1.0;1.0;1.0;;
 12;
 3;0,1,2;,
 3;0,2,3;,
 3;4,6,5;,
 3;4,7,6;,
 3;0,4,5;,
 3;0,5,1;,
 3;3,2,6;,
 3;3,6,7;,
 3;1,5,6;,
 3;1,6,2;,
 3;0,3,7;,
 3;0,7,4;;

 MeshNormals {
  8;
  -0.577;-0.577;-0.577;,
  0.577;-0.577;-0.577;,
  0.577;0.577;-0.577;,
  -0.577;0.577;-0.577;,
  -0.577;-0.577;0.577;,
  0.577;-0.577;0.577;,
  0.577;0.577;0.577;,
  -0.577;0.577;0.577;;
  12;
  3;0,1,2;,
  3;0,2,3;,
  3;4,6,5;,
  3;4,7,6;,
  3;0,4,5;,
  3;0,5,1;,
  3;3,2,6;,
  3;3,6,7;,
  3;1,5,6;,
  3;1,6,2;,
  3;0,3,7;,
  3;0,7,4;;
 }

 MeshTextureCoords {
  8;
  0.0;0.0;,
  1.0;0.0;,
  1.0;1.0;,
  0.0;1.0;,
  0.0;0.0;,
  1.0;0.0;,
  1.0;1.0;,
  0.0;1.0;;
 }

 MeshMaterialList {
  1;
  12;
  0,0,0,0,0,0,0,0,0,0,0,0;;
  Material RedMat {
   0.8;0.2;0.1;1.0;;
   5.0;
   0.0;0.0;0.0;;
   0.0;0.0;0.0;;
  }
 }
}
"""


def _tiny_frame_x() -> str:
    """A Frame containing a transform matrix + a tiny one-triangle Mesh."""
    return """xof 0303txt 0032

Frame Root {
 FrameTransformMatrix {
  1.0,0.0,0.0,0.0,
  0.0,1.0,0.0,0.0,
  0.0,0.0,1.0,0.0,
  2.0,3.0,4.0,1.0;;
 }

 Mesh Triangle {
  3;
  0.0;0.0;0.0;,
  1.0;0.0;0.0;,
  0.0;1.0;0.0;;
  1;
  3;0,1,2;;
 }
}
"""


# ----- happy path ------------------------------------------------------
def test_x_importer_loads_tiny_cube(tmp_path: Path) -> None:
    path = tmp_path / "cube.x"
    path.write_text(_tiny_cube_x(), encoding="utf-8")
    scene = XImporter().load(path)
    assert len(scene.meshes) == 1
    mesh = scene.meshes[0]
    assert mesh.positions.shape == (8, 3)
    assert mesh.indices.shape == (36,)
    assert mesh.normals is not None
    assert mesh.normals.shape == (8, 3)
    assert mesh.texcoords_0 is not None
    assert mesh.texcoords_0.shape == (8, 2)
    assert mesh.base_color is not None
    np.testing.assert_allclose(mesh.base_color, (0.8, 0.2, 0.1, 1.0), atol=1e-5)


def test_x_importer_attaches_mesh_to_scene_root(tmp_path: Path) -> None:
    path = tmp_path / "cube.x"
    path.write_text(_tiny_cube_x(), encoding="utf-8")
    scene = XImporter().load(path)
    assert scene.scene is not None
    children = list(scene.scene.root.children)
    assert len(children) == 1
    mesh_node = children[0]
    refs = [c for c in mesh_node.components if isinstance(c, MeshRefComponent)]
    assert len(refs) == 1
    assert refs[0].mesh_indices == (0,)


def test_x_importer_handles_frame_with_transform(tmp_path: Path) -> None:
    path = tmp_path / "frame.x"
    path.write_text(_tiny_frame_x(), encoding="utf-8")
    scene = XImporter().load(path)
    assert len(scene.meshes) == 1
    assert scene.scene is not None
    frame_nodes = list(scene.scene.root.children)
    assert len(frame_nodes) == 1
    frame_node = frame_nodes[0]
    assert frame_node.name == "Root"
    # Translation column is (2, 3, 4); decompose_trs pulls it out.
    np.testing.assert_allclose(
        frame_node.transform.translation, [2.0, 3.0, 4.0], atol=1e-5,
    )


def test_x_importer_skips_unknown_templates(tmp_path: Path) -> None:
    """Unknown templates inside a Mesh are ignored; the importer keeps reading."""
    text = """xof 0303txt 0032

Mesh M {
 3;
 0.0;0.0;0.0;,
 1.0;0.0;0.0;,
 0.0;1.0;0.0;;
 1;
 3;0,1,2;;
 SomeFutureTemplate {
  1;2;3;4;;
 }
}
"""
    path = tmp_path / "skip.x"
    path.write_text(text, encoding="utf-8")
    scene = XImporter().load(path)
    assert len(scene.meshes) == 1
    assert scene.meshes[0].positions.shape == (3, 3)


def test_x_importer_strips_line_comments(tmp_path: Path) -> None:
    text = """xof 0303txt 0032

// top-level comment
Mesh M {
 // explanatory comment
 3;
 0.0;0.0;0.0;,
 1.0;0.0;0.0;,
 0.0;1.0;0.0;;
 1;
 3;0,1,2;;
}
"""
    path = tmp_path / "comments.x"
    path.write_text(text, encoding="utf-8")
    scene = XImporter().load(path)
    assert scene.meshes[0].positions.shape == (3, 3)


# ----- error handling ---------------------------------------------------
def test_x_importer_rejects_missing_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.x"
    path.write_text("Mesh M { 0;; 0;; }", encoding="utf-8")
    with pytest.raises(MalformedAssetError, match="xof"):
        XImporter().load(path)


def test_x_importer_rejects_missing_file() -> None:
    with pytest.raises(MalformedAssetError, match="not found"):
        XImporter().load(Path("/does/not/exist.x"))


def test_x_importer_rejects_file_with_no_meshes(tmp_path: Path) -> None:
    path = tmp_path / "empty.x"
    path.write_text("xof 0303txt 0032\n\nFrame Root { }\n", encoding="utf-8")
    with pytest.raises(MalformedAssetError, match="no usable Mesh"):
        XImporter().load(path)


# ----- registry --------------------------------------------------------
def test_importer_manager_registers_x_extension() -> None:
    manager = ImporterManager(importers_root=_IMPORTERS_ROOT)
    manager.discover()
    importer = manager.importer_for(Path("dummy.x"))
    assert isinstance(importer, XImporter)


def test_importer_manager_rejects_unknown_extension() -> None:
    manager = ImporterManager(importers_root=_IMPORTERS_ROOT)
    manager.discover()
    with pytest.raises(UnsupportedFormatError):
        manager.importer_for(Path("dummy.unknown"))
