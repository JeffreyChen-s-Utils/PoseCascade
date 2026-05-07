"""Tests for the OBJ importer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from obj.importer import ObjImporter

from posecascade.errors import MalformedAssetError, UnsafePathError

_TRI_OBJ = """\
# trivial triangle
o my_triangle
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vn 0.0 0.0 1.0
vt 0.0 0.0
vt 1.0 0.0
vt 0.0 1.0
f 1/1/1 2/2/1 3/3/1
"""

_QUAD_OBJ = """\
v -1.0 -1.0 0.0
v  1.0 -1.0 0.0
v  1.0  1.0 0.0
v -1.0  1.0 0.0
f 1 2 3 4
"""


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_triangle(tmp_path: Path) -> None:
    obj_path = _write(tmp_path / "tri.obj", _TRI_OBJ)
    scene = ObjImporter().load(obj_path)
    assert len(scene.meshes) == 1
    mesh = scene.meshes[0]
    assert mesh.name == "my_triangle"
    assert mesh.positions.shape == (3, 3)
    assert mesh.indices.shape == (3,)
    assert mesh.normals is not None and mesh.normals.shape == (3, 3)
    assert mesh.texcoords_0 is not None and mesh.texcoords_0.shape == (3, 2)
    np.testing.assert_allclose(mesh.normals, np.tile([0.0, 0.0, 1.0], (3, 1)))


def test_quad_is_fan_triangulated(tmp_path: Path) -> None:
    obj_path = _write(tmp_path / "quad.obj", _QUAD_OBJ)
    scene = ObjImporter().load(obj_path)
    mesh = scene.meshes[0]
    assert mesh.indices.shape == (6,)
    # First triangle: 0,1,2; second: 0,2,3.
    np.testing.assert_array_equal(mesh.indices, [0, 1, 2, 0, 2, 3])


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MalformedAssetError):
        ObjImporter().load(tmp_path / "missing.obj")


def test_no_positions_raises(tmp_path: Path) -> None:
    obj_path = _write(tmp_path / "empty.obj", "# no vertices here\n")
    with pytest.raises(MalformedAssetError):
        ObjImporter().load(obj_path)


def test_face_with_too_few_corners_raises(tmp_path: Path) -> None:
    obj_path = _write(
        tmp_path / "bad.obj",
        "v 0 0 0\nv 1 0 0\nf 1 2\n",
    )
    with pytest.raises(MalformedAssetError):
        ObjImporter().load(obj_path)


def test_mtllib_traversal_blocked(tmp_path: Path) -> None:
    obj_path = _write(
        tmp_path / "evil.obj",
        "mtllib ../../../etc/passwd\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
    )
    with pytest.raises(UnsafePathError):
        ObjImporter().load(obj_path)


def test_negative_relative_indices(tmp_path: Path) -> None:
    body = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n"
    obj_path = _write(tmp_path / "rel.obj", body)
    scene = ObjImporter().load(obj_path)
    mesh = scene.meshes[0]
    assert mesh.indices.tolist() == [0, 1, 2]
