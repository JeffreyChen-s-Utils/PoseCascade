"""Tests for the PLY importer (ASCII)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ply.importer import PlyImporter

from posecascade.errors import MalformedAssetError

_ASCII_QUAD = """\
ply
format ascii 1.0
comment generated for tests
element vertex 4
property float x
property float y
property float z
property float nx
property float ny
property float nz
element face 2
property list uchar int vertex_indices
end_header
0.0 0.0 0.0 0.0 0.0 1.0
1.0 0.0 0.0 0.0 0.0 1.0
1.0 1.0 0.0 0.0 0.0 1.0
0.0 1.0 0.0 0.0 0.0 1.0
3 0 1 2
3 0 2 3
"""

_ASCII_QUAD_NGON = """\
ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
1 1 0
0 1 0
4 0 1 2 3
"""


def test_ascii_quad(tmp_path: Path) -> None:
    path = tmp_path / "quad.ply"
    path.write_text(_ASCII_QUAD, encoding="utf-8")
    scene = PlyImporter().load(path)
    mesh = scene.meshes[0]
    assert mesh.positions.shape == (4, 3)
    assert mesh.indices.shape == (6,)
    assert mesh.normals is not None and mesh.normals.shape == (4, 3)
    np.testing.assert_allclose(mesh.normals, np.tile([0.0, 0.0, 1.0], (4, 1)))


def test_quad_face_fan_triangulated(tmp_path: Path) -> None:
    path = tmp_path / "quad.ply"
    path.write_text(_ASCII_QUAD_NGON, encoding="utf-8")
    mesh = PlyImporter().load(path).meshes[0]
    assert mesh.indices.tolist() == [0, 1, 2, 0, 2, 3]


def test_missing_magic_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.ply"
    path.write_text("not_ply\n", encoding="utf-8")
    with pytest.raises(MalformedAssetError):
        PlyImporter().load(path)


def test_binary_format_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.ply"
    path.write_text(
        "ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n",
        encoding="utf-8",
    )
    with pytest.raises(MalformedAssetError):
        PlyImporter().load(path)


def test_truncated_body_raises(tmp_path: Path) -> None:
    bad = """\
ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
"""
    path = tmp_path / "bad.ply"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(MalformedAssetError):
        PlyImporter().load(path)
