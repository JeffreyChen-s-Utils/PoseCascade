"""Tests for the STL importer (ASCII + binary)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from stl.importer import StlImporter

from posecascade.errors import MalformedAssetError

_ASCII_TRI = """\
solid tri
  facet normal 0.0 0.0 1.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 1.0 0.0 0.0
      vertex 0.0 1.0 0.0
    endloop
  endfacet
endsolid tri
"""


def _binary_blob(triangles: list[tuple[tuple[float, float, float], ...]]) -> bytes:
    payload = bytearray(b"\x00" * 80)
    payload += struct.pack("<I", len(triangles))
    for normal, v0, v1, v2 in triangles:
        payload += struct.pack(
            "<12fH",
            *normal, *v0, *v1, *v2, 0,
        )
    return bytes(payload)


def test_ascii_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "tri.stl"
    path.write_text(_ASCII_TRI, encoding="utf-8")
    scene = StlImporter().load(path)
    mesh = scene.meshes[0]
    assert mesh.positions.shape == (3, 3)
    assert mesh.indices.tolist() == [0, 1, 2]
    np.testing.assert_allclose(mesh.normals, np.tile([0.0, 0.0, 1.0], (3, 1)))


def test_binary_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "tri.stl"
    triangles = [
        (
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    ]
    path.write_bytes(_binary_blob(triangles))
    scene = StlImporter().load(path)
    mesh = scene.meshes[0]
    assert mesh.positions.shape == (3, 3)
    np.testing.assert_allclose(mesh.positions[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(mesh.normals[0], [0.0, 0.0, 1.0])


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MalformedAssetError):
        StlImporter().load(tmp_path / "missing.stl")


def test_truncated_ascii_raises(tmp_path: Path) -> None:
    bad = "solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nendloop\nendfacet\n"
    path = tmp_path / "bad.stl"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(MalformedAssetError):
        StlImporter().load(path)


def test_empty_ascii_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.stl"
    path.write_text("solid x\nendsolid x\n", encoding="utf-8")
    with pytest.raises(MalformedAssetError):
        StlImporter().load(path)
