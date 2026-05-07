"""Tests for the legacy PMD importer.

Builds tiny synthetic PMD bytes via :mod:`tests.fixtures.mmd.build` and
exercises the SJIS / sentinel-parent / BDEF2-only / sphere-suffix branches
that distinguish PMD from PMX.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from pmd.importer import PmdImporter
from pmd.reader import parse_pmd
from pmx.types import PmxBdef2, PmxDeformType

from posecascade.errors import MalformedAssetError
from tests.fixtures.mmd.build import build_pmd_tiny


@pytest.fixture
def tiny_pmd_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "mmd" / "tiny.pmd"


def test_canonical_pmd_loads(tiny_pmd_path: Path) -> None:
    scene = PmdImporter().load(tiny_pmd_path)
    assert len(scene.meshes) == 1
    mesh = scene.meshes[0]
    assert mesh.positions.shape == (8, 3)
    assert mesh.indices.shape == (36,)
    assert len(scene.skins) == 1
    assert [bone.name for bone in scene.skins[0].joints] == ["root", "child"]


def test_pmd_vertex_with_same_bone_pair_is_bdef1() -> None:
    doc = parse_pmd(build_pmd_tiny())
    # Vertex 0 in the fixture uses bone1 == bone2 == 0 → BDEF1.
    vertex0 = doc.vertices[0]
    assert vertex0.deform_type == PmxDeformType.BDEF1
    assert isinstance(vertex0.deform, PmxBdef2)        # storage stays BDEF2-shaped
    assert vertex0.deform.weight1 == pytest.approx(1.0)


def test_pmd_vertex_with_distinct_bones_is_bdef2() -> None:
    doc = parse_pmd(build_pmd_tiny())
    vertex2 = doc.vertices[2]
    assert vertex2.deform_type == PmxDeformType.BDEF2
    assert vertex2.deform.weight1 == pytest.approx(0.5)


def test_pmd_root_parent_sentinel_maps_to_minus_one() -> None:
    doc = parse_pmd(build_pmd_tiny())
    assert doc.bones[0].parent_index == -1
    assert doc.bones[1].parent_index == 0


def test_pmd_diffuse_texture_path_kept(tiny_pmd_path: Path) -> None:
    scene = PmdImporter().load(tiny_pmd_path)
    assert scene.textures[0].name == "diffuse.png"


def test_pmd_wrong_magic_raises(tmp_path: Path) -> None:
    bytes_data = bytearray(build_pmd_tiny())
    bytes_data[:3] = b"FBX"
    path = tmp_path / "bad.pmd"
    path.write_bytes(bytes(bytes_data))
    with pytest.raises(MalformedAssetError, match="not a PMD file"):
        PmdImporter().load(path)


def test_pmd_unsupported_version_raises(tmp_path: Path) -> None:
    bytes_data = bytearray(build_pmd_tiny())
    bytes_data[3:7] = struct.pack("<f", 9.0)
    path = tmp_path / "v9.pmd"
    path.write_bytes(bytes(bytes_data))
    with pytest.raises(MalformedAssetError, match="unsupported PMD version"):
        PmdImporter().load(path)


def test_pmd_meshes_share_vertex_buffer() -> None:
    """PMD has one material per file in our fixture; sanity-check that the
    importer still produces full per-vertex skinning arrays even though the
    PMD weight format differs from PMX (uint8 % weight)."""
    scene = PmdImporter().load(
        Path(__file__).resolve().parent / "fixtures" / "mmd" / "tiny.pmd"
    )
    mesh = scene.meshes[0]
    assert mesh.joints_0 is not None
    np.testing.assert_allclose(mesh.weights_0[2], [0.5, 0.5, 0.0, 0.0])
