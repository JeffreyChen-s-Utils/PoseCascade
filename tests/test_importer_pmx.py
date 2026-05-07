"""Tests for the PMX 2.0 / 2.1 importer.

Builds tiny synthetic PMX byte buffers via :mod:`tests.fixtures.mmd.build`
(which uses ``struct`` directly, so a parser bug cannot silently agree
with the writer). Exercises every weight type, both text encodings, and
the cap / unsafe-path / malformed-magic refusal paths.
"""
from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pmx.importer import PmxImporter
from pmx.reader import parse_pmx
from pmx.types import (
    PmxBdef1,
    PmxBdef2,
    PmxBdef4,
    PmxDeformType,
    PmxSdef,
    PmxTextEncoding,
)

from posecascade.assets.importer_manager import ImporterManager
from posecascade.errors import MalformedAssetError
from tests.fixtures.mmd.build import (
    FixtureBuild,
    FixtureMaterial,
    FixtureVertex,
    _Bdef1,
    _Bdef2,
    _Bdef4,
    _Sdef,
    build_pmx,
    tiny_cube_spec,
)


# ----- fixtures ----------------------------------------------------------
@pytest.fixture
def tiny_pmx_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "mmd" / "tiny.pmx"


@pytest.fixture
def tiny_doc():
    return parse_pmx(build_pmx(tiny_cube_spec()))


# ----- happy path --------------------------------------------------------
def test_canonical_fixture_loads(tiny_pmx_path: Path) -> None:
    scene = PmxImporter().load(tiny_pmx_path)
    assert len(scene.meshes) == 1
    mesh = scene.meshes[0]
    assert mesh.positions.shape == (8, 3)
    assert mesh.indices.shape == (36,)
    assert mesh.joints_0 is not None
    assert mesh.weights_0 is not None
    assert mesh.base_color is not None
    assert mesh.base_color_texture_index == 0


def test_loads_via_importer_manager(tiny_pmx_path: Path) -> None:
    """Verify the discovery path works the same way the engine does at runtime."""
    importers_root = Path(__file__).resolve().parent.parent / "importers"
    manager = ImporterManager(importers_root=importers_root)
    manager.discover()
    importer = manager.importer_for(tiny_pmx_path)
    assert importer.__class__.__name__ == "PmxImporter"


def test_header_decodes_correctly(tiny_doc) -> None:
    h = tiny_doc.header
    assert h.version == pytest.approx(2.0)
    assert h.text_encoding == PmxTextEncoding.UTF8
    assert h.additional_uv_count == 0
    assert h.vertex_index_size == 2
    assert h.bone_index_size == 1


def test_names_round_trip(tiny_doc) -> None:
    assert tiny_doc.names.name_jp == "tiny"
    assert tiny_doc.names.name_en == "tiny"
    assert tiny_doc.names.comment_jp == ""


def test_scene_has_mesh_node_and_bone_root(tiny_pmx_path: Path) -> None:
    scene = PmxImporter().load(tiny_pmx_path).scene
    children = [child.name for child in scene.root.children]
    assert "tiny_mesh" in children
    assert "root" in children


# ----- text encoding switch ---------------------------------------------
def test_utf16_encoding(tmp_path: Path) -> None:
    spec = replace(tiny_cube_spec(), encoding_byte=0, name_jp="ミク", name_en="Miku")
    path = tmp_path / "u16.pmx"
    path.write_bytes(build_pmx(spec))
    scene = PmxImporter().load(path)
    assert scene.scene.name == "ミク"


def test_utf8_encoding_with_japanese(tmp_path: Path) -> None:
    spec = replace(tiny_cube_spec(), encoding_byte=1, name_jp="ミク")
    path = tmp_path / "u8.pmx"
    path.write_bytes(build_pmx(spec))
    doc = parse_pmx(path.read_bytes())
    assert doc.names.name_jp == "ミク"


# ----- skinning weight types --------------------------------------------
def test_bdef1_maps_to_full_weight_first_bone(tiny_doc) -> None:
    bdef1_vert = tiny_doc.vertices[0]
    assert bdef1_vert.deform_type == PmxDeformType.BDEF1
    assert isinstance(bdef1_vert.deform, PmxBdef1)


def test_bdef2_weight_complement(tiny_doc) -> None:
    bdef2_vert = tiny_doc.vertices[2]
    assert bdef2_vert.deform_type == PmxDeformType.BDEF2
    assert isinstance(bdef2_vert.deform, PmxBdef2)
    assert bdef2_vert.deform.weight1 == pytest.approx(0.5)


def test_bdef4_four_bones(tiny_doc) -> None:
    bdef4_vert = tiny_doc.vertices[4]
    assert bdef4_vert.deform_type == PmxDeformType.BDEF4
    assert isinstance(bdef4_vert.deform, PmxBdef4)
    assert sum(bdef4_vert.deform.weights) == pytest.approx(1.0)


def test_sdef_carries_correction_vectors(tiny_doc) -> None:
    sdef_vert = tiny_doc.vertices[6]
    assert sdef_vert.deform_type == PmxDeformType.SDEF
    assert isinstance(sdef_vert.deform, PmxSdef)
    assert sdef_vert.deform.c == (0.0, 0.5, 0.0)


def test_importer_flattens_sdef_to_bone_pair(tiny_pmx_path: Path) -> None:
    """SDEF retains its BDEF2-equivalent bone pair on import, so skinning
    still produces visually plausible output before SDEF correction lands."""
    scene = PmxImporter().load(tiny_pmx_path)
    mesh = scene.meshes[0]
    assert mesh.joints_0[6].tolist() == [0, 1, 0, 0]
    np.testing.assert_allclose(mesh.weights_0[6], [0.5, 0.5, 0.0, 0.0])


def test_importer_flattens_bdef4(tiny_pmx_path: Path) -> None:
    scene = PmxImporter().load(tiny_pmx_path)
    mesh = scene.meshes[0]
    np.testing.assert_allclose(mesh.weights_0[4], [0.7, 0.3, 0.0, 0.0])


# ----- bones --------------------------------------------------------------
def test_bone_root_has_negative_parent(tiny_doc) -> None:
    assert tiny_doc.bones[0].parent_index == -1
    assert tiny_doc.bones[1].parent_index == 0


def test_bone_node_local_translation_is_relative_to_parent(tiny_pmx_path: Path) -> None:
    """The PMX rest pose stores world bone positions; our scene-graph
    nodes hold *local* TRS so the world matrix collapses correctly when the
    parent later moves."""
    scene = PmxImporter().load(tiny_pmx_path)
    root_bone = next(c for c in scene.scene.root.children if c.name == "root")
    child_bone = root_bone.children[0]
    np.testing.assert_allclose(root_bone.transform.translation, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(child_bone.transform.translation, [0.0, 1.0, 0.0])


def test_skin_inverse_bind_translates_back_to_origin(tiny_pmx_path: Path) -> None:
    scene = PmxImporter().load(tiny_pmx_path)
    skin = scene.skins[0]
    # Bone 1 sits at (0, 1, 0) in world rest. Its IBM pulls back to origin,
    # so multiplying the rest world matrix by the IBM gives identity.
    np.testing.assert_allclose(skin.inverse_bind_matrices[1, :3, 3], [0.0, -1.0, 0.0])


# ----- materials / textures ----------------------------------------------
def test_material_face_index_count_matches_total(tiny_doc) -> None:
    total = sum(m.face_index_count for m in tiny_doc.materials)
    assert total == len(tiny_doc.indices)


def test_material_split_creates_one_mesh_per_material(tmp_path: Path) -> None:
    """Two materials, each consuming half the index buffer, produce two Meshes."""
    spec = tiny_cube_spec()
    half = len(spec.indices) // 2
    materials = (
        FixtureMaterial(name_jp="lower", texture_index=0, face_index_count=half),
        FixtureMaterial(
            name_jp="upper",
            texture_index=0,
            face_index_count=len(spec.indices) - half,
        ),
    )
    path = tmp_path / "two_mats.pmx"
    path.write_bytes(build_pmx(replace(spec, materials=materials)))
    scene = PmxImporter().load(path)
    assert len(scene.meshes) == 2
    assert scene.meshes[0].name == "lower"
    assert scene.meshes[1].name == "upper"
    assert scene.meshes[0].indices.size + scene.meshes[1].indices.size == len(spec.indices)


def test_external_texture_resolves_through_path_safety(tmp_path: Path) -> None:
    pixels = np.full((4, 4, 4), [200, 100, 50, 255], dtype=np.uint8)
    Image.fromarray(pixels, mode="RGBA").save(tmp_path / "diffuse.png")
    spec = replace(tiny_cube_spec(), textures=("diffuse.png",))
    pmx_path = tmp_path / "model.pmx"
    pmx_path.write_bytes(build_pmx(spec))
    scene = PmxImporter().load(pmx_path)
    np.testing.assert_array_equal(scene.textures[0].pixels[0, 0], [200, 100, 50, 255])


def test_path_traversal_falls_back_to_placeholder(tmp_path: Path) -> None:
    spec = replace(tiny_cube_spec(), textures=("../etc/passwd",))
    pmx_path = tmp_path / "evil.pmx"
    pmx_path.write_bytes(build_pmx(spec))
    scene = PmxImporter().load(pmx_path)
    assert scene.textures[0].pixels.shape == (1, 1, 4)
    assert scene.textures[0].pixels[0, 0, 0] == 255


def test_missing_texture_falls_back_to_placeholder(tmp_path: Path) -> None:
    spec = replace(tiny_cube_spec(), textures=("does_not_exist.png",))
    pmx_path = tmp_path / "missing.pmx"
    pmx_path.write_bytes(build_pmx(spec))
    scene = PmxImporter().load(pmx_path)
    assert scene.textures[0].pixels.shape == (1, 1, 4)


def test_no_diffuse_texture_yields_none_index(tmp_path: Path) -> None:
    spec = tiny_cube_spec()
    materials = (replace(spec.materials[0], texture_index=-1),)
    path = tmp_path / "no_tex.pmx"
    path.write_bytes(build_pmx(replace(spec, materials=materials)))
    scene = PmxImporter().load(path)
    assert scene.meshes[0].base_color_texture_index is None


# ----- malformed inputs --------------------------------------------------
def test_wrong_magic_raises(tmp_path: Path) -> None:
    bad = b"FBX\x20" + build_pmx(tiny_cube_spec())[4:]
    path = tmp_path / "bad.pmx"
    path.write_bytes(bad)
    with pytest.raises(MalformedAssetError, match="not a PMX file"):
        PmxImporter().load(path)


def test_unsupported_version_raises(tmp_path: Path) -> None:
    bytes_v3 = bytearray(build_pmx(tiny_cube_spec()))
    # Replace the version float (4 bytes after magic) with 3.0.
    bytes_v3[4:8] = struct.pack("<f", 3.0)
    path = tmp_path / "v3.pmx"
    path.write_bytes(bytes(bytes_v3))
    with pytest.raises(MalformedAssetError, match="unsupported PMX version"):
        PmxImporter().load(path)


def test_truncated_file_raises(tmp_path: Path) -> None:
    full = build_pmx(tiny_cube_spec())
    path = tmp_path / "trunc.pmx"
    path.write_bytes(full[: len(full) // 2])
    with pytest.raises(MalformedAssetError, match="unexpected end of file"):
        PmxImporter().load(path)


def test_face_index_out_of_range_raises(tmp_path: Path) -> None:
    spec = replace(
        tiny_cube_spec(),
        indices=(99, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6,
                 0, 4, 5, 0, 5, 1, 2, 6, 7, 2, 7, 3,
                 1, 5, 6, 1, 6, 2, 0, 3, 7, 0, 7, 4),
    )
    path = tmp_path / "bad_index.pmx"
    path.write_bytes(build_pmx(spec))
    with pytest.raises(MalformedAssetError, match="out of range"):
        PmxImporter().load(path)


def test_face_index_count_not_multiple_of_three_raises(tmp_path: Path) -> None:
    spec = replace(
        tiny_cube_spec(),
        indices=(0, 1),
        materials=(replace(tiny_cube_spec().materials[0], face_index_count=2),),
    )
    path = tmp_path / "bad_facecount.pmx"
    path.write_bytes(build_pmx(spec))
    with pytest.raises(MalformedAssetError, match="not a multiple of 3"):
        PmxImporter().load(path)


def test_minimum_geometry_one_triangle(tmp_path: Path) -> None:
    """Edge case: smallest valid PMX (1 triangle, 3 verts, 1 bone, 1 mat)."""
    indices = (0, 1, 2)
    spec = FixtureBuild(
        name_jp="t", name_en="t",
        vertices=(
            FixtureVertex(position=(0.0, 0.0, 0.0), deform=_Bdef1(bone=0)),
            FixtureVertex(position=(1.0, 0.0, 0.0), deform=_Bdef1(bone=0)),
            FixtureVertex(position=(0.0, 1.0, 0.0), deform=_Bdef1(bone=0)),
        ),
        indices=indices,
        materials=(FixtureMaterial(name_jp="m", face_index_count=3),),
        bones=(),
    )
    path = tmp_path / "min.pmx"
    path.write_bytes(build_pmx(spec))
    scene = PmxImporter().load(path)
    assert scene.meshes[0].indices.size == 3
    # No bones → no skin
    assert scene.skins == ()


# ----- internal-toon vs external-toon -----------------------------------
def test_internal_toon_reference_is_kept(tiny_doc) -> None:
    mat = tiny_doc.materials[0]
    # tiny_cube_spec sets toon_mode=1 (internal) with reference 0 (toon01.bmp).
    assert int(mat.toon_mode) == 1
    assert mat.toon_reference == 0


def test_external_toon_reads_signed_index(tmp_path: Path) -> None:
    spec = tiny_cube_spec()
    materials = (replace(spec.materials[0], toon_mode=0, toon_reference=0),)
    path = tmp_path / "external_toon.pmx"
    path.write_bytes(build_pmx(replace(spec, materials=materials)))
    doc = parse_pmx(path.read_bytes())
    assert int(doc.materials[0].toon_mode) == 0
    assert doc.materials[0].toon_reference == 0


# ----- bonus: keep the helper imports load-bearing for IDE jumps ---------
__all__ = [
    "_Bdef1",
    "_Bdef2",
    "_Bdef4",
    "_Sdef",
]
