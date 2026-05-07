"""Tests for the VPD pose importer + writer + apply helper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pmx.importer import PmxImporter
from vpd.importer import VpdImporter
from vpd.reader import parse_vpd, parse_vpd_bytes
from vpd.types import VpdBoneOverride, VpdMorphOverride, VpdPose

from posecascade.animation.vpd_apply import apply_pose
from posecascade.animation.vpd_writer import serialize_vpd, serialize_vpd_bytes
from posecascade.errors import MalformedAssetError
from tests.fixtures.mmd.build import build_vpd_bytes, build_vpd_text


# ----- reader -----------------------------------------------------------
def test_bone_only_pose_parses() -> None:
    text = build_vpd_text(
        model_name="miku.osm",
        bones=(
            ("head", (0.0, 0.1, 0.0), (0.0, 0.0, 0.0, 1.0)),
            ("hand", (0.5, 0.0, 0.0), (0.1, 0.2, 0.3, 0.95)),
        ),
    )
    pose = parse_vpd(text)
    assert pose.model_name == "miku.osm"
    assert len(pose.bones) == 2
    assert pose.morphs == ()
    assert pose.bones[0].name == "head"
    assert pose.bones[1].translation == pytest.approx((0.5, 0.0, 0.0))


def test_morph_only_pose_parses() -> None:
    text = build_vpd_text(
        model_name="miku.osm",
        morphs=(("smile", 0.6), ("blink", 1.0)),
    )
    pose = parse_vpd(text)
    assert pose.bones == ()
    assert len(pose.morphs) == 2
    assert pose.morphs[0].name == "smile"
    assert pose.morphs[1].weight == pytest.approx(1.0)


def test_mixed_pose_parses_in_either_order() -> None:
    text = build_vpd_text(
        model_name="m.osm",
        bones=(("head", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
        morphs=(("smile", 0.3),),
    )
    pose = parse_vpd(text)
    assert len(pose.bones) == 1
    assert len(pose.morphs) == 1


def test_comments_are_stripped() -> None:
    text = (
        "Vocaloid Pose Data file\n"
        "; this is a comment\n"
        "\n"
        "m.osm;\n"
        "1;\n"
        "Bone0{head ; trailing comment\n"
        "  0.0,1.0,0.0; trans\n"
        "  0.0,0.0,0.0,1.0; rotation\n"
        "}\n"
    )
    pose = parse_vpd(text)
    assert pose.bones[0].name == "head"
    assert pose.bones[0].translation == pytest.approx((0.0, 1.0, 0.0))


def test_japanese_bone_name_round_trips_through_sjis() -> None:
    """A bone whose name uses kanji must round-trip ``parse → serialize → parse``."""
    pose = VpdPose(
        model_name="miku.osm",
        bones=(
            VpdBoneOverride(
                name="左腕",
                translation=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
    )
    sjis = serialize_vpd_bytes(pose)
    parsed = parse_vpd_bytes(sjis)
    assert parsed.bones[0].name == "左腕"


def test_missing_header_raises() -> None:
    with pytest.raises(MalformedAssetError, match="VPD header"):
        parse_vpd("not_a_vpd_file\n")


def test_empty_file_raises() -> None:
    with pytest.raises(MalformedAssetError, match="empty"):
        parse_vpd("")


# ----- writer -----------------------------------------------------------
def test_serialize_emits_canonical_layout() -> None:
    pose = VpdPose(
        model_name="m.osm",
        bones=(
            VpdBoneOverride(
                name="head", translation=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
    )
    text = serialize_vpd(pose)
    assert text.startswith("Vocaloid Pose Data file")
    assert "m.osm;" in text
    assert "Bone0{head" in text


def test_round_trip_preserves_values() -> None:
    pose = VpdPose(
        model_name="r.osm",
        bones=(
            VpdBoneOverride(
                name="hand", translation=(0.123456, -0.5, 0.0),
                rotation=(0.1, 0.2, 0.3, 0.95),
            ),
        ),
        morphs=(VpdMorphOverride(name="smile", weight=0.42),),
    )
    again = parse_vpd(serialize_vpd(pose))
    assert again.bones[0].name == pose.bones[0].name
    assert again.bones[0].translation == pytest.approx(pose.bones[0].translation, abs=1e-5)
    assert again.bones[0].rotation == pytest.approx(pose.bones[0].rotation, abs=1e-5)
    assert again.morphs[0].weight == pytest.approx(pose.morphs[0].weight, abs=1e-5)


# ----- importer -------------------------------------------------------
def test_vpd_importer_reads_disk_file(tmp_path: Path) -> None:
    path = tmp_path / "pose.vpd"
    path.write_bytes(build_vpd_bytes(
        bones=(("head", (0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
    ))
    pose = VpdImporter().load(path)
    assert pose.bones[0].name == "head"


def test_vpd_importer_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MalformedAssetError, match="VPD file not found"):
        VpdImporter().load(tmp_path / "nope.vpd")


# ----- apply ----------------------------------------------------------
_TINY_PMX = Path(__file__).resolve().parent / "fixtures" / "mmd" / "tiny.pmx"
_TINY_MORPHS_PMX = Path(__file__).resolve().parent / "fixtures" / "mmd" / "tiny_morphs.pmx"


def test_apply_pose_writes_bone_node_translation_and_rotation() -> None:
    scene = PmxImporter().load(_TINY_PMX)
    pose = VpdPose(
        model_name="tiny",
        bones=(
            VpdBoneOverride(
                name="child", translation=(0.5, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.7071068, 0.7071068),
            ),
        ),
    )
    result = apply_pose(pose, scene)
    assert result.bones_applied == 1
    assert result.bones_skipped == ()
    child = scene.skins[0].joints[1]
    np.testing.assert_allclose(child.transform.rotation, [0, 0, 0.7071068, 0.7071068], atol=1e-5)
    # Rest translation for ``child`` is (0, 1, 0); override adds (0.5, 0, 0).
    np.testing.assert_allclose(child.transform.translation, [0.5, 1.0, 0.0], atol=1e-5)


def test_apply_pose_silently_skips_missing_bones() -> None:
    scene = PmxImporter().load(_TINY_PMX)
    pose = VpdPose(
        model_name="tiny",
        bones=(
            VpdBoneOverride(
                name="missing_bone", translation=(1.0, 0, 0),
                rotation=(0, 0, 0, 1),
            ),
        ),
    )
    result = apply_pose(pose, scene)
    assert result.bones_applied == 0
    assert "missing_bone" in result.bones_skipped


def test_apply_pose_writes_morph_state() -> None:
    scene = PmxImporter().load(_TINY_MORPHS_PMX)
    pose = VpdPose(
        model_name="tiny",
        morphs=(VpdMorphOverride(name="vert_pull", weight=1.0),),
    )
    result = apply_pose(pose, scene)
    assert result.morphs_applied == 1
    assert result.morphs_skipped == ()


def test_apply_pose_skips_unknown_morphs() -> None:
    scene = PmxImporter().load(_TINY_MORPHS_PMX)
    pose = VpdPose(
        model_name="tiny",
        morphs=(VpdMorphOverride(name="not_a_morph", weight=1.0),),
    )
    result = apply_pose(pose, scene)
    assert result.morphs_applied == 0
    assert "not_a_morph" in result.morphs_skipped
