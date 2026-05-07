"""Tests for PMX display-frame extraction + the Track-list model."""
from __future__ import annotations

from pathlib import Path

import pytest
from pmx.importer import PmxImporter

from posecascade.animation.display_frames import (
    DisplayFrameElementKind,
)
from posecascade.animation.document import AnimationDocument
from posecascade.ui.track_list_model import (
    TrackKind,
    build_track_list,
)
from tests.fixtures.mmd.build import (
    FixtureBone,
    FixtureBuild,
    FixtureDisplayElement,
    FixtureDisplayFrame,
    FixtureMaterial,
    FixtureMorph,
    FixtureVertex,
    FixtureVertexMorphOffset,
    _Bdef1,
    build_pmx,
)


def _build_scene_with_groups(tmp_path: Path):       # noqa: ANN201 — internal helper
    spec = FixtureBuild(
        name_jp="r", name_en="r",
        vertices=(
            FixtureVertex(position=(-1, -1, -1), deform=_Bdef1(bone=0)),
            FixtureVertex(position=(1, -1, -1), deform=_Bdef1(bone=0)),
            FixtureVertex(position=(0, 1, 0), deform=_Bdef1(bone=0)),
        ),
        indices=(0, 2, 1),
        materials=(FixtureMaterial(name_jp="m", face_index_count=3),),
        bones=(
            FixtureBone(name_jp="head", position=(0, 1, 0)),
            FixtureBone(name_jp="arm", position=(1, 1, 0)),
            FixtureBone(name_jp="leg", position=(0, 0, 0)),
        ),
        morphs=(
            FixtureMorph(
                name="smile", morph_type=1,
                vertex_offsets=(
                    FixtureVertexMorphOffset(vertex_index=0, offset=(0.1, 0.0, 0.0)),
                ),
            ),
        ),
        display_frames=(
            FixtureDisplayFrame(
                name="Body",
                elements=(
                    FixtureDisplayElement(kind=0, index=0),    # head
                    FixtureDisplayElement(kind=0, index=1),    # arm
                ),
            ),
            FixtureDisplayFrame(
                name="表情",
                is_special=True,
                elements=(FixtureDisplayElement(kind=1, index=0),),    # smile
            ),
        ),
    )
    path = tmp_path / "groups.pmx"
    path.write_bytes(build_pmx(spec))
    return PmxImporter().load(path)


# ----- importer extraction --------------------------------------------
def test_pmx_importer_extracts_display_frame_groups(tmp_path: Path) -> None:
    scene = _build_scene_with_groups(tmp_path)
    assert len(scene.display_frame_groups) == 2
    body, face = scene.display_frame_groups
    assert body.name == "Body"
    assert face.name == "表情"
    assert face.is_special is True
    assert len(body.elements) == 2
    assert body.elements[0].kind == DisplayFrameElementKind.BONE
    assert body.elements[0].index == 0
    assert face.elements[0].kind == DisplayFrameElementKind.MORPH


def test_pmx_importer_keeps_groups_in_authored_order(tmp_path: Path) -> None:
    scene = _build_scene_with_groups(tmp_path)
    names = [group.name for group in scene.display_frame_groups]
    assert names == ["Body", "表情"]


def test_pmx_with_no_panels_has_empty_display_frame_groups() -> None:
    scene = PmxImporter().load(
        Path(__file__).resolve().parent / "fixtures" / "mmd" / "tiny.pmx",
    )
    assert scene.display_frame_groups == ()


# ----- track list model ------------------------------------------------
def test_track_list_groups_match_pmx_panels(tmp_path: Path) -> None:
    scene = _build_scene_with_groups(tmp_path)
    document = AnimationDocument()
    groups = build_track_list(document, scene)
    names = [group.name for group in groups]
    assert "Body" in names
    assert "表情" in names


def test_track_list_entries_resolve_bone_and_morph_names(tmp_path: Path) -> None:
    scene = _build_scene_with_groups(tmp_path)
    document = AnimationDocument()
    groups = build_track_list(document, scene)
    body = next(group for group in groups if group.name == "Body")
    bone_names = sorted(entry.display_name for entry in body.entries)
    assert bone_names == ["arm", "head"]
    face = next(group for group in groups if group.name == "表情")
    morph_names = [entry.display_name for entry in face.entries]
    assert morph_names == ["smile"]


def test_track_list_keyframe_counts_reflect_document(tmp_path: Path) -> None:
    """Adding a bone keyframe must show up in the corresponding entry's count."""
    from vmd.types import VmdBoneKeyframe  # noqa: PLC0415
    scene = _build_scene_with_groups(tmp_path)
    document = AnimationDocument()
    document.insert_bone_keyframe(
        VmdBoneKeyframe(
            bone_name="head", frame=0,
            position=(0, 0, 0), rotation=(0, 0, 0, 1),
            bezier_handles=tuple((20, 20, 107, 107) for _ in range(4)),
        ),
    )
    groups = build_track_list(document, scene)
    body = next(group for group in groups if group.name == "Body")
    head_entry = next(entry for entry in body.entries if entry.display_name == "head")
    assert head_entry.keyframe_count == 1


def test_track_list_synthetic_camera_group_only_when_keyframes_exist() -> None:
    from vmd.types import VmdCameraKeyframe  # noqa: PLC0415
    document = AnimationDocument()
    assert all(group.name != "Camera / Light" for group in build_track_list(document))
    document.insert_camera_keyframe(
        VmdCameraKeyframe(
            frame=0, distance=-30.0,
            target=(0, 0, 0), rotation=(0, 0, 0),
            bezier_handles=tuple((20, 20, 107, 107) for _ in range(6)),
            fov_degrees=30, perspective_off=False,
        ),
    )
    groups = build_track_list(document)
    scene_group = next(group for group in groups if group.name == "Camera / Light")
    assert any(entry.kind == TrackKind.CAMERA for entry in scene_group.entries)


def test_track_list_falls_back_to_other_for_unknown_bones(tmp_path: Path) -> None:
    """A VMD-imported document for a bone not in any panel must surface
    in the synthetic ``Other`` group instead of vanishing."""
    from vmd.types import VmdBoneKeyframe  # noqa: PLC0415
    scene = _build_scene_with_groups(tmp_path)
    document = AnimationDocument()
    document.insert_bone_keyframe(
        VmdBoneKeyframe(
            bone_name="leg",                      # exists on model, NOT in any panel
            frame=0, position=(0, 0, 0), rotation=(0, 0, 0, 1),
            bezier_handles=tuple((20, 20, 107, 107) for _ in range(4)),
        ),
    )
    groups = build_track_list(document, scene)
    other = next(group for group in groups if group.name == "Other")
    assert any(
        entry.display_name == "leg" and entry.kind == TrackKind.BONE
        for entry in other.entries
    )


# Pytest stays explicitly imported so editors find the parametrize utility.
__all__ = ["pytest"]
