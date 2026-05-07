"""Qt smoke tests for the multi-track timeline dock."""
from __future__ import annotations

from pathlib import Path

import pytest
from pmx.importer import PmxImporter

from posecascade.animation.document import AnimationDocument
from posecascade.ui.multi_track_timeline import MultiTrackTimelineDock
from posecascade.ui.track_list_model import TrackEntry, TrackKind
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


def _scene_with_panel(tmp_path: Path):              # noqa: ANN201
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
                elements=(FixtureDisplayElement(kind=0, index=0),),
            ),
            FixtureDisplayFrame(
                name="表情",
                is_special=True,
                elements=(FixtureDisplayElement(kind=1, index=0),),
            ),
        ),
    )
    path = tmp_path / "panel.pmx"
    path.write_bytes(build_pmx(spec))
    return PmxImporter().load(path)


def _select_first_bone_entry(dock: MultiTrackTimelineDock, target: str) -> TrackEntry | None:
    """Walk the dock's tree and select the first child item matching ``target``."""
    tree = dock._tree                                # noqa: SLF001 — test seam
    for index in range(tree.topLevelItemCount()):
        group_item = tree.topLevelItem(index)
        for child_index in range(group_item.childCount()):
            child = group_item.child(child_index)
            if child.text(0) == target:
                tree.setCurrentItem(child)
                return child.data(0, 0x0100)
    return None


# ----- tree population ------------------------------------------------
def test_dock_builds_tree_from_document_and_scene(qapp: object, tmp_path: Path) -> None:
    scene = _scene_with_panel(tmp_path)
    dock = MultiTrackTimelineDock(document=AnimationDocument(), scene=scene)
    tree = dock._tree                                # noqa: SLF001
    top_names = {tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())}
    assert "Body" in top_names
    assert "表情" in top_names


def test_dock_refresh_picks_up_document_mutations(qapp: object, tmp_path: Path) -> None:
    """Manually inserting a keyframe + calling ``refresh_tree`` updates the
    keyframe-count column in the affected row."""
    scene = _scene_with_panel(tmp_path)
    document = AnimationDocument()
    dock = MultiTrackTimelineDock(document=document, scene=scene)
    dock.set_current_frame(0)
    _select_first_bone_entry(dock, "head")
    assert dock.insert_keyframe_at_current_frame() is True
    tree = dock._tree                                # noqa: SLF001
    body = next(
        tree.topLevelItem(i)
        for i in range(tree.topLevelItemCount())
        if tree.topLevelItem(i).text(0) == "Body"
    )
    head_row = body.child(0)
    assert head_row.text(1) == "1"


# ----- edit operations -------------------------------------------------
def test_insert_at_current_frame_routes_through_command_stack(
    qapp: object, tmp_path: Path,
) -> None:
    scene = _scene_with_panel(tmp_path)
    document = AnimationDocument()
    dock = MultiTrackTimelineDock(document=document, scene=scene)
    dock.set_current_frame(7)
    _select_first_bone_entry(dock, "head")
    dock.insert_keyframe_at_current_frame()
    assert document.find_bone_keyframe("head", 7) is not None
    assert dock.stack.can_undo()


def test_undo_after_insert_clears_keyframe(qapp: object, tmp_path: Path) -> None:
    scene = _scene_with_panel(tmp_path)
    document = AnimationDocument()
    dock = MultiTrackTimelineDock(document=document, scene=scene)
    dock.set_current_frame(3)
    _select_first_bone_entry(dock, "head")
    dock.insert_keyframe_at_current_frame()
    assert dock.undo() is True
    assert document.find_bone_keyframe("head", 3) is None


def test_redo_after_undo_re_executes(qapp: object, tmp_path: Path) -> None:
    scene = _scene_with_panel(tmp_path)
    document = AnimationDocument()
    dock = MultiTrackTimelineDock(document=document, scene=scene)
    dock.set_current_frame(2)
    _select_first_bone_entry(dock, "head")
    dock.insert_keyframe_at_current_frame()
    dock.undo()
    assert dock.redo() is True
    assert document.find_bone_keyframe("head", 2) is not None


def test_delete_selected_keyframe(qapp: object, tmp_path: Path) -> None:
    scene = _scene_with_panel(tmp_path)
    document = AnimationDocument()
    dock = MultiTrackTimelineDock(document=document, scene=scene)
    dock.set_current_frame(5)
    _select_first_bone_entry(dock, "head")
    dock.insert_keyframe_at_current_frame()
    assert dock.delete_selected_keyframe(5) is True
    assert document.find_bone_keyframe("head", 5) is None


def test_insert_returns_false_without_selection(qapp: object, tmp_path: Path) -> None:
    scene = _scene_with_panel(tmp_path)
    dock = MultiTrackTimelineDock(document=AnimationDocument(), scene=scene)
    dock.set_current_frame(0)
    # Nothing selected — operation must short-circuit.
    assert dock.insert_keyframe_at_current_frame() is False


def test_insert_handles_morph_track(qapp: object, tmp_path: Path) -> None:
    scene = _scene_with_panel(tmp_path)
    document = AnimationDocument()
    dock = MultiTrackTimelineDock(document=document, scene=scene)
    dock.set_current_frame(4)
    _select_first_bone_entry(dock, "smile")
    assert dock.insert_keyframe_at_current_frame() is True
    assert any(
        kf.morph_name == "smile" and kf.frame == 4
        for kf in document.morph_keyframes
    )


def test_set_current_frame_updates_label(qapp: object, tmp_path: Path) -> None:
    scene = _scene_with_panel(tmp_path)
    dock = MultiTrackTimelineDock(document=AnimationDocument(), scene=scene)
    dock.set_current_frame(42)
    assert dock._frame_label.text() == "Frame: 42"        # noqa: SLF001


# Keep the helpers reachable for IDE jumps.
__all__ = ["TrackKind", "pytest"]
