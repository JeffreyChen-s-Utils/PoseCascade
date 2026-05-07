"""Tests for :class:`AnimationDocument` + the undo / redo command stack."""
from __future__ import annotations

from pathlib import Path

import pytest
from vmd.reader import parse_vmd
from vmd.types import (
    VmdBoneKeyframe,
    VmdCameraKeyframe,
    VmdMorphKeyframe,
)
from vmd.writer import serialize_vmd

from posecascade.animation.commands import (
    CommandStack,
    CompoundCommand,
    DeleteBoneKeyframe,
    InsertBoneKeyframe,
    InsertCameraKeyframe,
    InsertLightKeyframe,
    InsertMorphKeyframe,
    InsertSelfShadowKeyframe,
)
from posecascade.animation.document import AnimationDocument

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mmd"


def _document_from_wave() -> AnimationDocument:
    return AnimationDocument.from_motion(parse_vmd((_FIXTURES / "wave.vmd").read_bytes()))


def _bone_kf(bone_name: str = "child", frame: int = 0) -> VmdBoneKeyframe:
    return VmdBoneKeyframe(
        bone_name=bone_name,
        frame=frame,
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        bezier_handles=tuple((20, 20, 107, 107) for _ in range(4)),
    )


# ----- document round-trip --------------------------------------------
def test_document_from_motion_round_trips_via_writer() -> None:
    """Loading a fixture into a document and emitting via to_motion + writer
    must reproduce the original bytes."""
    data = (_FIXTURES / "wave.vmd").read_bytes()
    doc = AnimationDocument.from_motion(parse_vmd(data))
    assert serialize_vmd(doc.to_motion()) == data


def test_empty_document_to_motion_emits_zero_count_sections() -> None:
    doc = AnimationDocument()
    motion = doc.to_motion()
    assert motion.bone_keyframes == ()
    assert motion.camera_keyframes == ()


# ----- bone keyframe ops ----------------------------------------------
def test_insert_bone_keyframe_appends() -> None:
    doc = AnimationDocument()
    doc.insert_bone_keyframe(_bone_kf(frame=0))
    doc.insert_bone_keyframe(_bone_kf(frame=10))
    assert len(doc.bone_keyframes) == 2


def test_insert_bone_keyframe_overwrites_existing() -> None:
    doc = AnimationDocument()
    doc.insert_bone_keyframe(_bone_kf(frame=0))
    replacement = VmdBoneKeyframe(
        bone_name="child", frame=0,
        position=(1.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        bezier_handles=tuple((20, 20, 107, 107) for _ in range(4)),
    )
    doc.insert_bone_keyframe(replacement)
    assert len(doc.bone_keyframes) == 1
    assert doc.bone_keyframes[0].position == (1.0, 0.0, 0.0)


def test_delete_bone_keyframe_returns_removed_record() -> None:
    doc = AnimationDocument()
    kf = _bone_kf(frame=5)
    doc.insert_bone_keyframe(kf)
    removed = doc.delete_bone_keyframe("child", 5)
    assert removed == kf
    assert doc.find_bone_keyframe("child", 5) is None


def test_delete_bone_keyframe_missing_returns_none() -> None:
    doc = AnimationDocument()
    assert doc.delete_bone_keyframe("ghost", 0) is None


# ----- morph + camera + light + self-shadow ops -----------------------
def test_insert_morph_keyframe_overwrites_same_name_frame() -> None:
    doc = AnimationDocument()
    doc.insert_morph_keyframe(VmdMorphKeyframe(morph_name="smile", frame=0, weight=0.0))
    doc.insert_morph_keyframe(VmdMorphKeyframe(morph_name="smile", frame=0, weight=1.0))
    assert len(doc.morph_keyframes) == 1
    assert doc.morph_keyframes[0].weight == pytest.approx(1.0)


def test_camera_keyframes_keyed_only_on_frame() -> None:
    doc = AnimationDocument()
    doc.insert_camera_keyframe(_camera_kf(frame=0, fov=30))
    doc.insert_camera_keyframe(_camera_kf(frame=0, fov=60))
    assert len(doc.camera_keyframes) == 1
    assert doc.camera_keyframes[0].fov_degrees == 60


# ----- command stack --------------------------------------------------
def test_command_stack_push_executes_and_records_undo() -> None:
    doc = AnimationDocument()
    stack = CommandStack()
    cmd = InsertBoneKeyframe(document=doc, keyframe=_bone_kf())
    stack.push(cmd)
    assert len(doc.bone_keyframes) == 1
    assert stack.can_undo()
    assert not stack.can_redo()


def test_command_stack_undo_reverts() -> None:
    doc = AnimationDocument()
    stack = CommandStack()
    stack.push(InsertBoneKeyframe(document=doc, keyframe=_bone_kf()))
    stack.undo()
    assert doc.bone_keyframes == []
    assert stack.can_redo()


def test_command_stack_redo_re_executes() -> None:
    doc = AnimationDocument()
    stack = CommandStack()
    stack.push(InsertBoneKeyframe(document=doc, keyframe=_bone_kf()))
    stack.undo()
    stack.redo()
    assert len(doc.bone_keyframes) == 1


def test_command_stack_new_push_clears_redo() -> None:
    doc = AnimationDocument()
    stack = CommandStack()
    stack.push(InsertBoneKeyframe(document=doc, keyframe=_bone_kf(frame=0)))
    stack.undo()
    assert stack.can_redo()
    stack.push(InsertBoneKeyframe(document=doc, keyframe=_bone_kf(frame=10)))
    assert not stack.can_redo()


def test_insert_bone_undo_restores_previous_keyframe_when_overwriting() -> None:
    """Re-inserting on the same (bone, frame) must restore the original on
    undo, not just leave an empty slot."""
    doc = AnimationDocument()
    original = _bone_kf(frame=0)
    doc.insert_bone_keyframe(original)
    stack = CommandStack()
    replacement = VmdBoneKeyframe(
        bone_name="child", frame=0,
        position=(5.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        bezier_handles=tuple((20, 20, 107, 107) for _ in range(4)),
    )
    stack.push(InsertBoneKeyframe(document=doc, keyframe=replacement))
    stack.undo()
    assert doc.bone_keyframes == [original]


def test_delete_bone_undo_restores_record() -> None:
    doc = _document_from_wave()
    initial_count = len(doc.bone_keyframes)
    stack = CommandStack()
    stack.push(DeleteBoneKeyframe(document=doc, bone_name="child", frame=5))
    assert len(doc.bone_keyframes) == initial_count - 1
    stack.undo()
    assert len(doc.bone_keyframes) == initial_count


def test_compound_command_runs_children_in_order_and_undoes_in_reverse() -> None:
    doc = AnimationDocument()
    stack = CommandStack()
    children = (
        InsertBoneKeyframe(document=doc, keyframe=_bone_kf(frame=0)),
        InsertBoneKeyframe(document=doc, keyframe=_bone_kf(frame=10)),
        InsertBoneKeyframe(document=doc, keyframe=_bone_kf(frame=20)),
    )
    stack.push(CompoundCommand(children=children))
    assert len(doc.bone_keyframes) == 3
    stack.undo()
    assert doc.bone_keyframes == []


# ----- coverage for the remaining insert commands ---------------------
def _camera_kf(frame: int, fov: int = 30) -> VmdCameraKeyframe:
    return VmdCameraKeyframe(
        frame=frame,
        distance=-30.0,
        target=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        bezier_handles=tuple((20, 20, 107, 107) for _ in range(6)),
        fov_degrees=fov,
        perspective_off=False,
    )


def test_insert_camera_keyframe_command_round_trips() -> None:
    doc = AnimationDocument()
    stack = CommandStack()
    stack.push(InsertCameraKeyframe(document=doc, keyframe=_camera_kf(frame=0)))
    assert doc.camera_keyframes
    stack.undo()
    assert doc.camera_keyframes == []


def test_insert_light_keyframe_command_round_trips() -> None:
    from vmd.types import VmdLightKeyframe  # noqa: PLC0415
    doc = AnimationDocument()
    stack = CommandStack()
    stack.push(
        InsertLightKeyframe(
            document=doc,
            keyframe=VmdLightKeyframe(
                frame=0,
                color=(1.0, 1.0, 1.0),
                direction=(0.0, -1.0, 0.0),
            ),
        ),
    )
    assert doc.light_keyframes
    stack.undo()
    assert doc.light_keyframes == []


def test_insert_morph_keyframe_command_round_trips() -> None:
    doc = AnimationDocument()
    stack = CommandStack()
    stack.push(
        InsertMorphKeyframe(
            document=doc,
            keyframe=VmdMorphKeyframe(morph_name="smile", frame=0, weight=1.0),
        ),
    )
    assert doc.morph_keyframes
    stack.undo()
    assert doc.morph_keyframes == []


def test_insert_self_shadow_keyframe_command_round_trips() -> None:
    from vmd.types import VmdSelfShadowKeyframe, VmdSelfShadowMode  # noqa: PLC0415
    doc = AnimationDocument()
    stack = CommandStack()
    stack.push(
        InsertSelfShadowKeyframe(
            document=doc,
            keyframe=VmdSelfShadowKeyframe(
                frame=0, mode=VmdSelfShadowMode.ON, distance=0.0,
            ),
        ),
    )
    assert doc.self_shadow_keyframes
    stack.undo()
    assert doc.self_shadow_keyframes == []
