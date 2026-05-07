"""Round-trip tests for the VMD writer."""
from __future__ import annotations

from pathlib import Path

import pytest
from vmd.reader import parse_vmd
from vmd.writer import serialize_vmd

from tests.fixtures.mmd.build import (
    build_vmd_camera_motion,
    build_vmd_morphs,
    build_vmd_wave,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mmd"


def _round_trip(data: bytes) -> bytes:
    return serialize_vmd(parse_vmd(data))


def test_wave_vmd_round_trips_byte_identical() -> None:
    """Bone keyframes only — the canonical Phase 3 fixture."""
    data = (_FIXTURES / "wave.vmd").read_bytes()
    assert _round_trip(data) == data


def test_morphs_vmd_round_trips_byte_identical() -> None:
    """Morph keyframes only — the Phase 4 fixture."""
    data = (_FIXTURES / "morphs.vmd").read_bytes()
    assert _round_trip(data) == data


def test_camera_motion_round_trips_byte_identical() -> None:
    """Camera + light + self-shadow keyframes — Phase 8 territory."""
    data = build_vmd_camera_motion(
        camera_keyframes=(
            (0, -30.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 30, False),
            (60, -45.0, (1.0, 1.0, 0.0), (0.1, 0.2, 0.3), 60, True),
        ),
        light_keyframes=(
            (0, (1.0, 1.0, 1.0), (-0.5, -1.0, 0.5)),
            (60, (0.6, 0.6, 0.8), (-0.2, -1.0, 0.5)),
        ),
        self_shadow_keyframes=(
            (0, 1, 0.0),
            (15, 0, 0.0),
            (30, 2, 0.05),
        ),
    )
    assert _round_trip(data) == data


def test_round_trip_preserves_bone_keyframe_values() -> None:
    """Even when bytes drift slightly (e.g. due to optional padding the
    writer chooses to zero-fill) the parsed values must stay identical."""
    data = build_vmd_wave()
    motion = parse_vmd(data)
    re_motion = parse_vmd(serialize_vmd(motion))
    assert motion.bone_keyframes == re_motion.bone_keyframes


def test_round_trip_preserves_morph_keyframe_values() -> None:
    motion = parse_vmd(build_vmd_morphs())
    re_motion = parse_vmd(serialize_vmd(motion))
    assert motion.morph_keyframes == re_motion.morph_keyframes


def test_round_trip_preserves_camera_keyframe_values() -> None:
    data = build_vmd_camera_motion(
        camera_keyframes=(
            (10, -20.0, (0.5, 1.0, -0.5), (0.0, 0.0, 0.0), 45, False),
        ),
    )
    motion = parse_vmd(data)
    re_motion = parse_vmd(serialize_vmd(motion))
    assert motion.camera_keyframes == re_motion.camera_keyframes


def test_round_trip_preserves_light_keyframe_values() -> None:
    data = build_vmd_camera_motion(
        light_keyframes=((5, (0.8, 0.7, 0.6), (-0.3, -0.9, 0.3)),),
    )
    motion = parse_vmd(data)
    re_motion = parse_vmd(serialize_vmd(motion))
    assert motion.light_keyframes == re_motion.light_keyframes


def test_round_trip_preserves_self_shadow_keyframe_values() -> None:
    data = build_vmd_camera_motion(self_shadow_keyframes=((20, 2, 0.05),))
    motion = parse_vmd(data)
    re_motion = parse_vmd(serialize_vmd(motion))
    assert motion.self_shadow_keyframes == re_motion.self_shadow_keyframes


def test_empty_motion_round_trips() -> None:
    """An empty motion (no keyframes anywhere) is still a valid VMD."""
    data = build_vmd_camera_motion()
    motion = parse_vmd(data)
    re_motion = parse_vmd(serialize_vmd(motion))
    assert motion == re_motion


def test_japanese_bone_name_round_trips() -> None:
    """SJIS truncation is lossless for short kanji names."""
    import struct  # noqa: PLC0415

    from tests.fixtures.mmd.build import _vmd_bone_record  # noqa: PLC0415

    signature = b"Vocaloid Motion Data 0002" + b"\x00" * 5
    model_name = b"\x00" * 20
    record = _vmd_bone_record(
        bone_name="左腕", frame=0,
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    bone_section = struct.pack("<I", 1) + record
    empty = struct.pack("<I", 0)
    data = signature + model_name + bone_section + empty * 5
    motion = parse_vmd(data)
    assert motion.bone_keyframes[0].bone_name == "左腕"
    re_motion = parse_vmd(serialize_vmd(motion))
    assert re_motion.bone_keyframes[0].bone_name == "左腕"


# Keep ``pytest`` reachable for downstream consumers of this module.
__all__ = ["pytest"]
