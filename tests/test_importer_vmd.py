"""Tests for the VMD importer + bezier curve sampler.

Exercises both the byte-level parser (``vmd.reader.parse_vmd``) and the
engine-side adapter (``vmd.importer.build_motion_asset``).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from vmd.importer import VmdImporter, build_motion_asset
from vmd.reader import _decode_bone_bezier, parse_vmd

from posecascade.animation.vmd_curves import evaluate_bezier
from posecascade.animation.vmd_track import VMD_FRAMES_PER_SECOND
from posecascade.errors import MalformedAssetError
from tests.fixtures.mmd.build import build_vmd_wave


@pytest.fixture
def wave_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "mmd" / "wave.vmd"


# ----- bezier curve sampler ---------------------------------------------
def test_bezier_endpoints_clamp() -> None:
    assert evaluate_bezier((20, 20, 107, 107), -0.5) == 0.0
    assert evaluate_bezier((20, 20, 107, 107), 0.0) == 0.0
    assert evaluate_bezier((20, 20, 107, 107), 1.0) == 1.0
    assert evaluate_bezier((20, 20, 107, 107), 1.5) == 1.0


def test_linear_handles_pass_through() -> None:
    """When the handles sit on the diagonal, the curve eases the same as t."""
    for t in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert evaluate_bezier((20, 20, 107, 107), t) == pytest.approx(t, abs=1.0e-6)


def test_ease_in_curve_below_diagonal() -> None:
    """Strong ease-in: low Y for early X — the eased value should sag below t=0.5."""
    eased = evaluate_bezier((100, 0, 100, 0), 0.5)
    assert eased < 0.4


def test_ease_out_curve_above_diagonal() -> None:
    """Strong ease-out: high Y for early X — the eased value should rise above 0.5."""
    eased = evaluate_bezier((20, 80, 20, 100), 0.5)
    assert eased > 0.6


def test_step_handles_returns_finite_value() -> None:
    """Degenerate (0, 127, 0, 127) handles must not loop the bisection forever."""
    eased = evaluate_bezier((0, 127, 0, 127), 0.5)
    assert 0.0 <= eased <= 1.0


# ----- reader ------------------------------------------------------------
def test_canonical_vmd_loads(wave_path: Path) -> None:
    motion = parse_vmd(wave_path.read_bytes())
    assert motion.header.signature.startswith("Vocaloid Motion Data")
    assert motion.header.model_name == "tiny"
    assert len(motion.bone_keyframes) == 5
    assert motion.bone_keyframes[0].bone_name == "child"
    assert motion.bone_keyframes[0].frame == 0
    assert motion.bone_keyframes[2].frame == 10


def test_morph_section_present_but_empty(wave_path: Path) -> None:
    motion = parse_vmd(wave_path.read_bytes())
    assert motion.morph_keyframes == ()


def test_truncated_vmd_after_bone_section_is_tolerated() -> None:
    """Older VMD writers omit camera / light / etc. — parser must accept that."""
    full = build_vmd_wave()
    # Strip the morph section onward (the writer emits 6 trailing zero counts;
    # take the first count byte alone to simulate an early-stop EOF).
    truncated = full[: len(full) - 4 * 5]   # drop last 5 zero-count uint32s
    motion = parse_vmd(truncated)
    assert len(motion.bone_keyframes) == 5
    assert motion.morph_keyframes == ()


def test_unknown_self_shadow_mode_raises(tmp_path: Path) -> None:
    """A self-shadow record with an out-of-range mode byte should fail loudly."""
    full = bytearray(build_vmd_wave())
    # Replace the empty self-shadow section count with 1 record + a bogus mode byte.
    # Layout from the end: morph(4) camera(4) light(4) self_shadow(4) ik(4) — locate offset.
    self_shadow_count_offset = len(full) - 4 * 2
    full[self_shadow_count_offset:self_shadow_count_offset + 4] = struct.pack("<I", 1)
    record = struct.pack("<I", 0) + bytes([99]) + struct.pack("<f", 0.0)
    full = full[:self_shadow_count_offset + 4] + record + full[self_shadow_count_offset + 4:]
    path = tmp_path / "bad.vmd"
    path.write_bytes(bytes(full))
    with pytest.raises(MalformedAssetError, match="self-shadow mode"):
        VmdImporter().load(path)


# ----- importer adapter --------------------------------------------------
def test_importer_groups_keyframes_by_bone(wave_path: Path) -> None:
    asset = VmdImporter().load(wave_path)
    assert len(asset.bone_tracks) == 1
    track = asset.bone_tracks[0]
    assert track.name_key == "child"
    assert track.frames.tolist() == [0, 5, 10, 15, 20]
    assert track.bezier_handles.shape == (5, 4, 4)


def test_importer_returns_target_model_name(wave_path: Path) -> None:
    asset = VmdImporter().load(wave_path)
    assert asset.target_model_name == "tiny"


def test_motion_asset_duration_frames(wave_path: Path) -> None:
    asset = VmdImporter().load(wave_path)
    assert asset.duration_frames == 20
    assert asset.duration_frames / VMD_FRAMES_PER_SECOND == pytest.approx(20.0 / 30.0)


def test_keyframes_sort_when_out_of_order(tmp_path: Path) -> None:
    """The importer must restore frame order even if the source bytes
    aren't already sorted (some VMD writers concatenate streams)."""
    motion = parse_vmd(build_vmd_wave())
    shuffled_frames = motion.bone_keyframes[::-1]
    motion_shuffled = type(motion)(
        header=motion.header,
        bone_keyframes=shuffled_frames,
        morph_keyframes=motion.morph_keyframes,
        camera_keyframes=motion.camera_keyframes,
        light_keyframes=motion.light_keyframes,
        self_shadow_keyframes=motion.self_shadow_keyframes,
        ik_keyframes=motion.ik_keyframes,
    )
    asset = build_motion_asset(motion_shuffled)
    assert asset.bone_tracks[0].frames.tolist() == [0, 5, 10, 15, 20]


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MalformedAssetError, match="VMD file not found"):
        VmdImporter().load(tmp_path / "nope.vmd")


# ----- bezier handle decoding sanity -------------------------------------
def test_bone_bezier_layout_columns_per_channel(wave_path: Path) -> None:
    """Channel ``c``'s ``(X1, Y1, X2, Y2)`` lives at bytes ``c, 4+c, 8+c, 12+c``.

    Sanity-check the decoder by feeding it a hand-crafted block where each
    channel has unique values.
    """
    block = bytearray(64)
    for channel in range(4):
        block[channel + 0] = 11 + channel
        block[channel + 4] = 22 + channel
        block[channel + 8] = 33 + channel
        block[channel + 12] = 44 + channel
    decoded = _decode_bone_bezier(bytes(block))
    np.testing.assert_array_equal(
        np.asarray(decoded, dtype=np.uint8),
        np.asarray(
            [
                [11, 22, 33, 44],
                [12, 23, 34, 45],
                [13, 24, 35, 46],
                [14, 25, 36, 47],
            ],
            dtype=np.uint8,
        ),
    )
