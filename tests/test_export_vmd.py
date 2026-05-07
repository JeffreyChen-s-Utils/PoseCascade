"""Tests for the high-level VMD export wrapper."""
from __future__ import annotations

from pathlib import Path

import pytest
from vmd.reader import parse_vmd

from posecascade.animation.document import AnimationDocument
from posecascade.export.vmd import export_animation_to_vmd

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mmd"


def test_export_writes_byte_identical_round_trip(tmp_path: Path) -> None:
    """Loading a VMD into a document, exporting, and re-parsing should
    yield the same parsed motion (the writer round-trip is already
    tested at the byte level; this asserts the wrapper integrates the
    document → motion conversion correctly)."""
    motion = parse_vmd((_FIXTURES / "wave.vmd").read_bytes())
    document = AnimationDocument.from_motion(motion)
    output = tmp_path / "wave_out.vmd"
    written = export_animation_to_vmd(document, output)
    assert written == output.resolve()
    re_motion = parse_vmd(output.read_bytes())
    assert re_motion.bone_keyframes == motion.bone_keyframes


def test_export_creates_parent_dirs(tmp_path: Path) -> None:
    document = AnimationDocument()
    output = tmp_path / "deep" / "sub" / "empty.vmd"
    export_animation_to_vmd(document, output)
    assert output.exists()


# Keep ``pytest`` reachable for IDE jumps.
__all__ = ["pytest"]
