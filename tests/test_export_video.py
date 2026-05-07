"""Tests for the ffmpeg video exporter (skipped when ffmpeg is missing)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from posecascade.export.image_sequence import export_image_sequence
from posecascade.export.video import (
    FfmpegFailedError,
    FfmpegNotFoundError,
    export_video_from_image_sequence,
    find_ffmpeg,
)


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


pytestmark = pytest.mark.skipif(
    not _has_ffmpeg(),
    reason="ffmpeg not on PATH",
)


def _solid_blue(_frame: int) -> np.ndarray:
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[..., 2] = 255
    pixels[..., 3] = 255
    return pixels


def _populate_image_sequence(tmp_path: Path) -> Path:
    """Drop a small image sequence under ``tmp_path`` and return the dir."""
    output_dir = tmp_path / "frames"
    export_image_sequence(
        render_frame_fn=_solid_blue,
        start_frame=0, end_frame=4, output_dir=output_dir,
    )
    return output_dir


def test_find_ffmpeg_returns_a_path() -> None:
    assert find_ffmpeg() is not None


def test_export_creates_mp4(tmp_path: Path) -> None:
    output = tmp_path / "out.mp4"
    result = export_video_from_image_sequence(
        input_dir=_populate_image_sequence(tmp_path),
        output_path=output,
        fps=30,
    )
    assert result.exists()
    assert result.stat().st_size > 0


def test_export_overwrite_false_refuses_to_replace(tmp_path: Path) -> None:
    output = tmp_path / "out.mp4"
    output.write_bytes(b"\x00" * 16)
    with pytest.raises(FfmpegFailedError):
        export_video_from_image_sequence(
            input_dir=_populate_image_sequence(tmp_path),
            output_path=output,
            fps=30,
            overwrite=False,
        )


def test_export_invalid_codec_raises(tmp_path: Path) -> None:
    with pytest.raises(FfmpegFailedError):
        export_video_from_image_sequence(
            input_dir=_populate_image_sequence(tmp_path),
            output_path=tmp_path / "out.mp4",
            fps=30,
            codec="not_a_real_codec",
        )


def test_missing_ffmpeg_path_raises(tmp_path: Path, monkeypatch) -> None:    # noqa: ANN001
    """Force ``find_ffmpeg`` to return ``None`` and verify the dedicated exception."""
    from posecascade.export import video as video_mod  # noqa: PLC0415
    monkeypatch.setattr(video_mod, "find_ffmpeg", lambda: None)
    with pytest.raises(FfmpegNotFoundError, match="install"):
        export_video_from_image_sequence(
            input_dir=_populate_image_sequence(tmp_path),
            output_path=tmp_path / "out.mp4",
            fps=30,
        )


def test_subprocess_error_reraised_as_ffmpeg_not_found(
    tmp_path: Path, monkeypatch,                                          # noqa: ANN001
) -> None:
    """Simulate the case where ``ffmpeg`` is on PATH at probe-time but
    fails to actually launch (rare, but possible during package
    upgrades). The error path translates ``FileNotFoundError`` to
    :class:`FfmpegNotFoundError`."""
    def boom(*_args, **_kwargs) -> subprocess.CompletedProcess:
        raise FileNotFoundError("simulated path-broken ffmpeg")
    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(FfmpegNotFoundError, match="failed to start"):
        export_video_from_image_sequence(
            input_dir=_populate_image_sequence(tmp_path),
            output_path=tmp_path / "out.mp4",
            fps=30,
        )


# Keep ``pytest`` reachable for IDE jumps.
__all__ = ["pytest"]
