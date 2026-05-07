"""Tests for the offline image-sequence exporter."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from posecascade.export.image_sequence import (
    ExportImageSequenceError,
    export_image_sequence,
)


def _solid_red(_frame: int) -> np.ndarray:
    """Frame callable that returns a 4×4 solid-red RGBA buffer."""
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[..., 0] = 255    # R
    pixels[..., 3] = 255    # A
    return pixels


def _frame_indexed(frame: int) -> np.ndarray:
    """Frame callable that encodes the frame index in the red channel."""
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    pixels[..., 0] = frame & 0xFF
    pixels[..., 3] = 255
    return pixels


def test_export_writes_one_png_per_frame(tmp_path: Path) -> None:
    paths = export_image_sequence(
        render_frame_fn=_solid_red,
        start_frame=0,
        end_frame=4,
        output_dir=tmp_path,
    )
    assert len(paths) == 5
    for index, path in enumerate(paths):
        assert path.exists()
        assert path.name == f"frame{index:04d}.png"


def test_filename_padding_respects_argument(tmp_path: Path) -> None:
    paths = export_image_sequence(
        render_frame_fn=_solid_red,
        start_frame=10,
        end_frame=12,
        output_dir=tmp_path,
        padding=2,
    )
    assert paths[0].name == "frame10.png"


def test_filename_prefix_can_be_customised(tmp_path: Path) -> None:
    paths = export_image_sequence(
        render_frame_fn=_solid_red,
        start_frame=0,
        end_frame=0,
        output_dir=tmp_path,
        filename_prefix="shot_a_",
    )
    assert paths[0].name == "shot_a_0000.png"


def test_each_frame_callable_invoked_with_its_frame_index(tmp_path: Path) -> None:
    """The encoded red value should round-trip through the saved PNG."""
    from PIL import Image  # noqa: PLC0415 — lazy
    paths = export_image_sequence(
        render_frame_fn=_frame_indexed,
        start_frame=10, end_frame=12,
        output_dir=tmp_path,
    )
    for expected, path in enumerate(paths, start=10):
        rgba = np.asarray(Image.open(path).convert("RGBA"))
        assert int(rgba[0, 0, 0]) == expected


def test_existing_files_blocked_unless_overwrite(tmp_path: Path) -> None:
    export_image_sequence(
        render_frame_fn=_solid_red,
        start_frame=0, end_frame=2, output_dir=tmp_path,
    )
    with pytest.raises(ExportImageSequenceError, match="already contains"):
        export_image_sequence(
            render_frame_fn=_solid_red,
            start_frame=0, end_frame=2, output_dir=tmp_path,
        )


def test_overwrite_true_replaces_existing(tmp_path: Path) -> None:
    export_image_sequence(
        render_frame_fn=_solid_red,
        start_frame=0, end_frame=2, output_dir=tmp_path,
    )
    export_image_sequence(
        render_frame_fn=_solid_red,
        start_frame=0, end_frame=2, output_dir=tmp_path,
        overwrite=True,
    )
    assert sorted(tmp_path.glob("frame*.png"))     # still exist


def test_inverted_frame_range_raises(tmp_path: Path) -> None:
    with pytest.raises(ExportImageSequenceError, match="start_frame"):
        export_image_sequence(
            render_frame_fn=_solid_red,
            start_frame=10, end_frame=5, output_dir=tmp_path,
        )


def test_render_callable_returning_wrong_shape_raises(tmp_path: Path) -> None:
    def bad_callable(_frame: int) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)        # RGB, not RGBA

    with pytest.raises(ExportImageSequenceError, match=r"\(H, W, 4\) RGBA"):
        export_image_sequence(
            render_frame_fn=bad_callable,
            start_frame=0, end_frame=0, output_dir=tmp_path,
        )


def test_render_callable_returning_wrong_dtype_raises(tmp_path: Path) -> None:
    def bad_callable(_frame: int) -> np.ndarray:
        return np.zeros((4, 4, 4), dtype=np.float32)

    with pytest.raises(ExportImageSequenceError, match="uint8"):
        export_image_sequence(
            render_frame_fn=bad_callable,
            start_frame=0, end_frame=0, output_dir=tmp_path,
        )


def test_output_dir_created_when_missing(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "out"
    paths = export_image_sequence(
        render_frame_fn=_solid_red,
        start_frame=0, end_frame=0, output_dir=nested,
    )
    assert paths[0].exists()
