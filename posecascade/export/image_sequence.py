"""Render-frame-callable → numbered PNG sequence on disk.

The caller threads in ``render_frame_fn(frame_index) -> ndarray (H, W, 4)
uint8`` so the exporter stays decoupled from the renderer + FBO setup.
A unit test can pass a deterministic dummy that fills in solid colours
and assert the file count + filename pattern; an integration test can
swap in the real :class:`Renderer.draw` + ``glReadPixels`` path.

Output filenames are zero-padded so a downstream ffmpeg pass picks
them up via ``frame%04d.png`` without sorting hiccups.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from numpy.typing import NDArray

from posecascade.errors import PoseCascadeError

DEFAULT_FILENAME_PREFIX = "frame"
DEFAULT_PADDING = 4
_RGBA_NDIM = 3
_RGBA_CHANNELS = 4


class ExportImageSequenceError(PoseCascadeError):
    """Raised when the exporter can't write — bad path / collision / etc."""


def export_image_sequence(
    *,
    render_frame_fn: Callable[[int], NDArray],
    start_frame: int,
    end_frame: int,
    output_dir: Path,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
    padding: int = DEFAULT_PADDING,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Render frames ``start_frame..end_frame`` (inclusive) and write PNGs.

    Returns the tuple of paths in render order. Raises
    :class:`ExportImageSequenceError` when ``output_dir`` already
    contains files matching the same prefix and ``overwrite`` is
    ``False`` — guards against an accidental dump-on-top.
    """
    if start_frame > end_frame:
        raise ExportImageSequenceError(
            f"start_frame {start_frame} > end_frame {end_frame}",
        )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = sorted(output_dir.glob(f"{filename_prefix}*.png"))
        if existing:
            raise ExportImageSequenceError(
                f"output dir already contains {len(existing)} matching files; "
                f"pass overwrite=True to replace",
            )
    return tuple(
        _render_one_frame(
            render_frame_fn=render_frame_fn,
            frame=frame,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
            padding=padding,
        )
        for frame in range(start_frame, end_frame + 1)
    )


def _render_one_frame(
    *,
    render_frame_fn: Callable[[int], NDArray],
    frame: int,
    output_dir: Path,
    filename_prefix: str,
    padding: int,
) -> Path:
    pixels = render_frame_fn(frame)
    if pixels.ndim != _RGBA_NDIM or pixels.shape[2] != _RGBA_CHANNELS:
        raise ExportImageSequenceError(
            f"render_frame_fn must return (H, W, 4) RGBA, got {pixels.shape}",
        )
    if pixels.dtype.kind != "u" or pixels.dtype.itemsize != 1:
        raise ExportImageSequenceError(
            f"render_frame_fn must return uint8, got {pixels.dtype}",
        )
    from PIL import Image  # noqa: PLC0415 — lazy import keeps non-Pillow tests fast
    filename = output_dir / f"{filename_prefix}{frame:0{padding}d}.png"
    Image.fromarray(pixels, mode="RGBA").save(filename)
    return filename
