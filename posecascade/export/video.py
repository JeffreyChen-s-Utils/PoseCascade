"""Assemble a PNG image sequence into an MP4 via ``ffmpeg``.

We deliberately keep the dependency external — bundling ffmpeg with
PoseCascade would balloon the install size for the 5 % of users who
ever export. The exporter probes ``ffmpeg`` on PATH at call time and
raises :class:`FfmpegNotFoundError` when missing so the UI can show a
"please install ffmpeg" dialog instead of a generic failure.

Subprocess invocations are list-of-args + ``shell=False`` (the bandit
default + best practice), and stderr is captured then truncated in
the error message — full ffmpeg dumps are several KB which is too
much for a one-line UI toast.
"""
from __future__ import annotations

import shutil
import subprocess  # nosec B404 — invoke ffmpeg via list-of-args, shell=False
from pathlib import Path

from posecascade.errors import PoseCascadeError

DEFAULT_CODEC = "libx264"
DEFAULT_PIXEL_FORMAT = "yuv420p"
_DEFAULT_FILENAME_PREFIX = "frame"
_DEFAULT_PADDING = 4
_STDERR_PREVIEW_BYTES = 1024


class FfmpegNotFoundError(PoseCascadeError):
    """``ffmpeg`` is not on PATH — the user must install it."""


class FfmpegFailedError(PoseCascadeError):
    """``ffmpeg`` ran but exited non-zero. Carries a stderr preview."""


def find_ffmpeg() -> Path | None:
    """Return the absolute path to ``ffmpeg`` on PATH (or ``None``)."""
    found = shutil.which("ffmpeg")
    return Path(found) if found else None


def export_video_from_image_sequence(
    *,
    input_dir: Path,
    output_path: Path,
    fps: int = 30,
    codec: str = DEFAULT_CODEC,
    pixel_format: str = DEFAULT_PIXEL_FORMAT,
    filename_prefix: str = _DEFAULT_FILENAME_PREFIX,
    padding: int = _DEFAULT_PADDING,
    overwrite: bool = True,
) -> Path:
    """Encode every ``frameNNNN.png`` under ``input_dir`` into ``output_path``."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise FfmpegNotFoundError(
            "ffmpeg not found on PATH — install it from "
            "https://ffmpeg.org/download.html before exporting video.",
        )
    input_dir = Path(input_dir).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(input_dir / f"{filename_prefix}%0{padding}d.png")
    args: list[str] = [
        str(ffmpeg),
        "-y" if overwrite else "-n",
        "-framerate", str(int(fps)),
        "-i", pattern,
        "-c:v", codec,
        "-pix_fmt", pixel_format,
        str(output_path),
    ]
    try:
        result = subprocess.run(    # noqa: S603  # nosec B603 — args list, shell=False, paths validated
            args,
            check=False,
            capture_output=True,
            shell=False,
        )
    except FileNotFoundError as err:
        raise FfmpegNotFoundError(
            f"ffmpeg invocation failed to start: {err}",
        ) from err
    if result.returncode != 0:
        preview = result.stderr[-_STDERR_PREVIEW_BYTES:].decode("utf-8", errors="replace")
        raise FfmpegFailedError(
            f"ffmpeg exited with code {result.returncode}: {preview.strip()}",
        )
    return output_path
