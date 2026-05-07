"""Offline exporters: VMD motion, PNG image sequence, MP4 video.

Each exporter takes a fully-resolved set of inputs (a populated
:class:`AnimationDocument`, a callable that renders one frame, an
ffmpeg-on-PATH expectation, …) and writes to disk. The export dialog
in :mod:`posecascade.ui.export_dialog` collects the parameters from
the user; this package is the headless half — testable without any
GUI at all.
"""

from posecascade.export.image_sequence import (
    ExportImageSequenceError,
    export_image_sequence,
)
from posecascade.export.video import (
    FfmpegFailedError,
    FfmpegNotFoundError,
    export_video_from_image_sequence,
)
from posecascade.export.vmd import export_animation_to_vmd

__all__ = [
    "ExportImageSequenceError",
    "FfmpegFailedError",
    "FfmpegNotFoundError",
    "export_animation_to_vmd",
    "export_image_sequence",
    "export_video_from_image_sequence",
]
