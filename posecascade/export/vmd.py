"""Convenience VMD export wrapper.

The byte-level writer lives in :mod:`vmd.writer` (Phase 11). The UI
side wants a single one-liner that takes "the current edit document
+ a destination path" and writes the file, with no detail about
``VmdMotion`` plumbing in between. This module owns that wrapper.
"""
from __future__ import annotations

from pathlib import Path

from vmd.writer import serialize_vmd

from posecascade.animation.document import AnimationDocument


def export_animation_to_vmd(
    document: AnimationDocument, path: Path,
) -> Path:
    """Serialise ``document`` to ``path`` and return the resolved path.

    Reads the document's current state through ``to_motion`` (the
    snapshot that the writer + the renderer's playback path already
    consume) so a save-then-reload yields a byte-identical file when
    no edits happen in between.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialize_vmd(document.to_motion()))
    return path
