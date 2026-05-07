"""Qt smoke tests for the export dialog."""
from __future__ import annotations

from pathlib import Path

import pytest

from posecascade.ui.export_dialog import (
    ExportDialog,
    ExportSpec,
    ExportTarget,
)


def test_dialog_starts_with_no_spec(qapp: object) -> None:
    dialog = ExportDialog()
    assert dialog.latest_spec is None


def test_vmd_tab_emits_spec_with_path(qapp: object) -> None:
    dialog = ExportDialog()
    dialog._tabs.setCurrentIndex(int(ExportTarget.VMD))           # noqa: SLF001
    dialog._vmd_path.setText("/tmp/out.vmd")                       # noqa: SLF001
    received: list[ExportSpec] = []
    dialog.export_requested.connect(received.append)
    dialog._on_ok()                                                # noqa: SLF001
    assert len(received) == 1
    spec = received[0]
    assert spec.target == ExportTarget.VMD
    assert spec.output_path == Path("/tmp/out.vmd")


def test_image_sequence_tab_collects_frame_range(qapp: object) -> None:
    dialog = ExportDialog()
    dialog._tabs.setCurrentIndex(int(ExportTarget.IMAGE_SEQUENCE))    # noqa: SLF001
    dialog._sequence_dir.setText("/tmp/frames")                       # noqa: SLF001
    dialog._sequence_start.setValue(10)                               # noqa: SLF001
    dialog._sequence_end.setValue(50)                                 # noqa: SLF001
    dialog._sequence_padding.setValue(6)                              # noqa: SLF001
    dialog._on_ok()                                                   # noqa: SLF001
    spec = dialog.latest_spec
    assert spec is not None
    assert spec.start_frame == 10
    assert spec.end_frame == 50
    assert spec.padding == 6


def test_video_tab_collects_codec_and_fps(qapp: object) -> None:
    dialog = ExportDialog()
    dialog._tabs.setCurrentIndex(int(ExportTarget.VIDEO))             # noqa: SLF001
    dialog._video_path.setText("/tmp/out.mp4")                        # noqa: SLF001
    dialog._video_fps.setValue(60)                                    # noqa: SLF001
    dialog._video_codec.setCurrentText("libx265")                     # noqa: SLF001
    dialog._on_ok()                                                   # noqa: SLF001
    spec = dialog.latest_spec
    assert spec is not None
    assert spec.fps == 60
    assert spec.codec == "libx265"


def test_empty_path_does_not_emit(qapp: object) -> None:
    dialog = ExportDialog()
    received: list[ExportSpec] = []
    dialog.export_requested.connect(received.append)
    # Output path empty — _on_ok short-circuits without accepting.
    dialog._tabs.setCurrentIndex(int(ExportTarget.VMD))           # noqa: SLF001
    dialog._on_ok()                                               # noqa: SLF001
    assert received == []


def test_post_effects_checkbox_default_true(qapp: object) -> None:
    dialog = ExportDialog()
    assert dialog._sequence_post_effects.isChecked() is True       # noqa: SLF001
    assert dialog._video_post_effects.isChecked() is True          # noqa: SLF001


# Keep ``pytest`` reachable for IDE jumps.
__all__ = ["pytest"]
