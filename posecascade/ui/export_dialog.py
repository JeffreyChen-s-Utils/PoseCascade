"""Export dialog — collect parameters for a VMD / image-sequence / video export.

The dialog is a thin shell over the headless export package: it
gathers user input into an :class:`ExportSpec` then emits
``export_requested(spec)``. The integrator wires that signal up to
the actual exporter call so the UI itself stays free of disk I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

DEFAULT_FPS = 30
DEFAULT_PADDING = 4
DEFAULT_END_FRAME = 900
_DEFAULT_VIDEO_CODECS = ("libx264", "libx265", "mpeg4", "prores")
_MAX_FRAME = 1_000_000


class ExportTarget(IntEnum):
    """Which export pipeline the dialog wired up."""

    VMD = 0
    IMAGE_SEQUENCE = 1
    VIDEO = 2


@dataclass
class ExportSpec:
    """One export request — the dialog hands this off to the integrator."""

    target: ExportTarget
    output_path: Path
    start_frame: int = 0
    end_frame: int = DEFAULT_END_FRAME
    fps: int = DEFAULT_FPS
    padding: int = DEFAULT_PADDING
    codec: str = "libx264"
    include_post_effects: bool = True
    extra: dict[str, object] = field(default_factory=dict)


class ExportDialog(QDialog):
    """Modal dialog for the user to pick an export pipeline + parameters."""

    export_requested = Signal(ExportSpec)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export")
        self._spec: ExportSpec | None = None
        self._tabs = QTabWidget()
        self._build_ui()

    @property
    def latest_spec(self) -> ExportSpec | None:
        """The :class:`ExportSpec` from the most recent OK click (or ``None``)."""
        return self._spec

    # ----- internal ----------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._tabs)
        self._build_vmd_tab()
        self._build_image_sequence_tab()
        self._build_video_tab()
        button_row = QHBoxLayout()
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.clicked.connect(self.reject)
        self._ok_button = QPushButton("Export")
        self._ok_button.clicked.connect(self._on_ok)
        button_row.addStretch(1)
        button_row.addWidget(self._cancel_button)
        button_row.addWidget(self._ok_button)
        layout.addLayout(button_row)

    def _build_vmd_tab(self) -> None:
        widget = QWidget()
        form = QFormLayout(widget)
        self._vmd_path = QLineEdit()
        self._vmd_path.setPlaceholderText("/path/to/output.vmd")
        form.addRow("Output", self._vmd_path)
        self._tabs.addTab(widget, "VMD")

    def _build_image_sequence_tab(self) -> None:
        widget = QWidget()
        form = QFormLayout(widget)
        self._sequence_dir = QLineEdit()
        self._sequence_dir.setPlaceholderText("/path/to/output_dir")
        form.addRow("Output dir", self._sequence_dir)
        self._sequence_start = QSpinBox()
        self._sequence_start.setRange(0, _MAX_FRAME)
        self._sequence_start.setValue(0)
        form.addRow("Start frame", self._sequence_start)
        self._sequence_end = QSpinBox()
        self._sequence_end.setRange(0, _MAX_FRAME)
        self._sequence_end.setValue(DEFAULT_END_FRAME)
        form.addRow("End frame", self._sequence_end)
        self._sequence_padding = QSpinBox()
        self._sequence_padding.setRange(1, 8)
        self._sequence_padding.setValue(DEFAULT_PADDING)
        form.addRow("Filename padding", self._sequence_padding)
        self._sequence_post_effects = QCheckBox("Include post-effect chain")
        self._sequence_post_effects.setChecked(True)
        form.addRow(self._sequence_post_effects)
        self._tabs.addTab(widget, "Image sequence")

    def _build_video_tab(self) -> None:
        widget = QWidget()
        form = QFormLayout(widget)
        self._video_path = QLineEdit()
        self._video_path.setPlaceholderText("/path/to/output.mp4")
        form.addRow("Output", self._video_path)
        self._video_start = QSpinBox()
        self._video_start.setRange(0, _MAX_FRAME)
        form.addRow("Start frame", self._video_start)
        self._video_end = QSpinBox()
        self._video_end.setRange(0, _MAX_FRAME)
        self._video_end.setValue(DEFAULT_END_FRAME)
        form.addRow("End frame", self._video_end)
        self._video_fps = QSpinBox()
        self._video_fps.setRange(1, 120)
        self._video_fps.setValue(DEFAULT_FPS)
        form.addRow("FPS", self._video_fps)
        self._video_codec = QComboBox()
        for codec in _DEFAULT_VIDEO_CODECS:
            self._video_codec.addItem(codec)
        form.addRow("Codec", self._video_codec)
        self._video_post_effects = QCheckBox("Include post-effect chain")
        self._video_post_effects.setChecked(True)
        form.addRow(self._video_post_effects)
        self._tabs.addTab(widget, "Video")

    def _on_ok(self) -> None:
        spec = self._build_spec()
        if spec is None:
            return
        self._spec = spec
        self.export_requested.emit(spec)
        self.accept()

    def _build_spec(self) -> ExportSpec | None:
        index = self._tabs.currentIndex()
        if index == ExportTarget.VMD:
            return self._build_vmd_spec()
        if index == ExportTarget.IMAGE_SEQUENCE:
            return self._build_image_sequence_spec()
        return self._build_video_spec()

    def _build_vmd_spec(self) -> ExportSpec | None:
        path_text = self._vmd_path.text().strip()
        if not path_text:
            return None
        return ExportSpec(target=ExportTarget.VMD, output_path=Path(path_text))

    def _build_image_sequence_spec(self) -> ExportSpec | None:
        dir_text = self._sequence_dir.text().strip()
        if not dir_text:
            return None
        return ExportSpec(
            target=ExportTarget.IMAGE_SEQUENCE,
            output_path=Path(dir_text),
            start_frame=int(self._sequence_start.value()),
            end_frame=int(self._sequence_end.value()),
            padding=int(self._sequence_padding.value()),
            include_post_effects=bool(self._sequence_post_effects.isChecked()),
        )

    def _build_video_spec(self) -> ExportSpec | None:
        path_text = self._video_path.text().strip()
        if not path_text:
            return None
        return ExportSpec(
            target=ExportTarget.VIDEO,
            output_path=Path(path_text),
            start_frame=int(self._video_start.value()),
            end_frame=int(self._video_end.value()),
            fps=int(self._video_fps.value()),
            codec=self._video_codec.currentText(),
            include_post_effects=bool(self._video_post_effects.isChecked()),
        )
