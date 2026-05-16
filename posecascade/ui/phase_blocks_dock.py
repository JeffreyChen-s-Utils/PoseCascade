"""Visual phase-block editor for declarative animations (MVP-2).

A right-dock that renders every phase of the bound
:class:`AnimationJsonDocument` as a draggable card in a vertical list.
Each card summarises the phase (name, duration, pose, gait kind,
bones / morphs counts) so the author can scan the timeline at a glance.
Selecting a card reveals an inline form below the list with the
common-edit fields (name, duration_sec, blend_in_sec, blend_out_sec,
pose, hand_L, hand_R, body.yaw_rad, body.lean_x_rad).

Drag-and-drop on the list reorders the underlying ``phases`` array.
``+`` / ``Duplicate`` / ``Delete`` buttons cover the rest of the common
authoring needs. Advanced fields (gait, explicit bones, morphs,
expression strings) intentionally fall through to the JSON dock —
trying to express those cleanly in a Qt form would balloon the dock
and the author would still want raw access for the long tail.
"""
from __future__ import annotations

import contextlib
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from posecascade.ui.animation_command_stack import AnimationCommandStack
from posecascade.ui.animation_json_document import AnimationJsonDocument
from posecascade.ui.phase_editors import (
    BonesEditor,
    GaitEditor,
    MorphsEditor,
    TranslationEditor,
)
from posecascade.ui.phase_timeline_view import PhaseTimelineView
from posecascade.utils.logging import get_logger

_log = get_logger(__name__)

# Built-in pose / hand library names exposed in the form's combos. Custom
# entries in the document's ``pose_library`` / ``hand_library`` would
# need a richer combo populator; deferred until users actually author
# those (the bundled animations stick to built-ins).
_BUILTIN_POSES = (
    "(none)", "rest_arms", "v_arms_up", "arms_to_chest",
    "hip_pop_L", "hip_pop_R", "point_L", "point_R", "hands_clasp",
    "wave_L", "wave_R",
)
_BUILTIN_HANDS_L = (
    "(none)", "peace_L", "fist_L", "point_L", "open_palm_L", "thumbs_up_L",
)
_BUILTIN_HANDS_R = (
    "(none)", "peace_R", "fist_R", "point_R", "open_palm_R", "thumbs_up_R",
)
_NONE_LABEL = "(none)"

# Min / max bounds for the spin boxes. Generous enough that a worried
# user can't get stuck against the limit, tight enough that an
# accidental thousand-second blend doesn't blow up the timeline.
_DURATION_MIN_SEC = 0.05
_DURATION_MAX_SEC = 300.0
_BLEND_MAX_SEC = 60.0
_ROTATION_LIMIT_RAD = 100.0  # ~16 full turns — plenty for the spin-box clamp


class PhaseBlocksDock(QDockWidget):
    """Block-style phase editor sharing one document with the JSON dock."""

    # Mirrors ``AnimationJsonDock.reload_requested`` — keeps the wiring
    # uniform from MainWindow's perspective. Payload is the current
    # serialised document text.
    reload_requested = Signal(str)

    def __init__(
        self,
        document: AnimationJsonDocument | None = None,
        command_stack: AnimationCommandStack | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Phase blocks", parent)
        self._document = document if document is not None else AnimationJsonDocument(self)
        # Caller may supply a shared stack (MainWindow does, so undo
        # works across both docks). When omitted we keep a private one
        # so standalone use still has Ctrl+Z support.
        self._stack = command_stack if command_stack is not None else AnimationCommandStack(
            self._document, self,
        )
        self._timeline = PhaseTimelineView(document=self._document)
        self._list = QListWidget()
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection,
        )
        self._add_btn = QPushButton("+ Add")
        self._dup_btn = QPushButton("Duplicate")
        self._del_btn = QPushButton("Delete")
        self._reload_btn = QPushButton("Reload into runtime")
        self._form = _PhaseForm(self._document)
        self._build_ui()
        self._wire_signals()
        self._refresh_from_document()

    # ----- public API ---------------------------------------------------

    @property
    def document(self) -> AnimationJsonDocument:
        return self._document

    def set_document(self, document: AnimationJsonDocument) -> None:
        """Swap the bound document (used by MVP-3 when MainWindow shares one)."""
        # Disconnect raises if nothing was connected — first call after
        # __init__ on a fresh document we created internally. Suppress
        # both Qt's RuntimeError and the TypeError some bindings raise.
        with contextlib.suppress(TypeError, RuntimeError):
            self._document.changed.disconnect(self._refresh_from_document)
        self._document = document
        self._form.set_document(document)
        self._document.changed.connect(self._refresh_from_document)
        self._refresh_from_document()

    # ----- UI plumbing --------------------------------------------------

    def _build_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._add_btn)
        toolbar.addWidget(self._dup_btn)
        toolbar.addWidget(self._del_btn)
        toolbar.addStretch(1)
        toolbar.addWidget(self._reload_btn)
        layout.addLayout(toolbar)
        # Horizontal timeline strip on top — gives the author the
        # "what does my reel look like end-to-end" view before the
        # vertical card list below.
        layout.addWidget(self._timeline)
        layout.addWidget(self._list, stretch=1)
        layout.addWidget(self._form, stretch=1)
        self.setWidget(container)

    def _wire_signals(self) -> None:
        self._document.changed.connect(self._refresh_from_document)
        self._list.currentRowChanged.connect(self._on_selection_changed)
        # Timeline → list / list → timeline cross-selection.
        self._timeline.phase_selected.connect(self._list.setCurrentRow)
        self._timeline.phase_moved.connect(self._on_timeline_moved)
        self._timeline.phase_duration_changed.connect(
            self._on_timeline_duration_changed,
        )
        # Ctrl+Z / Ctrl+Y route through the shared command stack —
        # picks up edits made through either dock.
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self._stack.undo)
        self.addAction(undo_action)
        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self._stack.redo)
        self.addAction(redo_action)
        # Internal-move drops change the row order — Qt fires
        # ``rowsMoved`` on the underlying model. We push that into the
        # document so the source of truth stays consistent.
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        self._add_btn.clicked.connect(self._on_add_clicked)
        self._dup_btn.clicked.connect(self._on_dup_clicked)
        self._del_btn.clicked.connect(self._on_del_clicked)
        self._reload_btn.clicked.connect(self._on_reload_clicked)

    def _refresh_from_document(self) -> None:
        """Rebuild the list from the document. Preserves selection by index."""
        previous = self._list.currentRow()
        self._list.blockSignals(True)
        try:
            self._list.clear()
            for idx, phase in enumerate(self._document.phases()):
                item = QListWidgetItem(_phase_summary(phase))
                item.setData(Qt.ItemDataRole.UserRole, idx)
                self._list.addItem(item)
        finally:
            self._list.blockSignals(False)
        if 0 <= previous < self._list.count():
            self._list.setCurrentRow(previous)
            self._form.show_phase(previous)
        else:
            self._form.show_phase(None)

    def _on_selection_changed(self, row: int) -> None:
        self._form.show_phase(row if row >= 0 else None)
        self._timeline.select(row if row >= 0 else None)

    def _on_timeline_moved(self, source: int, dest: int) -> None:
        if self._document.move_phase(source, dest):
            self._list.setCurrentRow(dest)

    def _on_timeline_duration_changed(self, idx: int, duration: float) -> None:
        self._document.update_phase_field(idx, "duration_sec", float(duration))

    def _on_rows_moved(
        self,
        _parent: object,
        start: int,
        _end: int,
        _dest_parent: object,
        dest_row: int,
    ) -> None:
        """Translate Qt's ``rowsMoved`` into a document ``move_phase`` call.

        Qt's ``dest_row`` is the row BEFORE removal, hence the ``-1``
        when the drop lands after the source. The list widget is
        single-selection so ``start == end``.
        """
        target = dest_row - 1 if dest_row > start else dest_row
        if target == start:
            return
        # ``move_phase`` will re-emit ``changed`` → ``_refresh_from_document``,
        # which rebuilds the list. Suppress the secondary signal that
        # Qt fires on the new selection to avoid the form jumping.
        self._document.move_phase(start, target)

    def _on_add_clicked(self) -> None:
        self._stack.push_snapshot("add phase")
        new_idx = self._document.add_phase()
        self._list.setCurrentRow(new_idx)

    def _on_dup_clicked(self) -> None:
        idx = self._list.currentRow()
        if idx < 0:
            return
        self._stack.push_snapshot("duplicate phase")
        new_idx = self._document.duplicate_phase(idx)
        if new_idx is not None:
            self._list.setCurrentRow(new_idx)

    def _on_del_clicked(self) -> None:
        idx = self._list.currentRow()
        if idx < 0:
            return
        self._stack.push_snapshot("delete phase")
        self._document.remove_phase(idx)

    def _on_reload_clicked(self) -> None:
        self.reload_requested.emit(self._document.text())


class _PhaseForm(QWidget):
    """Inline form for the selected phase, covering every common field.

    Split into two parts:

    * a "Basic" form (name / duration / blends / pose / hands / body
      yaw + lean) directly bound to the document.
    * Four child editors (``GaitEditor``, ``TranslationEditor``,
      ``BonesEditor``, ``MorphsEditor``) for the deeper sections —
      each writes back into ``phases[idx]`` on edit.

    Wrapped in a :class:`QScrollArea` so the dock stays usable on the
    half-screen tabs without forcing the user to resize.
    """

    def __init__(
        self,
        document: AnimationJsonDocument,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._document = document
        self._current_idx: int | None = None
        self._loading = False
        self._build_widgets()
        scroll_inner = self._build_scroll_inner()

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(scroll_inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

        self._wire_signals()
        self.show_phase(None)

    def set_document(self, document: AnimationJsonDocument) -> None:
        self._document = document
        self.show_phase(None)

    def _build_widgets(self) -> None:
        """Construct the per-field widgets + their basic-form layout.

        Extracted from ``__init__`` so the constructor stays under the
        cyclomatic-complexity bound — the form has nine basic fields
        plus four child editors, which adds up fast.
        """
        self._basic_group = QGroupBox("Basic")
        self._name_edit = QLineEdit()
        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(_DURATION_MIN_SEC, _DURATION_MAX_SEC)
        self._duration_spin.setDecimals(3)
        self._duration_spin.setSingleStep(0.1)
        self._blend_in_spin = QDoubleSpinBox()
        self._blend_in_spin.setRange(0.0, _BLEND_MAX_SEC)
        self._blend_in_spin.setDecimals(3)
        self._blend_out_spin = QDoubleSpinBox()
        self._blend_out_spin.setRange(0.0, _BLEND_MAX_SEC)
        self._blend_out_spin.setDecimals(3)
        self._pose_combo = QComboBox()
        self._pose_combo.addItems(_BUILTIN_POSES)
        self._hand_l_combo = QComboBox()
        self._hand_l_combo.addItems(_BUILTIN_HANDS_L)
        self._hand_r_combo = QComboBox()
        self._hand_r_combo.addItems(_BUILTIN_HANDS_R)
        self._yaw_spin = QDoubleSpinBox()
        self._yaw_spin.setRange(-_ROTATION_LIMIT_RAD, _ROTATION_LIMIT_RAD)
        self._yaw_spin.setDecimals(4)
        self._lean_spin = QDoubleSpinBox()
        self._lean_spin.setRange(-_ROTATION_LIMIT_RAD, _ROTATION_LIMIT_RAD)
        self._lean_spin.setDecimals(4)
        self._gait_editor = GaitEditor()
        self._translation_editor = TranslationEditor()
        self._bones_editor = BonesEditor()
        self._morphs_editor = MorphsEditor()
        basic_layout = QFormLayout(self._basic_group)
        basic_layout.addRow("Name:", self._name_edit)
        basic_layout.addRow("Duration (s):", self._duration_spin)
        basic_layout.addRow("Blend in (s):", self._blend_in_spin)
        basic_layout.addRow("Blend out (s):", self._blend_out_spin)
        basic_layout.addRow("Pose:", self._pose_combo)
        basic_layout.addRow("Hand L:", self._hand_l_combo)
        basic_layout.addRow("Hand R:", self._hand_r_combo)
        basic_layout.addRow("Body yaw (rad):", self._yaw_spin)
        basic_layout.addRow("Body lean X (rad):", self._lean_spin)

    def _build_scroll_inner(self) -> QWidget:
        """Vertical stack of basic form + the four child editors."""
        scroll_inner = QWidget()
        inner_layout = QVBoxLayout(scroll_inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.addWidget(self._basic_group)
        inner_layout.addWidget(self._gait_editor)
        inner_layout.addWidget(self._translation_editor)
        inner_layout.addWidget(self._bones_editor)
        inner_layout.addWidget(self._morphs_editor)
        inner_layout.addStretch(1)
        return scroll_inner

    def show_phase(self, idx: int | None) -> None:
        """Repopulate every section from ``phases[idx]``; disable if ``None``."""
        self._current_idx = idx
        if idx is None:
            self.setEnabled(False)
            return
        phases = self._document.phases()
        if not 0 <= idx < len(phases):
            self.setEnabled(False)
            return
        phase = phases[idx]
        self._loading = True
        try:
            self.setEnabled(True)
            self._name_edit.setText(str(phase.get("name", "")))
            self._duration_spin.setValue(float(phase.get("duration_sec", 1.0)))
            self._blend_in_spin.setValue(float(phase.get("blend_in_sec", 0.0)))
            self._blend_out_spin.setValue(float(phase.get("blend_out_sec", 0.0)))
            self._pose_combo.setCurrentText(_pose_label(phase.get("pose")))
            self._hand_l_combo.setCurrentText(phase.get("hand_L") or _NONE_LABEL)
            self._hand_r_combo.setCurrentText(phase.get("hand_R") or _NONE_LABEL)
            body = phase.get("body") or {}
            self._yaw_spin.setValue(_safe_scalar(body.get("yaw_rad", 0.0)))
            self._lean_spin.setValue(_safe_scalar(body.get("lean_x_rad", 0.0)))
            self._gait_editor.set_value(phase.get("gait"))
            self._translation_editor.set_value(body.get("translation", {}))
            self._bones_editor.set_value(phase.get("bones") or {})
            self._morphs_editor.set_value(phase.get("morphs") or {})
        finally:
            self._loading = False

    def _wire_signals(self) -> None:
        self._name_edit.editingFinished.connect(self._on_name_changed)
        self._duration_spin.valueChanged.connect(
            lambda v: self._write("duration_sec", float(v)),
        )
        self._blend_in_spin.valueChanged.connect(
            lambda v: self._write("blend_in_sec", float(v)),
        )
        self._blend_out_spin.valueChanged.connect(
            lambda v: self._write("blend_out_sec", float(v)),
        )
        self._pose_combo.currentTextChanged.connect(self._on_pose_changed)
        self._hand_l_combo.currentTextChanged.connect(
            lambda t: self._write("hand_L", None if t == _NONE_LABEL else t),
        )
        self._hand_r_combo.currentTextChanged.connect(
            lambda t: self._write("hand_R", None if t == _NONE_LABEL else t),
        )
        self._yaw_spin.valueChanged.connect(
            lambda v: self._write_body("yaw_rad", float(v)),
        )
        self._lean_spin.valueChanged.connect(
            lambda v: self._write_body("lean_x_rad", float(v)),
        )
        self._gait_editor.changed.connect(self._on_gait_changed)
        self._translation_editor.changed.connect(self._on_translation_changed)
        self._bones_editor.changed.connect(self._on_bones_changed)
        self._morphs_editor.changed.connect(self._on_morphs_changed)

    def _on_name_changed(self) -> None:
        text = self._name_edit.text().strip()
        if not text:
            return
        self._write("name", text)

    def _on_pose_changed(self, text: str) -> None:
        self._write("pose", None if text == _NONE_LABEL else text)

    def _on_gait_changed(self, value: Any) -> None:
        self._write("gait", value)  # ``None`` deletes the key

    def _on_bones_changed(self, value: dict[str, Any]) -> None:
        # An empty dict means "no explicit bone overrides"; drop the
        # key entirely to keep the JSON clean.
        self._write("bones", value if value else None)

    def _on_morphs_changed(self, value: dict[str, Any]) -> None:
        self._write("morphs", value if value else None)

    def _on_translation_changed(self, value: Any) -> None:
        if self._loading or self._current_idx is None:
            return
        phases = self._document.phases()
        if not 0 <= self._current_idx < len(phases):
            return
        body = phases[self._current_idx].setdefault("body", {})
        if not isinstance(body, dict):
            return
        body["translation"] = value
        self._document.changed.emit()

    def _write(self, key: str, value: Any) -> None:
        if self._loading or self._current_idx is None:
            return
        self._document.update_phase_field(self._current_idx, key, value)

    def _write_body(self, key: str, value: float) -> None:
        if self._loading or self._current_idx is None:
            return
        phases = self._document.phases()
        if not 0 <= self._current_idx < len(phases):
            return
        body = phases[self._current_idx].setdefault("body", {})
        if not isinstance(body, dict):
            return
        body[key] = value
        self._document.changed.emit()


def _phase_summary(phase: dict[str, Any]) -> str:
    """One-line summary shown in the list rows.

    Author-friendly — the most relevant fields surface first. Empty
    sections are silently omitted so a minimal idle phase reads
    ``"breathe · 4.0s"`` not ``"breathe · 4.0s · 0 bones · 0 morphs"``.
    """
    name = str(phase.get("name", "(unnamed)"))
    duration = float(phase.get("duration_sec", 0.0))
    pieces: list[str] = [f"{name} · {duration:.1f}s"]
    if isinstance(phase.get("pose"), str):
        pieces.append(f"pose={phase['pose']}")
    gait = phase.get("gait")
    if isinstance(gait, dict):
        pieces.append(f"gait={gait.get('kind', '?')}")
    bones = phase.get("bones")
    if isinstance(bones, dict) and bones:
        pieces.append(f"{len(bones)} bones")
    morphs = phase.get("morphs")
    if isinstance(morphs, dict) and morphs:
        pieces.append(f"{len(morphs)} morphs")
    return "  ·  ".join(pieces)


def _pose_label(value: Any) -> str:
    """Map a stored pose value (str / dict / None) to a combo label."""
    if value is None:
        return _NONE_LABEL
    if isinstance(value, dict):
        return str(value.get("name", _NONE_LABEL))
    return str(value)


def _safe_scalar(value: Any) -> float:
    """Coerce a JSON value that might be a curve dict/string back to a float.

    Form spin boxes can only show numbers; expressions and curve specs
    pass through unchanged in the document but render as 0.0 in the
    form. This is the explicit MVP-2 cut: the JSON dock is where the
    long tail lives.
    """
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


__all__ = ["PhaseBlocksDock"]
