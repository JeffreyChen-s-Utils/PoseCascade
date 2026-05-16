"""Composite per-section editors used by the phase form.

The phase-blocks dock's inline form (MVP-1/2 covered the common fields)
delegates to four editors defined here for the deeper sections:

* :class:`GaitEditor` — walking / stride choice + per-kind fields.
* :class:`TranslationEditor` — XYZ value curves OR a stair-translation
  block, picked via a kind combo.
* :class:`BonesEditor` — table of bone name × (x / y / z curve) cells.
* :class:`MorphsEditor` — table of morph name × curve cells.

Each editor follows the same contract as :class:`CurveEditor`:

* ``value()`` returns a JSON-friendly representation matching the
  schema's expected shape.
* ``set_value(value)`` repopulates from any accepted shape.
* ``changed`` signal fires on user edits (suppressed during
  ``set_value`` to avoid bouncing back into the document).
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from posecascade.ui.curve_editor import CurveEditor

# Bounds tuned for the phase form spin boxes. Matches the limits in
# :mod:`phase_blocks_dock`; centralising would require a constants
# module, which we'll add when a fifth caller appears.
_ROTATION_LIMIT = 100.0
_DURATION_MIN_SEC = 0.05
_DURATION_MAX_SEC = 300.0
_AXES = ("x_rad", "y_rad", "z_rad")
_AXIS_LABELS = ("x", "y", "z")


# ---------------------------------------------------------------------------
# Gait editor
# ---------------------------------------------------------------------------


class GaitEditor(QGroupBox):
    """Kind picker + walking / stride field set.

    Emits ``changed`` with the assembled dict (or ``None`` when the
    "(no gait)" kind is selected). ``None`` translates to removing the
    ``gait`` field from the phase entirely.
    """

    changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Gait", parent)
        self._kind = QComboBox()
        self._kind.addItems(["(none)", "walking", "stride"])
        # walking shared with stride
        self._step_cycle = _spin(_DURATION_MIN_SEC, _DURATION_MAX_SEC, step=0.1)
        self._leg_swing = _spin(-_ROTATION_LIMIT, _ROTATION_LIMIT)
        self._knee_bend = _spin(-_ROTATION_LIMIT, _ROTATION_LIMIT)
        self._arm_swing = _spin(-_ROTATION_LIMIT, _ROTATION_LIMIT)
        self._arm_hang = _spin(-_ROTATION_LIMIT, _ROTATION_LIMIT)
        # stride-only
        self._step_count = QSpinBox()
        self._step_count.setRange(1, 32)
        self._leading_lift = _spin(-_ROTATION_LIMIT, _ROTATION_LIMIT)
        self._trailing_back = _spin(-_ROTATION_LIMIT, _ROTATION_LIMIT)
        self._knee_bend_stride = _spin(-_ROTATION_LIMIT, _ROTATION_LIMIT)
        self._arm_swing_stride = _spin(-_ROTATION_LIMIT, _ROTATION_LIMIT)

        self._loading = False

        layout = QFormLayout(self)
        layout.addRow("Kind:", self._kind)
        layout.addRow("Step cycle (s):", self._step_cycle)
        layout.addRow("Leg swing:", self._leg_swing)
        layout.addRow("Knee bend:", self._knee_bend)
        layout.addRow("Arm swing:", self._arm_swing)
        layout.addRow("Arm hang:", self._arm_hang)
        layout.addRow("Step count:", self._step_count)
        layout.addRow("Leading lift:", self._leading_lift)
        layout.addRow("Trailing back:", self._trailing_back)
        layout.addRow("Knee bend (stride):", self._knee_bend_stride)
        layout.addRow("Arm swing (stride):", self._arm_swing_stride)
        self._layout = layout

        self._wire_signals()
        self.set_value(None)

    def value(self) -> Any:
        kind = self._kind.currentText()
        if kind == "(none)":
            return None
        if kind == "walking":
            return {
                "kind": "walking",
                "step_cycle_sec": float(self._step_cycle.value()),
                "leg_swing_amplitude": float(self._leg_swing.value()),
                "knee_bend": float(self._knee_bend.value()),
                "arm_swing_amplitude": float(self._arm_swing.value()),
                "arm_hang_rad": float(self._arm_hang.value()),
            }
        return {
            "kind": "stride",
            "step_count": int(self._step_count.value()),
            "leading_lift_rad": float(self._leading_lift.value()),
            "trailing_back_rad": float(self._trailing_back.value()),
            "knee_bend_rad": float(self._knee_bend_stride.value()),
            "arm_swing_amplitude_rad": float(self._arm_swing_stride.value()),
            "arm_hang_rad": float(self._arm_hang.value()),
        }

    def set_value(self, value: Any) -> None:
        self._loading = True
        try:
            if not isinstance(value, dict):
                self._kind.setCurrentText("(none)")
                self._reveal_for("(none)")
                return
            kind = value.get("kind", "walking")
            self._kind.setCurrentText(kind)
            if kind == "walking":
                self._step_cycle.setValue(
                    _coerce_float(value.get("step_cycle_sec"), 1.0),
                )
                self._leg_swing.setValue(
                    _coerce_float(value.get("leg_swing_amplitude"), 0.0),
                )
                self._knee_bend.setValue(
                    _coerce_float(value.get("knee_bend"), 0.0),
                )
                self._arm_swing.setValue(
                    _coerce_float(value.get("arm_swing_amplitude"), 0.0),
                )
                self._arm_hang.setValue(
                    _coerce_float(value.get("arm_hang_rad"), 0.0),
                )
            elif kind == "stride":
                self._step_count.setValue(int(value.get("step_count", 1)))
                self._leading_lift.setValue(
                    _coerce_float(value.get("leading_lift_rad"), 0.0),
                )
                self._trailing_back.setValue(
                    _coerce_float(value.get("trailing_back_rad"), 0.0),
                )
                self._knee_bend_stride.setValue(
                    _coerce_float(value.get("knee_bend_rad"), 0.0),
                )
                self._arm_swing_stride.setValue(
                    _coerce_float(value.get("arm_swing_amplitude_rad"), 0.0),
                )
                self._arm_hang.setValue(
                    _coerce_float(value.get("arm_hang_rad"), 0.0),
                )
            self._reveal_for(kind)
        finally:
            self._loading = False

    def _wire_signals(self) -> None:
        self._kind.currentTextChanged.connect(self._on_kind_changed)
        for widget in (
            self._step_cycle, self._leg_swing, self._knee_bend, self._arm_swing,
            self._arm_hang, self._leading_lift, self._trailing_back,
            self._knee_bend_stride, self._arm_swing_stride,
        ):
            widget.valueChanged.connect(self._emit)
        self._step_count.valueChanged.connect(self._emit)

    def _on_kind_changed(self, kind: str) -> None:
        self._reveal_for(kind)
        self._emit()

    def _reveal_for(self, kind: str) -> None:
        """Walking and stride share the arm-hang row; everything else is per-kind.

        Hiding via :meth:`QFormLayout.setRowVisible` keeps the dock
        height compact when the author hasn't authored a gait yet.
        """
        walking_rows = {1, 2, 3, 4, 5}      # step_cycle … arm_hang
        stride_rows = {5, 6, 7, 8, 9, 10}   # arm_hang shared, then stride-only
        visible = {0}  # kind row always visible
        if kind == "walking":
            visible |= walking_rows
        elif kind == "stride":
            visible |= stride_rows
        # ``setRowVisible`` is available on Qt 6.4+.
        for row in range(self._layout.rowCount()):
            self._layout.setRowVisible(row, row in visible)

    def _emit(self) -> None:
        if self._loading:
            return
        self.changed.emit(self.value())


# ---------------------------------------------------------------------------
# Translation editor
# ---------------------------------------------------------------------------


class TranslationEditor(QGroupBox):
    """XYZ value curves *or* a stair-translation block.

    The kind combo toggles between ``"xyz"`` (three :class:`CurveEditor`
    rows) and ``"stair"`` (a small form mirroring the schema's
    stair_translation fields). Returning value matches what the schema
    accepts: an ``{x, y, z}`` dict in the XYZ case, or a ``{"stair": …}``
    dict in the stair case.
    """

    changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Body translation", parent)
        self._kind = QComboBox()
        self._kind.addItems(["xyz", "stair"])
        self._x = CurveEditor()
        self._y = CurveEditor()
        self._z = CurveEditor()
        self._base_z = _spin(-_ROTATION_LIMIT, _ROTATION_LIMIT)
        self._rise = _spin(-_ROTATION_LIMIT, _ROTATION_LIMIT)
        self._forward = _spin(-_ROTATION_LIMIT, _ROTATION_LIMIT)
        self._step_count = QSpinBox()
        self._step_count.setRange(1, 64)

        self._loading = False

        layout = QFormLayout(self)
        layout.addRow("Kind:", self._kind)
        layout.addRow("X:", self._x)
        layout.addRow("Y:", self._y)
        layout.addRow("Z:", self._z)
        layout.addRow("Stair base Z:", self._base_z)
        layout.addRow("Stair rise:", self._rise)
        layout.addRow("Stair forward:", self._forward)
        layout.addRow("Stair step count:", self._step_count)
        self._layout = layout

        self._wire_signals()
        self.set_value({})

    def value(self) -> Any:
        kind = self._kind.currentText()
        if kind == "xyz":
            return {
                "x": self._x.value(),
                "y": self._y.value(),
                "z": self._z.value(),
            }
        return {
            "stair": {
                "base_z": float(self._base_z.value()),
                "rise": float(self._rise.value()),
                "forward": float(self._forward.value()),
                "step_count": int(self._step_count.value()),
            },
        }

    def set_value(self, value: Any) -> None:
        self._loading = True
        try:
            if isinstance(value, list):
                # ``[x, y, z]`` shorthand.
                pad = list(value) + [0.0] * (3 - len(value))
                self._kind.setCurrentText("xyz")
                self._x.set_value(pad[0])
                self._y.set_value(pad[1])
                self._z.set_value(pad[2])
                self._reveal_for("xyz")
                return
            if not isinstance(value, dict):
                value = {}
            if "stair" in value:
                self._kind.setCurrentText("stair")
                stair = value["stair"] or {}
                self._base_z.setValue(_coerce_float(stair.get("base_z"), 0.0))
                self._rise.setValue(_coerce_float(stair.get("rise"), 0.0))
                self._forward.setValue(_coerce_float(stair.get("forward"), 0.0))
                self._step_count.setValue(int(stair.get("step_count", 1)))
            else:
                self._kind.setCurrentText("xyz")
                self._x.set_value(value.get("x", 0.0))
                self._y.set_value(value.get("y", 0.0))
                self._z.set_value(value.get("z", 0.0))
            self._reveal_for(self._kind.currentText())
        finally:
            self._loading = False

    def _wire_signals(self) -> None:
        self._kind.currentTextChanged.connect(self._on_kind_changed)
        for curve in (self._x, self._y, self._z):
            curve.changed.connect(lambda _value: self._emit())
        for widget in (self._base_z, self._rise, self._forward):
            widget.valueChanged.connect(self._emit)
        self._step_count.valueChanged.connect(self._emit)

    def _on_kind_changed(self, kind: str) -> None:
        self._reveal_for(kind)
        self._emit()

    def _reveal_for(self, kind: str) -> None:
        xyz_rows = {1, 2, 3}
        stair_rows = {4, 5, 6, 7}
        visible = {0}
        visible |= xyz_rows if kind == "xyz" else stair_rows
        for row in range(self._layout.rowCount()):
            self._layout.setRowVisible(row, row in visible)

    def _emit(self) -> None:
        if self._loading:
            return
        self.changed.emit(self.value())


# ---------------------------------------------------------------------------
# Bones editor — table of bone × axis curve cells.
# ---------------------------------------------------------------------------


class BonesEditor(QGroupBox):
    """Per-bone per-axis curve table.

    Each row is one bone name; the four columns are bone name and the
    three axes (x / y / z). Clicking an axis cell opens a transient
    :class:`CurveEditor` popup (rendered as an inline child below the
    table when the cell is selected) — keeps the table compact while
    still letting authors edit any curve shape, not just constants.
    """

    changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Bones", parent)
        self._table = QTableWidget(0, _BONE_TABLE_COLUMNS)
        self._table.setHorizontalHeaderLabels(["Bone", "x", "y", "z"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch,
        )
        self._add_btn = QPushButton("+ Bone")
        self._del_btn = QPushButton("Remove")
        self._curve_editor = CurveEditor()
        self._curve_label = QLineEdit()
        self._curve_label.setReadOnly(True)

        self._loading = False
        self._current_cell: tuple[int, int] | None = None

        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self._add_btn)
        toolbar.addWidget(self._del_btn)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)
        layout.addWidget(self._table)
        layout.addWidget(self._curve_label)
        layout.addWidget(self._curve_editor)
        self._curve_editor.setEnabled(False)

        self._wire_signals()

    def value(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, 0)
            if name_item is None or not name_item.text().strip():
                continue
            name = name_item.text().strip()
            axes: dict[str, Any] = {}
            for col, axis_key in enumerate(_AXES, start=1):
                cell = self._table.item(row, col)
                if cell is None:
                    continue
                payload = cell.data(_AXIS_CURVE_ROLE)
                if payload is not None:
                    axes[axis_key] = payload
            if axes:
                out[name] = axes
        return out

    def set_value(self, value: Any) -> None:
        self._loading = True
        try:
            self._table.setRowCount(0)
            if not isinstance(value, dict):
                return
            for bone_name, axes in value.items():
                if not isinstance(axes, dict):
                    continue
                row = self._table.rowCount()
                self._table.insertRow(row)
                self._table.setItem(row, 0, QTableWidgetItem(str(bone_name)))
                for col, axis_key in enumerate(_AXES, start=1):
                    cell = QTableWidgetItem(_summarise_curve(axes.get(axis_key)))
                    if axis_key in axes:
                        cell.setData(_AXIS_CURVE_ROLE, axes[axis_key])
                    self._table.setItem(row, col, cell)
        finally:
            self._loading = False
        self._select_first()

    def _wire_signals(self) -> None:
        self._add_btn.clicked.connect(self._on_add)
        self._del_btn.clicked.connect(self._on_remove)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.currentCellChanged.connect(self._on_current_cell_changed)
        self._curve_editor.changed.connect(self._on_curve_edited)

    def _select_first(self) -> None:
        if self._table.rowCount() > 0:
            self._table.setCurrentCell(0, 1)

    def _on_add(self) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(f"bone_{row + 1}"))
        for col in range(1, _BONE_TABLE_COLUMNS):
            self._table.setItem(row, col, QTableWidgetItem(""))
        self._emit()

    def _on_remove(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        self._table.removeRow(row)
        self._emit()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        # Column 0 is the bone name; the others are curve cells and
        # their text content is read-only (set by the curve editor's
        # summariser).
        if item.column() == 0:
            self._emit()

    def _on_current_cell_changed(
        self, row: int, col: int, _prev_row: int, _prev_col: int,
    ) -> None:
        if col == 0 or row < 0:
            self._current_cell = None
            self._curve_editor.setEnabled(False)
            self._curve_label.setText("")
            return
        self._current_cell = (row, col)
        cell = self._table.item(row, col)
        axis = _AXIS_LABELS[col - 1]
        bone_name = self._table.item(row, 0)
        bone_text = bone_name.text() if bone_name else "(unnamed)"
        self._curve_label.setText(f"{bone_text}.{axis}")
        self._curve_editor.setEnabled(True)
        payload = cell.data(_AXIS_CURVE_ROLE) if cell is not None else None
        self._curve_editor.set_value(payload if payload is not None else 0.0)

    def _on_curve_edited(self, value: Any) -> None:
        if self._loading or self._current_cell is None:
            return
        row, col = self._current_cell
        cell = self._table.item(row, col)
        if cell is None:
            cell = QTableWidgetItem("")
            self._table.setItem(row, col, cell)
        cell.setData(_AXIS_CURVE_ROLE, value)
        cell.setText(_summarise_curve(value))
        self._emit()

    def _emit(self) -> None:
        if self._loading:
            return
        self.changed.emit(self.value())


_BONE_TABLE_COLUMNS = 4
_AXIS_CURVE_ROLE = 0x0100   # Qt.ItemDataRole.UserRole numeric value


def _summarise_curve(value: Any) -> str:
    """One-cell summary so the bone table reads at a glance."""
    summary_map: dict[type, Any] = {}
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.3g}"
    if isinstance(value, str):
        return value[:30] + ("…" if len(value) > _SUMMARY_MAX_CHARS else "")
    if isinstance(value, list):
        return f"[{value[0]} → {value[1]}]" if len(value) >= _LINEAR_LEN else "[]"
    if isinstance(value, dict):
        return value.get("kind", "?")
    _ = summary_map  # keep the lint-friendly local alive
    return "…"


_SUMMARY_MAX_CHARS = 30
_LINEAR_LEN = 2


# ---------------------------------------------------------------------------
# Morphs editor — name → curve table.
# ---------------------------------------------------------------------------


class MorphsEditor(QGroupBox):
    """Table of morph name → curve cells.

    Authoring shape mirrors :class:`BonesEditor`'s pattern: rows are
    names, a single curve column shows the summary, the selected row
    opens its curve in a child :class:`CurveEditor`.
    """

    changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Morphs", parent)
        self._table = QTableWidget(0, _MORPH_TABLE_COLUMNS)
        self._table.setHorizontalHeaderLabels(["Morph", "Curve"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch,
        )
        self._add_btn = QPushButton("+ Morph")
        self._del_btn = QPushButton("Remove")
        self._curve_editor = CurveEditor()
        self._curve_label = QLineEdit()
        self._curve_label.setReadOnly(True)
        self._loading = False
        self._current_row: int | None = None

        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self._add_btn)
        toolbar.addWidget(self._del_btn)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)
        layout.addWidget(self._table)
        layout.addWidget(self._curve_label)
        layout.addWidget(self._curve_editor)
        self._curve_editor.setEnabled(False)

        self._wire_signals()

    def value(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for row in range(self._table.rowCount()):
            name = self._table.item(row, 0)
            curve = self._table.item(row, 1)
            if name is None or not name.text().strip() or curve is None:
                continue
            payload = curve.data(_AXIS_CURVE_ROLE)
            if payload is not None:
                out[name.text().strip()] = payload
        return out

    def set_value(self, value: Any) -> None:
        self._loading = True
        try:
            self._table.setRowCount(0)
            if not isinstance(value, dict):
                return
            for morph_name, curve in value.items():
                row = self._table.rowCount()
                self._table.insertRow(row)
                self._table.setItem(row, 0, QTableWidgetItem(str(morph_name)))
                cell = QTableWidgetItem(_summarise_curve(curve))
                cell.setData(_AXIS_CURVE_ROLE, curve)
                self._table.setItem(row, 1, cell)
        finally:
            self._loading = False
        if self._table.rowCount() > 0:
            self._table.setCurrentCell(0, 1)

    def _wire_signals(self) -> None:
        self._add_btn.clicked.connect(self._on_add)
        self._del_btn.clicked.connect(self._on_remove)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.currentCellChanged.connect(self._on_current_cell_changed)
        self._curve_editor.changed.connect(self._on_curve_edited)

    def _on_add(self) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(f"morph_{row + 1}"))
        cell = QTableWidgetItem("0")
        cell.setData(_AXIS_CURVE_ROLE, 0.0)
        self._table.setItem(row, 1, cell)
        self._emit()

    def _on_remove(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        self._table.removeRow(row)
        self._emit()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        if item.column() == 0:
            self._emit()

    def _on_current_cell_changed(
        self, row: int, _col: int, _prev_row: int, _prev_col: int,
    ) -> None:
        if row < 0:
            self._current_row = None
            self._curve_editor.setEnabled(False)
            self._curve_label.setText("")
            return
        self._current_row = row
        morph_name = self._table.item(row, 0)
        self._curve_label.setText(
            morph_name.text() if morph_name else "(unnamed)",
        )
        cell = self._table.item(row, 1)
        payload = cell.data(_AXIS_CURVE_ROLE) if cell is not None else None
        self._curve_editor.setEnabled(True)
        self._curve_editor.set_value(payload if payload is not None else 0.0)

    def _on_curve_edited(self, value: Any) -> None:
        if self._loading or self._current_row is None:
            return
        cell = self._table.item(self._current_row, 1)
        if cell is None:
            cell = QTableWidgetItem("")
            self._table.setItem(self._current_row, 1, cell)
        cell.setData(_AXIS_CURVE_ROLE, value)
        cell.setText(_summarise_curve(value))
        self._emit()

    def _emit(self) -> None:
        if self._loading:
            return
        self.changed.emit(self.value())


_MORPH_TABLE_COLUMNS = 2


def _spin(low: float, high: float, *, step: float = 0.05) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setDecimals(4)
    spin.setSingleStep(step)
    return spin


def _coerce_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


__all__ = ["BonesEditor", "GaitEditor", "MorphsEditor", "TranslationEditor"]
