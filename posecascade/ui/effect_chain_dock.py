"""Effect chain dock — enable / reorder / tune the post-effect pipeline.

The dock works on a live :class:`EffectChain`: every UI gesture
mutates the chain in place + emits ``chain_changed`` so the integrator
can rebuild the GL pipeline. The chain itself is shared state — same
object the renderer's executor will read each frame.

Per-effect uniform editing is intentionally minimal: spin boxes for
scalar uniforms, a textbox for colours, a checkbox for booleans.
Anything fancier (gradient pickers, swatch palettes) belongs in a
separate inspector panel.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from posecascade.render.effects.chain import EffectChain
from posecascade.render.effects.descriptor import (
    EffectUniform,
    EffectUniformKind,
    UniformValue,
)


@dataclass
class _ChainRow:
    enabled: QCheckBox
    label: QLabel


_VECTOR_DECIMALS = 3
_VECTOR_STEP = 0.01


class EffectChainDock(QDockWidget):
    """Right-dock effect chain editor."""

    chain_changed = Signal()

    def __init__(self, chain: EffectChain, parent: QWidget | None = None) -> None:
        super().__init__("Effects", parent)
        self._chain = chain
        self._chain_list = QListWidget()
        self._uniforms_container = QWidget()
        self._uniforms_layout = QFormLayout(self._uniforms_container)
        self._rows: list[_ChainRow] = []
        self._build_ui()
        self.refresh()

    # ----- public API --------------------------------------------------
    @property
    def chain(self) -> EffectChain:
        return self._chain

    def refresh(self) -> None:
        """Rebuild the entry list + the uniform editor for the current row."""
        previous_row = self._chain_list.currentRow()
        self._chain_list.blockSignals(True)
        try:
            self._chain_list.clear()
            self._rows.clear()
            for index, entry in enumerate(self._chain.entries):
                self._add_chain_row(index, entry)
        finally:
            self._chain_list.blockSignals(False)
        if 0 <= previous_row < self._chain_list.count():
            self._chain_list.setCurrentRow(previous_row)
        elif self._chain_list.count() > 0:
            self._chain_list.setCurrentRow(0)
        self._refresh_uniforms()

    def selected_index(self) -> int:
        return int(self._chain_list.currentRow())

    def move_selected_up(self) -> None:
        index = self.selected_index()
        if index <= 0:
            return
        self._chain.move(index, index - 1)
        self._after_chain_edit()
        self._chain_list.setCurrentRow(index - 1)

    def move_selected_down(self) -> None:
        index = self.selected_index()
        if index < 0 or index >= len(self._chain) - 1:
            return
        self._chain.move(index, index + 1)
        self._after_chain_edit()
        self._chain_list.setCurrentRow(index + 1)

    def remove_selected(self) -> None:
        index = self.selected_index()
        if index < 0:
            return
        self._chain.remove_at(index)
        self._after_chain_edit()

    # ----- internal ----------------------------------------------------
    def _build_ui(self) -> None:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 4, 8, 8)

        list_row = QHBoxLayout()
        list_row.addWidget(self._chain_list, 1)

        controls = QVBoxLayout()
        self._up_button = QPushButton("↑")
        self._up_button.clicked.connect(self.move_selected_up)
        controls.addWidget(self._up_button)

        self._down_button = QPushButton("↓")
        self._down_button.clicked.connect(self.move_selected_down)
        controls.addWidget(self._down_button)

        self._remove_button = QPushButton("Remove")
        self._remove_button.clicked.connect(self.remove_selected)
        controls.addWidget(self._remove_button)

        controls.addStretch(1)
        list_row.addLayout(controls)
        layout.addLayout(list_row, 2)

        self._chain_list.currentRowChanged.connect(lambda _row: self._refresh_uniforms())
        self._uniforms_container.setLayout(self._uniforms_layout)
        layout.addWidget(self._uniforms_container, 3)

        self.setWidget(container)

    def _add_chain_row(self, index: int, entry) -> None:    # noqa: ANN001
        item = QListWidgetItem()
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(4, 2, 4, 2)
        enabled = QCheckBox()
        enabled.setChecked(entry.enabled)
        enabled.toggled.connect(
            lambda checked, idx=index: self._on_enable_toggled(idx, checked),
        )
        label = QLabel(entry.descriptor.name)
        row_layout.addWidget(enabled)
        row_layout.addWidget(label, 1)
        item.setSizeHint(row_widget.sizeHint())
        self._chain_list.addItem(item)
        self._chain_list.setItemWidget(item, row_widget)
        self._rows.append(_ChainRow(enabled=enabled, label=label))

    def _refresh_uniforms(self) -> None:
        # Clear the existing form rows.
        while self._uniforms_layout.rowCount() > 0:
            self._uniforms_layout.removeRow(0)
        index = self.selected_index()
        if index < 0 or index >= len(self._chain):
            return
        entry = self._chain.entries[index]
        for uniform in entry.descriptor.uniforms:
            widget = self._build_uniform_widget(index, uniform, entry.effective_value(uniform.name))
            if widget is not None:
                self._uniforms_layout.addRow(uniform.name, widget)

    def _build_uniform_widget(
        self, entry_index: int, uniform: EffectUniform, value: UniformValue | None,
    ) -> QWidget | None:
        if uniform.kind == EffectUniformKind.SCALAR:
            return self._build_scalar_spin(entry_index, uniform, value)
        if uniform.kind == EffectUniformKind.BOOL:
            return self._build_bool_check(entry_index, uniform, value)
        if uniform.kind in (EffectUniformKind.VEC3_COLOR, EffectUniformKind.VEC4_COLOR):
            return self._build_vector_row(entry_index, uniform, value)
        return None

    def _build_scalar_spin(
        self, entry_index: int, uniform: EffectUniform, value: UniformValue | None,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        if uniform.minimum is not None:
            spin.setMinimum(float(uniform.minimum))
        if uniform.maximum is not None:
            spin.setMaximum(float(uniform.maximum))
        if uniform.step is not None:
            spin.setSingleStep(float(uniform.step))
        spin.setValue(float(value if value is not None else uniform.default))   # type: ignore[arg-type]
        spin.valueChanged.connect(
            lambda v, name=uniform.name: self._set_uniform(entry_index, name, float(v)),
        )
        return spin

    def _build_bool_check(
        self, entry_index: int, uniform: EffectUniform, value: UniformValue | None,
    ) -> QCheckBox:
        check = QCheckBox()
        check.setChecked(bool(value if value is not None else uniform.default))
        check.toggled.connect(
            lambda checked, name=uniform.name: self._set_uniform(entry_index, name, bool(checked)),
        )
        return check

    def _build_vector_row(
        self, entry_index: int, uniform: EffectUniform, value: UniformValue | None,
    ) -> QWidget:
        host = QWidget()
        layout = QHBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        components: list[QDoubleSpinBox] = []
        current = tuple(value if value is not None else uniform.default)        # type: ignore[arg-type]
        for component_value in current:
            spin = QDoubleSpinBox()
            spin.setRange(-1.0e6, 1.0e6)
            spin.setDecimals(_VECTOR_DECIMALS)
            spin.setSingleStep(_VECTOR_STEP)
            spin.setValue(float(component_value))
            spin.valueChanged.connect(
                lambda _v, name=uniform.name, spins=components:
                    self._set_uniform(
                        entry_index, name,
                        tuple(float(s.value()) for s in spins),
                    ),
            )
            components.append(spin)
            layout.addWidget(spin)
        return host

    def _on_enable_toggled(self, entry_index: int, checked: bool) -> None:
        self._chain.set_enabled(entry_index, checked)
        self.chain_changed.emit()

    def _set_uniform(
        self, entry_index: int, uniform_name: str, value: UniformValue,
    ) -> None:
        self._chain.set_uniform(entry_index, uniform_name, value)
        self.chain_changed.emit()

    def _after_chain_edit(self) -> None:
        self.refresh()
        self.chain_changed.emit()
