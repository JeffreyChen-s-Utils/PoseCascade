"""Slot list dock — shows all loaded models + per-slot controls.

The dock paints one row per :class:`~posecascade.scene.model_slot.ModelSlot`:

- a visibility checkbox (toggles ``slot.visible``),
- the slot name,
- spin boxes for the slot's world-space translation (``transform.translation``).

The dock emits ``slot_visibility_changed(name: str, visible: bool)`` and
``slot_translation_changed(name: str, x, y, z)`` so the integrator can
trigger a redraw / refresh the player. State edits go straight onto
the slot — the dock is the source of truth, not a snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from posecascade.i18n import t
from posecascade.scene.model_slot import ModelSlot, SceneSlots
from posecascade.utils.math3d import vec3

_TRANSLATION_RANGE = 1.0e6
_TRANSLATION_DECIMALS = 3
_TRANSLATION_STEP = 0.1


@dataclass
class _SlotRow:
    """Cached widget refs for one slot's row."""

    slot: ModelSlot
    list_item: QListWidgetItem
    visibility: QCheckBox
    translation_spins: tuple[QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox]


class SlotListDock(QDockWidget):
    """Left dock listing every loaded model slot."""

    slot_visibility_changed = Signal(str, bool)
    slot_translation_changed = Signal(str, float, float, float)

    def __init__(
        self,
        slots: SceneSlots,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(t("slot.title"), parent)
        self._slots = slots
        self._rows: list[_SlotRow] = []
        self._list = QListWidget()
        self.setWidget(self._build_container())
        self.refresh()

    # ----- public API --------------------------------------------------
    @property
    def slots(self) -> SceneSlots:
        return self._slots

    def refresh(self) -> None:
        """Rebuild the list view from the current :class:`SceneSlots`."""
        self._list.clear()
        self._rows.clear()
        for slot in self._slots:
            self._append_row(slot)

    # ----- internal ----------------------------------------------------
    def _build_container(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.addWidget(self._list, 1)
        return container

    def _append_row(self, slot: ModelSlot) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(4, 2, 4, 2)

        visibility = QCheckBox()
        visibility.setChecked(slot.visible)
        visibility.setToolTip(t("slot.visibility.tooltip", name=slot.name))
        visibility.toggled.connect(
            lambda checked, name=slot.name: self._on_visibility_toggled(name, checked),
        )
        row_layout.addWidget(visibility)

        row_layout.addWidget(QLabel(slot.name), 1)

        translation_spins = self._build_translation_spins(slot)
        axis_labels = ("X", "Y", "Z")
        for axis_label, spin in zip(axis_labels, translation_spins, strict=True):
            spin.setToolTip(
                t("slot.translation.tooltip", name=slot.name, axis=axis_label),
            )
            row_layout.addWidget(spin)

        item = QListWidgetItem()
        item.setSizeHint(row_widget.sizeHint())
        self._list.addItem(item)
        self._list.setItemWidget(item, row_widget)
        self._rows.append(
            _SlotRow(
                slot=slot,
                list_item=item,
                visibility=visibility,
                translation_spins=translation_spins,
            ),
        )

    def _build_translation_spins(
        self, slot: ModelSlot,
    ) -> tuple[QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox]:
        translation = slot.transform.translation
        spins: list[QDoubleSpinBox] = []
        for axis in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-_TRANSLATION_RANGE, _TRANSLATION_RANGE)
            spin.setDecimals(_TRANSLATION_DECIMALS)
            spin.setSingleStep(_TRANSLATION_STEP)
            spin.setValue(float(translation[axis]))
            spin.valueChanged.connect(
                lambda _value, name=slot.name: self._on_translation_changed(name),
            )
            spins.append(spin)
        return spins[0], spins[1], spins[2]

    def _on_visibility_toggled(self, slot_name: str, checked: bool) -> None:
        slot = self._slots.find(slot_name)
        if slot is None:
            return
        slot.visible = bool(checked)
        self.slot_visibility_changed.emit(slot_name, slot.visible)

    def _on_translation_changed(self, slot_name: str) -> None:
        row = next((r for r in self._rows if r.slot.name == slot_name), None)
        if row is None:
            return
        x = float(row.translation_spins[0].value())
        y = float(row.translation_spins[1].value())
        z = float(row.translation_spins[2].value())
        row.slot.transform.set_translation(vec3(x, y, z))
        self.slot_translation_changed.emit(slot_name, x, y, z)
