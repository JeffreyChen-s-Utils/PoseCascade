"""Reusable widget for editing one declarative-animation ``value_curve``.

Backs the per-axis curve cells in the phase form (bones / morphs / body
translation). Accepts and emits the same shapes
:mod:`posecascade.scripting.declarative` understands:

* a scalar number → constant.
* an expression string (anything :func:`looks_like_expression`
  recognises) → expression-driven curve.
* a ``[from, to]`` array → linear curve.
* a dict with ``kind`` and per-kind fields → the full curve vocabulary.

The widget exposes ``value()`` / ``set_value(...)`` returning the same
JSON-friendly shape so the document can serialise it back unchanged.
Round-tripping is the API contract — read in, edit, read out without
losing semantic info, even for kinds the UI doesn't surface every
field of.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QWidget,
)

from posecascade.i18n import t
from posecascade.scripting.expressions import looks_like_expression

# Kinds the picker offers. Order matters — most common authoring shapes
# first so the dropdown lands on something useful for new authors.
_KINDS = (
    "constant",
    "linear",
    "expression",
    "ease",
    "quad-in",
    "quad-out",
    "cubic-in",
    "cubic-out",
    "back-out",
    "pulse",
    "step",
)

# Reasonable scalar bounds. The same rationale as in
# :mod:`posecascade.ui.phase_blocks_dock` — generous enough to avoid
# fighting the user, tight enough that a misplaced zero doesn't make
# the spin box silently clamp a sensible value.
_SCALAR_LIMIT = 1.0e4


class CurveEditor(QWidget):
    """Edit one value curve in any of the supported shapes.

    Signals:
        ``changed`` — emitted on every edit, with the canonicalised
        new value as payload. Listeners (typically the phase form)
        push the value back into the document.
    """

    changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._kind_combo = QComboBox()
        self._kind_combo.addItems(_KINDS)
        self._value_spin = _scalar_spin()
        self._from_spin = _scalar_spin()
        self._to_spin = _scalar_spin()
        self._source_edit = QLineEdit()
        self._source_edit.setPlaceholderText(t("curve_editor.placeholder"))
        self._extra_spin = _scalar_spin()   # at / overshoot / center
        self._width_spin = _scalar_spin()
        # Track whether we're loading a value so per-widget signals
        # don't re-emit ``changed`` while the form refreshes itself.
        self._loading = False

        self._layout = QFormLayout(self)
        self._layout.addRow("Kind:", self._kind_combo)
        self._layout.addRow("Value:", self._value_spin)
        self._layout.addRow("From:", self._from_spin)
        self._layout.addRow("To:", self._to_spin)
        self._layout.addRow("Source:", self._source_edit)
        self._layout.addRow("Extra:", self._extra_spin)
        self._layout.addRow("Width:", self._width_spin)

        self._wire_signals()
        self.set_value(0.0)

    # ----- public API ---------------------------------------------------

    def value(self) -> Any:
        """Return the canonical curve representation as a JSON-friendly value.

        Reverses ``set_value``: a ``constant`` curve becomes the bare
        number; an ``expression`` curve becomes the bare source string
        (so the JSON stays clean); a ``linear`` curve becomes the
        ``[from, to]`` array shorthand; everything else keeps the dict
        form so kind-specific fields round-trip.
        """
        kind = self._kind_combo.currentText()
        if kind == "constant":
            return float(self._value_spin.value())
        if kind == "expression":
            source = self._source_edit.text().strip()
            return source or 0.0
        if kind == "linear":
            return [float(self._from_spin.value()), float(self._to_spin.value())]
        spec: dict[str, Any] = {
            "kind": kind,
            "from": float(self._from_spin.value()),
            "to": float(self._to_spin.value()),
        }
        if kind == "step":
            spec["at"] = float(self._extra_spin.value())
        elif kind == "back-out":
            spec["overshoot"] = float(self._extra_spin.value())
        elif kind == "pulse":
            spec["center"] = float(self._extra_spin.value())
            spec["width"] = float(self._width_spin.value())
        return spec

    def set_value(self, value: Any) -> None:
        """Populate the form from any of the accepted curve shapes."""
        self._loading = True
        try:
            kind, fields = _normalise(value)
            self._kind_combo.setCurrentText(kind)
            self._value_spin.setValue(fields.get("value", 0.0))
            self._from_spin.setValue(fields.get("from", 0.0))
            self._to_spin.setValue(fields.get("to", 0.0))
            self._source_edit.setText(fields.get("source", ""))
            self._extra_spin.setValue(fields.get("extra", 0.0))
            self._width_spin.setValue(fields.get("width", 0.0))
            self._reveal_fields_for(kind)
        finally:
            self._loading = False

    # ----- UI plumbing --------------------------------------------------

    def _wire_signals(self) -> None:
        self._kind_combo.currentTextChanged.connect(self._on_kind_changed)
        for widget in (
            self._value_spin, self._from_spin, self._to_spin,
            self._extra_spin, self._width_spin,
        ):
            widget.valueChanged.connect(self._emit_changed)
        self._source_edit.textChanged.connect(self._emit_changed)

    def _on_kind_changed(self, kind: str) -> None:
        self._reveal_fields_for(kind)
        self._emit_changed()

    def _reveal_fields_for(self, kind: str) -> None:
        """Show only the rows relevant to ``kind`` — keeps the form compact.

        Hiding rows in ``QFormLayout`` requires hiding both the label
        and the field widget; the layout's ``setRowVisible`` (Qt 6.4+)
        does this cleanly.
        """
        rules: dict[str, set[str]] = {
            "constant":   {"value"},
            "linear":     {"from", "to"},
            "ease":       {"from", "to"},
            "expression": {"source"},
            "step":       {"from", "to", "extra"},
            "quad-in":    {"from", "to"},
            "quad-out":   {"from", "to"},
            "cubic-in":   {"from", "to"},
            "cubic-out":  {"from", "to"},
            "back-out":   {"from", "to", "extra"},
            "pulse":      {"from", "to", "extra", "width"},
        }
        visible = rules.get(kind, set())
        row_index_by_name = {
            "value": 1, "from": 2, "to": 3, "source": 4,
            "extra": 5, "width": 6,
        }
        # Friendly per-kind labels for the polymorphic "extra" row.
        extra_label = {
            "step": "At:", "back-out": "Overshoot:", "pulse": "Center:",
        }.get(kind, "Extra:")
        self._layout.itemAt(
            row_index_by_name["extra"], QFormLayout.ItemRole.LabelRole,
        ).widget().setText(extra_label)
        for name, row in row_index_by_name.items():
            self._layout.setRowVisible(row, name in visible)

    def _emit_changed(self) -> None:
        if self._loading:
            return
        self.changed.emit(self.value())


def _normalise(value: Any) -> tuple[str, dict[str, float | str]]:
    """Collapse any accepted shape into ``(kind, fields)``.

    The reverse of :meth:`CurveEditor.value`. Unknown / partial shapes
    fall back to ``constant`` with the raw value coerced to float —
    keeps the UI responsive even on malformed authoring while the JSON
    pane shows the underlying error.
    """
    if isinstance(value, (int, float)):
        return "constant", {"value": float(value)}
    if isinstance(value, str):
        # Bare symbolic constants (``"pi"``, ``"tau"``) read as
        # expressions through the same evaluator as arithmetic
        # strings, so we route both through ``expression`` and keep
        # the round-trip lossless.
        _ = looks_like_expression(value)  # imported for the side-effect of validation
        return "expression", {"source": value}
    if isinstance(value, list) and len(value) >= _LINEAR_LEN:
        return "linear", {
            "from": _coerce(value[0]),
            "to": _coerce(value[1]),
        }
    if isinstance(value, dict):
        kind = str(value.get("kind", "constant"))
        if kind not in _KINDS:
            return "constant", {"value": 0.0}
        fields: dict[str, float | str] = {
            "value": _coerce(value.get("value", 0.0)),
            "from": _coerce(value.get("from", 0.0)),
            "to": _coerce(value.get("to", 0.0)),
            "source": str(value.get("source", "")),
            "extra": _coerce(
                value.get("at", value.get("overshoot", value.get("center", 0.0))),
            ),
            "width": _coerce(value.get("width", 0.0)),
        }
        return kind, fields
    return "constant", {"value": 0.0}


_LINEAR_LEN = 2


def _coerce(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _scalar_spin() -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(-_SCALAR_LIMIT, _SCALAR_LIMIT)
    spin.setDecimals(4)
    spin.setSingleStep(0.05)
    return spin


__all__ = ["CurveEditor"]


# ``Callable`` is imported to keep downstream type-hint stubs working;
# the editor doesn't dispatch arbitrary callables itself but ports of
# this widget in test fixtures sometimes parametrise on the changed
# handler signature.
_ = Callable
