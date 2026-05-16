"""Round-trip tests for the curve editor widget."""
from __future__ import annotations

import pytest


@pytest.fixture
def editor(qapp: object):
    from posecascade.ui.curve_editor import CurveEditor  # noqa: PLC0415

    return CurveEditor()


def test_scalar_round_trips_as_constant(editor) -> None:
    editor.set_value(0.5)
    assert editor.value() == 0.5


def test_expression_string_round_trips(editor) -> None:
    editor.set_value("0.1 * sin(elapsed * tau)")
    assert editor.value() == "0.1 * sin(elapsed * tau)"


def test_linear_array_round_trips(editor) -> None:
    editor.set_value([0.0, 1.5])
    assert editor.value() == [0.0, 1.5]


def test_dict_curve_with_overshoot_round_trips(editor) -> None:
    editor.set_value({"kind": "back-out", "from": 0.0, "to": 1.0, "overshoot": 1.7})
    out = editor.value()
    assert out["kind"] == "back-out"
    assert out["from"] == 0.0
    assert out["to"] == 1.0
    assert out["overshoot"] == pytest.approx(1.7, abs=1.0e-3)


def test_pulse_round_trips_center_and_width(editor) -> None:
    editor.set_value({
        "kind": "pulse", "from": 0.0, "to": 0.5, "center": 0.6, "width": 0.3,
    })
    out = editor.value()
    assert out["kind"] == "pulse"
    assert out["center"] == pytest.approx(0.6, abs=1.0e-3)
    assert out["width"] == pytest.approx(0.3, abs=1.0e-3)


def test_changed_signal_fires_on_kind_switch(editor) -> None:
    received: list[object] = []
    editor.changed.connect(received.append)
    editor.set_value(0.0)
    received.clear()
    editor._kind_combo.setCurrentText("linear")  # noqa: SLF001
    assert received  # at least one emission triggered by the kind change


def test_changed_signal_silenced_during_set_value(editor) -> None:
    """``set_value`` is a model-driven refresh; the form must not echo back."""
    received: list[object] = []
    editor.changed.connect(received.append)
    editor.set_value({"kind": "ease", "from": 0.0, "to": 1.0})
    assert received == []
