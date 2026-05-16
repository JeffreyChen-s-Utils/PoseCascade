"""Tests for the gait / translation / bones / morphs editors."""
from __future__ import annotations

import pytest


@pytest.fixture
def gait(qapp: object):
    from posecascade.ui.phase_editors import GaitEditor  # noqa: PLC0415

    return GaitEditor()


@pytest.fixture
def translation(qapp: object):
    from posecascade.ui.phase_editors import TranslationEditor  # noqa: PLC0415

    return TranslationEditor()


@pytest.fixture
def bones(qapp: object):
    from posecascade.ui.phase_editors import BonesEditor  # noqa: PLC0415

    return BonesEditor()


@pytest.fixture
def morphs(qapp: object):
    from posecascade.ui.phase_editors import MorphsEditor  # noqa: PLC0415

    return MorphsEditor()


# --- GaitEditor ----------------------------------------------------------


def test_gait_none_round_trips(gait) -> None:
    gait.set_value(None)
    assert gait.value() is None


def test_gait_walking_round_trips(gait) -> None:
    gait.set_value({
        "kind": "walking",
        "step_cycle_sec": 1.2,
        "leg_swing_amplitude": 0.3,
        "knee_bend": -0.2,
        "arm_swing_amplitude": 0.25,
        "arm_hang_rad": -0.5,
    })
    out = gait.value()
    assert out["kind"] == "walking"
    assert out["step_cycle_sec"] == pytest.approx(1.2, abs=1.0e-3)
    assert out["leg_swing_amplitude"] == pytest.approx(0.3, abs=1.0e-3)


def test_gait_stride_round_trips(gait) -> None:
    gait.set_value({
        "kind": "stride", "step_count": 4,
        "leading_lift_rad": -0.4, "trailing_back_rad": 0.15,
        "knee_bend_rad": 0.2, "arm_swing_amplitude_rad": 0.3,
        "arm_hang_rad": -0.5,
    })
    out = gait.value()
    assert out["kind"] == "stride"
    assert out["step_count"] == 4
    assert out["leading_lift_rad"] == pytest.approx(-0.4, abs=1.0e-3)


# --- TranslationEditor ---------------------------------------------------


def test_translation_xyz_round_trips(translation) -> None:
    translation.set_value({"x": 0.0, "y": 0.5, "z": "0.005 * sin(elapsed)"})
    out = translation.value()
    assert out["x"] == 0.0
    assert out["y"] == 0.5
    assert out["z"] == "0.005 * sin(elapsed)"


def test_translation_list_shorthand_round_trips(translation) -> None:
    translation.set_value([0.0, 0.0, 1.5])
    out = translation.value()
    # The XYZ editor stores per-axis curves, so the round-trip lands in
    # the dict form. Both shapes are accepted by the runtime.
    assert out["x"] == 0.0
    assert out["z"] == 1.5


def test_translation_stair_round_trips(translation) -> None:
    translation.set_value({
        "stair": {
            "base_z": -0.2, "rise": 0.1, "forward": 0.2, "step_count": 5,
        },
    })
    out = translation.value()
    assert "stair" in out
    assert out["stair"]["step_count"] == 5
    assert out["stair"]["rise"] == pytest.approx(0.1, abs=1.0e-3)


# --- BonesEditor ---------------------------------------------------------


def test_bones_round_trip_preserves_axis_curves(bones) -> None:
    initial = {
        "head":  {"x_rad": "0.04 * sin(elapsed * tau)", "y_rad": 0.2},
        "chest": {"x_rad": 0.025},
    }
    bones.set_value(initial)
    out = bones.value()
    assert "head" in out
    assert out["head"]["x_rad"] == "0.04 * sin(elapsed * tau)"
    assert out["head"]["y_rad"] == 0.2
    assert out["chest"]["x_rad"] == 0.025


def test_bones_add_row(bones) -> None:
    bones.set_value({})
    bones._on_add()  # noqa: SLF001
    assert bones._table.rowCount() == 1  # noqa: SLF001
    # Empty curves don't emit a key — the row needs at least one axis
    # set to land in the output. Until then ``value()`` is empty.
    assert bones.value() == {}


# --- MorphsEditor --------------------------------------------------------


def test_morphs_round_trip(morphs) -> None:
    initial = {"smile": 0.5, "blink": "max(0, sin(elapsed))"}
    morphs.set_value(initial)
    out = morphs.value()
    assert out["smile"] == 0.5
    assert out["blink"] == "max(0, sin(elapsed))"
