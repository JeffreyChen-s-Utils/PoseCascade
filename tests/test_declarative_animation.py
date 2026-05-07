"""Tests for the JSON-driven declarative animation runtime.

Stage 1 covers: schema parsing, scalar / curve resolution, phase
selection over the loop, body root translation + yaw, walking-gait
bone drivers, and the foot-planter / ground binding wired through
``floor`` if the document declares one. Stride / lock-target gait,
physics chain tunings and morph timelines are stage 2 — schema
fields for those parse without raising but the runtime no-ops on
them.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.scene.transform import Transform
from posecascade.scripting.declarative import (
    DeclarativeAnimationError,
    DeclarativeRuntime,
    load_animation,
    parse_animation,
)


def _node(name: str, **kw) -> Node:
    transform = Transform()
    if "translation" in kw:
        transform.set_translation(kw["translation"])
    if "rotation" in kw:
        transform.set_rotation(kw["rotation"])
    return Node(name=name, transform=transform)


def _build_minimal_scene() -> Scene:
    """Scene with a character root + the bones the walking gait writes to."""
    scene = Scene(name="test")
    root = _node("Sketchfab_model")
    for bone in (
        "upper_leg_L", "upper_leg_R",
        "lower_leg_L", "lower_leg_R",
        "upper_arm_L", "upper_arm_R",
        "head", "chest", "hip", "foot_L", "foot_R",
    ):
        root.add_child(_node(bone))
    scene.root.add_child(root)
    return scene


def _minimal_doc() -> dict:
    return {
        "schema_version": 1,
        "name": "minimal",
        "loop_sec": 2.0,
        "rig": {"character_root": "Sketchfab_model"},
        "phases": [
            {
                "name": "still",
                "duration_sec": 2.0,
                "body": {"yaw_rad": 0.0},
            },
        ],
    }


# --- Parsing ----------------------------------------------------------------


def test_parse_minimal_document_succeeds() -> None:
    """A document with one phase and no gait parses cleanly."""
    parsed = parse_animation(_minimal_doc())
    assert parsed.name == "minimal"
    assert parsed.loop_sec == 2.0
    assert len(parsed.phases) == 1
    assert parsed.phases[0].name == "still"


def test_parse_rejects_unsupported_schema_version() -> None:
    doc = _minimal_doc()
    doc["schema_version"] = 99
    with pytest.raises(DeclarativeAnimationError, match="schema_version"):
        parse_animation(doc)


def test_parse_rejects_empty_phases() -> None:
    doc = _minimal_doc()
    doc["phases"] = []
    with pytest.raises(DeclarativeAnimationError, match="at least one phase"):
        parse_animation(doc)


def test_parse_resolves_pi_symbolic() -> None:
    """The string 'pi' resolves to ``math.pi`` for body.yaw_rad etc."""
    doc = _minimal_doc()
    doc["phases"][0]["body"]["yaw_rad"] = "pi"
    parsed = parse_animation(doc)
    # yaw_rad stays as the raw spec; resolver runs at update time.
    assert parsed.phases[0].body_yaw_rad == "pi"


def test_parse_unknown_ground_kind_rejected() -> None:
    doc = _minimal_doc()
    doc["ground"] = {"kind": "swamp"}
    with pytest.raises(DeclarativeAnimationError, match="ground kind"):
        parse_animation(doc)


# --- Runtime body trajectory ------------------------------------------------


def test_runtime_applies_root_yaw_and_translation() -> None:
    """At phase end the root has the yaw quaternion and end-Z translation."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["phases"][0]["body"] = {
        "yaw_rad": "pi",
        "translation": {"z": {"kind": "linear", "from": 0.0, "to": -0.20}},
    }
    parsed = parse_animation(doc)
    t_now = [0.0]
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    # Step to phase end (loop is 2 s, phase is 2 s).
    t_now[0] = 1.999
    hooks["update"](0.0)
    root = scene.find("Sketchfab_model")
    np.testing.assert_allclose(
        root.transform.translation, [0.0, 0.0, -0.20 * (1.999 / 2.0)], atol=1e-3,
    )
    # Yaw=π → quaternion is roughly (0, 1, 0, 0).
    rot = root.transform.rotation
    assert abs(abs(rot[1]) - 1.0) < 1e-3, f"yaw≈π expected, got {rot}"


def test_runtime_walking_gait_drives_legs_alternately() -> None:
    """Walking gait sets opposite-sign rotations on the L / R upper legs."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["phases"][0]["gait"] = {
        "kind": "walking", "step_cycle_sec": 1.0,
        "leg_swing_amplitude": 0.5, "knee_bend": -0.3,
        "arm_swing_amplitude": 0.4,
    }
    parsed = parse_animation(doc)
    t_now = [0.0]
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    # At t=0.25 s with 1 s cycle: phase = π/2, sin(π/2) = 1 → leg_l=+0.5,
    # leg_r=-0.5. Bones are children of the root with identity rest, so
    # final rotation = world-X delta * identity = world-X delta itself.
    t_now[0] = 0.25
    hooks["update"](0.0)
    leg_l = scene.find("upper_leg_L").transform.rotation
    leg_r = scene.find("upper_leg_R").transform.rotation
    # Expect their X components to have opposite signs.
    assert leg_l[0] * leg_r[0] < 0, (
        f"L / R legs not opposite: leg_L={leg_l}, leg_R={leg_r}"
    )


# --- Loader -----------------------------------------------------------------


def test_load_animation_returns_hooks() -> None:
    scene = _build_minimal_scene()
    api = {
        "scene": scene,
        "time": lambda: 0.0,
    }
    source = json.dumps(_minimal_doc())
    hooks = load_animation(source, "minimal.json", api)
    assert set(hooks.keys()) == {"start", "update", "on_event"}
    hooks["start"]()  # smoke


def test_load_animation_rejects_invalid_json() -> None:
    api = {"scene": _build_minimal_scene(), "time": lambda: 0.0}
    with pytest.raises(DeclarativeAnimationError, match="failed to parse"):
        load_animation("{not json", "broken.json", api)
