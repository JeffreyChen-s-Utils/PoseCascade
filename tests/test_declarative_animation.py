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
import math

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
from posecascade.utils.math3d import vec3


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


def test_schema_file_loads_and_describes_v1() -> None:
    """``schemas/animation_v1.json`` is well-formed JSON and pins the
    schema_version constant. Acts as a smoke check that the schema
    file we ship matches the runtime's parser version."""
    from pathlib import Path  # noqa: PLC0415
    schema_path = Path(__file__).parent.parent / "schemas" / "animation_v1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == 1
    assert "phases" in schema["properties"]
    assert "rig" in schema["properties"]
    assert "ground" in schema["properties"]
    assert schema["$defs"]["gait"]["oneOf"][0]["properties"]["kind"]["const"] == "walking"
    assert schema["$defs"]["gait"]["oneOf"][1]["properties"]["kind"]["const"] == "stride"


def test_load_animation_rejects_invalid_json() -> None:
    api = {"scene": _build_minimal_scene(), "time": lambda: 0.0}
    with pytest.raises(DeclarativeAnimationError, match="failed to parse"):
        load_animation("{not json", "broken.json", api)


# --- Stage 2: stride gait + stair body trajectory ---------------------------


def test_stride_gait_alternates_leading_leg_per_step() -> None:
    """Stride parity flips each step — leading_l on step 0/2/4, trailing on 1/3.

    Verifies by stepping into a stride with step_count=5 and a non-zero
    forward bell at the midpoint of step_t, then checking which side has
    the larger upper-leg X rotation.
    """
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["loop_sec"] = 1.0
    doc["phases"][0]["duration_sec"] = 1.0
    doc["phases"][0]["gait"] = {
        "kind": "stride", "step_count": 5,
        "leading_lift_rad": -0.70, "trailing_back_rad": 0.25,
        "knee_bend_rad": 0.30, "arm_swing_amplitude_rad": 0.40,
        "knee_bell": [0.10, 0.65], "forward_bell": [0.10, 0.65],
    }
    parsed = parse_animation(doc)
    t_now = [0.0]
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    # phase_t = 0.075 → step_pos = 0.375 → step_idx 0 (leading_l=True),
    # step_t = 0.375. Within bell window. L upper should have larger
    # |angle| than R upper because leading_lift > trailing_back.
    t_now[0] = 0.075
    hooks["update"](0.0)
    leg_l = scene.find("upper_leg_L").transform.rotation
    leg_r = scene.find("upper_leg_R").transform.rotation
    assert abs(leg_l[0]) > abs(leg_r[0]), (
        f"leading_l=True: |L|={abs(leg_l[0])} not > |R|={abs(leg_r[0])}"
    )
    # phase_t corresponding to step_idx=1 → leading_l=False → R should
    # be the leading leg (larger |angle|).
    t_now[0] = 0.275  # step_pos = 1.375
    hooks["update"](0.0)
    leg_l = scene.find("upper_leg_L").transform.rotation
    leg_r = scene.find("upper_leg_R").transform.rotation
    assert abs(leg_r[0]) > abs(leg_l[0]), (
        f"leading_l=False: |R|={abs(leg_r[0])} not > |L|={abs(leg_l[0])}"
    )


def test_stair_translation_climb_advances_y_z() -> None:
    """``stair`` body translation kind sweeps body Y up and Z forward
    over the phase, in step-quantised Y rises with cosine-ease in
    each step's rise window."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["loop_sec"] = 5.0
    doc["phases"][0]["duration_sec"] = 5.0
    doc["phases"][0]["body"] = {
        "translation": {
            "stair": {
                "base_z": -0.20, "rise": 0.10, "forward": 0.20,
                "step_count": 5, "ascending": True,
                "rise_window": [0.40, 0.75], "forward_sign": -1,
            },
        },
    }
    parsed = parse_animation(doc)
    t_now = [0.0]
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    # Beginning of phase: body at base_z, base_y=0.
    t_now[0] = 0.001
    hooks["update"](0.0)
    root = scene.find("Sketchfab_model")
    assert root.transform.translation[1] == 0.0
    np.testing.assert_allclose(root.transform.translation[2], -0.20, atol=1e-3)
    # Mid loop: body Y > 0, Z further into stairs.
    t_now[0] = 2.5  # phase_t = 0.5
    hooks["update"](0.0)
    assert root.transform.translation[1] > 0.0, "body Y did not rise into climb"
    assert root.transform.translation[2] < -0.20, "body Z did not advance into stairs"


def test_stair_translation_descend_returns_z_to_base() -> None:
    """Descending body starts at top stair Z and walks back toward base."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["loop_sec"] = 4.0
    doc["phases"][0]["duration_sec"] = 4.0
    doc["phases"][0]["body"] = {
        "translation": {
            "stair": {
                "base_z": -0.20, "rise": 0.10, "forward": 0.20,
                "step_count": 5, "ascending": False,
                "rise_window": [0.40, 0.75], "forward_sign": -1,
            },
        },
    }
    parsed = parse_animation(doc)
    t_now = [0.0]
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    # Start of descend: at top of stairs (base_z + forward_sign * forward).
    t_now[0] = 0.001
    hooks["update"](0.0)
    root = scene.find("Sketchfab_model")
    np.testing.assert_allclose(root.transform.translation[2], -0.40, atol=1e-3)
    # End of descend: back at base (-0.20).
    t_now[0] = 3.999
    hooks["update"](0.0)
    np.testing.assert_allclose(root.transform.translation[2], -0.20, atol=1e-3)


def test_stride_lock_alternates_trailing_side_per_step() -> None:
    """At each stride boundary the trailing-foot lock switches sides:
    step 0 (leading_l=True) snapshots R, step 1 snapshots L, etc.
    The opposite side's lock is cleared so the leading leg can swing
    freely."""
    scene = _build_minimal_scene()
    # Add proper hierarchical chain so analytical IK has bones to rotate.
    root = scene.find("Sketchfab_model")
    hip = _node("hip", translation=vec3(0.0, 0.4, 0.0))
    ul_l = _node("upper_leg_L_chain", translation=vec3(0.05, 0.0, 0.0))
    ll_l = _node("lower_leg_L_chain", translation=vec3(0.0, -0.2, 0.0))
    ft_l = _node("foot_L_chain", translation=vec3(0.0, -0.2, 0.0))
    ul_r = _node("upper_leg_R_chain", translation=vec3(-0.05, 0.0, 0.0))
    ll_r = _node("lower_leg_R_chain", translation=vec3(0.0, -0.2, 0.0))
    ft_r = _node("foot_R_chain", translation=vec3(0.0, -0.2, 0.0))
    ll_l.add_child(ft_l)
    ul_l.add_child(ll_l)
    hip.add_child(ul_l)
    ll_r.add_child(ft_r)
    ul_r.add_child(ll_r)
    hip.add_child(ul_r)
    root.add_child(hip)
    doc = _minimal_doc()
    doc["loop_sec"] = 5.0
    doc["phases"][0]["duration_sec"] = 5.0
    doc["rig"]["leg_chain_l"] = ["upper_leg_L_chain", "lower_leg_L_chain", "foot_L_chain"]
    doc["rig"]["leg_chain_r"] = ["upper_leg_R_chain", "lower_leg_R_chain", "foot_R_chain"]
    doc["phases"][0]["gait"] = {
        "kind": "stride", "step_count": 5,
        "leading_lift_rad": -0.7, "trailing_back_rad": 0.25,
        "knee_bend_rad": 0.3, "arm_swing_amplitude_rad": 0.4,
        "knee_bell": [0.10, 0.65], "forward_bell": [0.10, 0.65],
        "lock_trailing_foot": True,
    }
    parsed = parse_animation(doc)
    t_now = [0.05]  # step_idx=0, leading_l=True → R should be locked
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    assert "R" in runtime._foot_lock
    assert "L" not in runtime._foot_lock
    # Cross into step 1 → leading_l=False → L should be locked, R cleared.
    t_now[0] = 1.05
    hooks["update"](0.0)
    assert "L" in runtime._foot_lock
    assert "R" not in runtime._foot_lock


def test_stride_lock_disabled_when_flag_false() -> None:
    """``lock_trailing_foot: false`` skips the snapshot — useful when the
    user wants the trailing foot to follow the body (e.g. running)."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["loop_sec"] = 5.0
    doc["phases"][0]["duration_sec"] = 5.0
    doc["phases"][0]["gait"] = {
        "kind": "stride", "step_count": 5,
        "leading_lift_rad": -0.7, "trailing_back_rad": 0.25,
        "knee_bend_rad": 0.3, "arm_swing_amplitude_rad": 0.4,
        "lock_trailing_foot": False,
    }
    parsed = parse_animation(doc)
    t_now = [0.05]
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    assert runtime._foot_lock == {}


def test_runtime_drives_per_phase_morph_weights() -> None:
    """Phases declare ``morphs: {name: curve}``; the runtime resolves
    each curve per frame and writes the weight through ``api.morphs``.
    This is the JSON path equivalent of a script doing
    ``morphs.set("smile", value)`` each tick."""
    from posecascade.scripting.morph_api import MorphApi  # noqa: PLC0415
    scene = _build_minimal_scene()
    morph_api = MorphApi()
    doc = _minimal_doc()
    doc["loop_sec"] = 1.0
    doc["phases"][0]["duration_sec"] = 1.0
    doc["phases"][0]["morphs"] = {
        "smile": {"kind": "linear", "from": 0.0, "to": 1.0},
        "blink": "0.5 * sin(elapsed * tau)",
    }
    parsed = parse_animation(doc)
    t_now = [0.5]  # phase_t = 0.5 → smile = 0.5; elapsed=0.5 → sin(π) ≈ 0 → blink ≈ 0
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
        morph_api=morph_api,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    weights = dict(morph_api.current_weights())
    assert weights["smile"] == pytest.approx(0.5, abs=1e-3)
    assert weights["blink"] == pytest.approx(0.0, abs=1e-3)


def test_phase_without_morphs_does_not_write_weights() -> None:
    """Phases that don't declare morphs leave the weight map alone —
    so a previous phase's last weight stays until a later phase
    overwrites it (or the script clears explicitly)."""
    from posecascade.scripting.morph_api import MorphApi  # noqa: PLC0415
    scene = _build_minimal_scene()
    morph_api = MorphApi()
    morph_api.set("preexisting", 0.42)
    doc = _minimal_doc()  # no morphs
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        morph_api=morph_api,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    assert dict(morph_api.current_weights()) == {"preexisting": 0.42}


def test_value_curve_accepts_inline_expression_string() -> None:
    """A scalar string with arithmetic (e.g. ``"0.5 * sin(elapsed * tau)"``)
    is evaluated through the safe AST DSL when resolved per-frame."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["loop_sec"] = 1.0
    doc["phases"][0]["duration_sec"] = 1.0
    doc["phases"][0]["body"] = {
        # x position oscillates with elapsed time.
        "translation": {"x": "0.1 * sin(elapsed * tau)", "y": 0.0, "z": 0.0},
    }
    parsed = parse_animation(doc)
    t_now = [0.0]
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    # At t=0.25 (quarter of cycle), sin(π/2)=1 → x = 0.1.
    t_now[0] = 0.25
    hooks["update"](0.0)
    root = scene.find("Sketchfab_model")
    np.testing.assert_allclose(root.transform.translation[0], 0.1, atol=1e-3)


def test_value_curve_kind_expression_is_supported() -> None:
    """Explicit ``{kind: expression, source: ...}`` form, for cases where
    the author wants to be unambiguous."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["loop_sec"] = 1.0
    doc["phases"][0]["duration_sec"] = 1.0
    doc["phases"][0]["body"] = {
        "yaw_rad": {"kind": "expression", "source": "phase_t * pi"},
    }
    parsed = parse_animation(doc)
    t_now = [0.5]  # phase_t = 0.5
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    root = scene.find("Sketchfab_model")
    # yaw = 0.5 * π = π/2 → quaternion (0, sin(π/4), 0, cos(π/4))
    rot = root.transform.rotation
    np.testing.assert_allclose(rot[1], math.sin(math.pi / 4), atol=1e-3)


def test_physics_chains_parse_into_dict_of_floats() -> None:
    """``physics_chains`` parses param-by-param, converting symbolic
    floats and rejecting non-object entries."""
    doc = _minimal_doc()
    doc["physics_chains"] = {
        "hair_a": {"stiffness": 14.0, "damping": 0.45},
        "hair_b": {"stiffness": "tau", "damping": 0.30},
    }
    parsed = parse_animation(doc)
    assert parsed.physics_chains["hair_a"] == {"stiffness": 14.0, "damping": 0.45}
    assert parsed.physics_chains["hair_b"]["stiffness"] == pytest.approx(2 * 3.14159265, abs=1e-3)
