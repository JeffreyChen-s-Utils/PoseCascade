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
    with pytest.raises(DeclarativeAnimationError, match="at least one 'phases'"):
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


def test_walking_gait_arm_hang_produces_mirrored_z_rotation() -> None:
    """``arm_hang_rad`` composes a Z-axis tuck on each arm that mirrors
    across L / R — the rotation that flips a T-pose rest from horizontal
    ±X to a vertical hang at -Y. With swing=0 only the hang is visible
    in the bone rotation, and the L / R Z-quaternion components should
    have opposite signs of equal magnitude."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["loop_sec"] = 1.0
    doc["phases"][0]["duration_sec"] = 1.0
    doc["phases"][0]["gait"] = {
        "kind": "walking", "step_cycle_sec": 1.0,
        "leg_swing_amplitude": 0.0, "knee_bend": 0.0,
        "arm_swing_amplitude": 0.0, "arm_hang_rad": -1.45,
    }
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    arm_l = scene.find("upper_arm_L").transform.rotation
    arm_r = scene.find("upper_arm_R").transform.rotation
    np.testing.assert_allclose(arm_l[2], -arm_r[2], atol=1e-3)
    assert abs(arm_l[2]) > 0.5, (
        f"arm_l Z too small for arm_hang=-1.45: arm_l={arm_l}"
    )


def test_walking_gait_arms_swing_cross_body() -> None:
    """At peak swing the L and R arms have OPPOSITE-sign body-frame X
    swings — cross-body coordination falls out of the L / R amplitude
    flip in ``_set_arm``. Without the side flip, both arms would swing
    in unison (the T-pose-arms-flailing bug from the previous runtime)."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["loop_sec"] = 1.0
    doc["phases"][0]["duration_sec"] = 1.0
    doc["phases"][0]["gait"] = {
        "kind": "walking", "step_cycle_sec": 1.0,
        "leg_swing_amplitude": 0.0, "knee_bend": 0.0,
        "arm_swing_amplitude": 0.4, "arm_hang_rad": 0.0,
    }
    parsed = parse_animation(doc)
    t_now = [0.25]  # sin(π/2) = 1, peak swing
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    arm_l = scene.find("upper_arm_L").transform.rotation
    arm_r = scene.find("upper_arm_R").transform.rotation
    assert arm_l[0] * arm_r[0] < 0, (
        f"arms swing in unison instead of cross-body: arm_L={arm_l}, arm_R={arm_r}"
    )


def test_yaw_conjugation_isolates_bone_to_body_frame() -> None:
    """A body-frame X delta should produce the SAME bone-local rotation
    whether the body yaw is 0 or π. This is the core invariance behind
    yaw_to_world + parent-local conjugation: authors write a single
    amplitude that means "body-forward" at any orientation, and the
    runtime adjusts the world-frame composition so the visible motion
    matches that intent."""
    def _leg_l_local_rotation(yaw_value: object) -> np.ndarray:
        scene = _build_minimal_scene()
        doc = _minimal_doc()
        doc["loop_sec"] = 1.0
        doc["phases"][0]["duration_sec"] = 1.0
        doc["phases"][0]["body"]["yaw_rad"] = yaw_value
        doc["phases"][0]["gait"] = {
            "kind": "walking", "step_cycle_sec": 1.0,
            "leg_swing_amplitude": 0.5, "knee_bend": 0.0,
            "arm_swing_amplitude": 0.0, "arm_hang_rad": 0.0,
        }
        parsed = parse_animation(doc)
        t_now = [0.25]  # peak sin
        runtime = DeclarativeRuntime(
            animation=parsed, scene=scene, time=lambda: t_now[0],
        )
        hooks = runtime.hooks()
        hooks["start"]()
        hooks["update"](0.0)
        return scene.find("upper_leg_L").transform.rotation.copy()
    rot_zero = _leg_l_local_rotation(0.0)
    rot_pi = _leg_l_local_rotation("pi")
    np.testing.assert_allclose(rot_zero, rot_pi, atol=1e-3)


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


def test_lyrics_active_line_pushed_to_overlay() -> None:
    """At a moment inside a lyric line's window, that line's text is
    sent through the overlay callback."""
    scene = _build_minimal_scene()
    captured: list[str] = []
    doc = _minimal_doc()
    doc["loop_sec"] = 4.0
    doc["phases"][0]["duration_sec"] = 4.0
    doc["lyrics"] = [
        {"at_sec": 0.0, "text": "first line", "duration_sec": 1.0},
        {"at_sec": 2.0, "text": "second line", "duration_sec": 1.0},
    ]
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 2.5,
        overlay_api=captured.append,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    assert captured == ["second line"]


def test_lyrics_clears_overlay_between_lines() -> None:
    """Outside any lyric window, the overlay receives an empty string
    so previous text doesn't linger."""
    scene = _build_minimal_scene()
    captured: list[str] = []
    doc = _minimal_doc()
    doc["loop_sec"] = 4.0
    doc["phases"][0]["duration_sec"] = 4.0
    doc["lyrics"] = [
        {"at_sec": 0.0, "text": "hi", "duration_sec": 0.5},
    ]
    parsed = parse_animation(doc)
    t_now = [0.2]
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
        overlay_api=captured.append,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)  # active "hi"
    t_now[0] = 1.0  # past the line's end_sec (0.5)
    hooks["update"](0.0)
    assert captured == ["hi", ""]


def test_lyrics_only_pushes_on_change_not_every_frame() -> None:
    """Repeated updates inside the same active line don't re-push
    the same string — the overlay only sees transitions."""
    scene = _build_minimal_scene()
    captured: list[str] = []
    doc = _minimal_doc()
    doc["loop_sec"] = 4.0
    doc["phases"][0]["duration_sec"] = 4.0
    doc["lyrics"] = [
        {"at_sec": 0.0, "text": "x", "duration_sec": 2.0},
    ]
    parsed = parse_animation(doc)
    t_now = [0.1]
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
        overlay_api=captured.append,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    for sample in (0.1, 0.5, 1.0, 1.5):
        t_now[0] = sample
        hooks["update"](0.0)
    assert captured == ["x"]


def test_lyrics_at_beat_resolves_via_bpm() -> None:
    """``at_beat`` and ``duration_beats`` both resolve via the
    document-level bpm into seconds at parse time."""
    scene = _build_minimal_scene()
    captured: list[str] = []
    doc = _minimal_doc()
    doc["bpm"] = 120.0
    doc["loop_sec"] = 4.0
    doc["phases"][0]["duration_sec"] = 4.0
    doc["lyrics"] = [
        # 2 beats @ 120 bpm = 1.0 sec start; 2 beats = 1.0 sec duration
        {"at_beat": 2, "text": "beat-locked", "duration_beats": 2},
    ]
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 1.5,
        overlay_api=captured.append,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    assert captured == ["beat-locked"]


class _StubAudioPlayer:
    """In-memory AudioPlayer-shaped stub for declarative audio tests.

    Captures play / pause calls and returns a configurable
    ``current_time_seconds`` so sync_clock tests can pin the audio
    clock without spinning up QtMultimedia.
    """

    def __init__(self, *, clip: object) -> None:  # noqa: ARG002
        self.played = False
        self.attached = False
        self._t = 0.0

    def attach_qt(self) -> bool:
        self.attached = True
        return False  # simulate headless mode

    def play(self) -> None:
        self.played = True

    def current_time_seconds(self) -> float:
        return self._t


def test_audio_loads_and_plays_when_spec_present(tmp_path) -> None:
    """When the document declares an audio block AND the WAV exists,
    the runtime instantiates an AudioPlayer (via the factory hook)
    and calls play() at start."""
    wav_path = tmp_path / "song.wav"
    _write_minimal_wav(wav_path)
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["audio"] = {"path": "song.wav"}
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        source_dir=tmp_path,
        audio_player_factory=_StubAudioPlayer,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    player = runtime._audio_player
    assert isinstance(player, _StubAudioPlayer)
    assert player.played
    assert player.attached


def test_audio_sync_clock_replaces_time_provider(tmp_path) -> None:
    """With sync_clock: true, runtime.time() returns
    audio.current_time_seconds() - offset_sec."""
    wav_path = tmp_path / "song.wav"
    _write_minimal_wav(wav_path)
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["audio"] = {"path": "song.wav", "offset_sec": 0.5, "sync_clock": True}
    parsed = parse_animation(doc)

    def _player_factory(*, clip: object) -> _StubAudioPlayer:
        p = _StubAudioPlayer(clip=clip)
        p._t = 1.5  # noqa: SLF001 — test seam
        return p

    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 99.0,
        source_dir=tmp_path,
        audio_player_factory=_player_factory,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    # After start, runtime.time should read from the audio clock (1.5)
    # minus the offset (0.5) → 1.0, NOT the wall-clock 99.0.
    assert runtime.time() == pytest.approx(1.0, abs=1e-3)


def test_audio_missing_file_logs_and_continues(tmp_path) -> None:
    """A bogus audio path doesn't crash the animation — logs a warning
    and falls back to no audio."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["audio"] = {"path": "nonexistent.wav"}
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        source_dir=tmp_path,
        audio_player_factory=_StubAudioPlayer,
    )
    hooks = runtime.hooks()
    hooks["start"]()  # must not raise
    assert runtime._audio_player is None


def _write_minimal_wav(path) -> None:
    """Write a 0.1-second mono 16-bit silence WAV — enough for
    AudioClip.load_wav_file to parse without hitting actual audio."""
    import wave  # noqa: PLC0415

    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(b"\x00\x00" * 4410)


def _build_finger_scene() -> Scene:
    """Minimal scene with body bones plus a few VRoid-style finger bones
    so hand-preset tests have something to write to."""
    scene = _build_minimal_scene()
    root = scene.find("Sketchfab_model")
    for finger in (
        "J_Bip_L_Index1", "J_Bip_L_Index2", "J_Bip_L_Index3",
        "J_Bip_L_Middle1", "J_Bip_L_Middle2", "J_Bip_L_Middle3",
        "J_Bip_L_Ring1", "J_Bip_L_Ring2", "J_Bip_L_Ring3",
        "J_Bip_L_Little1", "J_Bip_L_Little2", "J_Bip_L_Little3",
        "J_Bip_L_Thumb1", "J_Bip_L_Thumb2", "J_Bip_L_Thumb3",
        "J_Bip_R_Index1", "J_Bip_R_Index2", "J_Bip_R_Index3",
        "J_Bip_R_Middle1", "J_Bip_R_Middle2", "J_Bip_R_Middle3",
        "J_Bip_R_Ring1", "J_Bip_R_Ring2", "J_Bip_R_Ring3",
        "J_Bip_R_Little1", "J_Bip_R_Little2", "J_Bip_R_Little3",
        "J_Bip_R_Thumb1", "J_Bip_R_Thumb2", "J_Bip_R_Thumb3",
    ):
        root.add_child(_node(finger))
    return scene


def test_hand_preset_peace_curls_ring_and_little_fingers() -> None:
    """``hand_L: 'peace_L'`` writes the 'peace' preset's finger curls
    onto the L-hand bones."""
    scene = _build_finger_scene()
    doc = _minimal_doc()
    doc["phases"][0]["hand_L"] = "peace_L"
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    # Ring + Little curl ~1.4 → quat[0] = sin(0.7) > 0.5
    assert scene.find("J_Bip_L_Ring1").transform.rotation[0] > 0.5
    assert scene.find("J_Bip_L_Little1").transform.rotation[0] > 0.5
    # Index + Middle stay extended → identity rotation
    np.testing.assert_allclose(
        scene.find("J_Bip_L_Index1").transform.rotation[0], 0.0, atol=1e-3,
    )


def test_hand_user_library_overrides_builtin() -> None:
    """A user-declared hand_library entry with the same name as a
    built-in replaces it."""
    scene = _build_finger_scene()
    doc = _minimal_doc()
    doc["hand_library"] = {
        "peace_L": {"J_Bip_L_Index1": {"x_rad": 1.0}},
    }
    doc["phases"][0]["hand_L"] = "peace_L"
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    np.testing.assert_allclose(
        scene.find("J_Bip_L_Index1").transform.rotation[0],
        math.sin(0.5),
        atol=1e-3,
    )
    # Other finger bones not in the user's preset are absent → at rest.
    np.testing.assert_allclose(
        scene.find("J_Bip_L_Ring1").transform.rotation[0], 0.0, atol=1e-3,
    )


def test_hand_preset_unknown_name_logged_and_skipped() -> None:
    """Unknown hand preset name doesn't raise."""
    scene = _build_finger_scene()
    doc = _minimal_doc()
    doc["phases"][0]["hand_L"] = "no_such_hand"
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)  # must not raise


def test_hand_and_body_pose_compose_with_phase_bones_override() -> None:
    """Body pose + hand preset + phase.bones layer correctly: hand
    preset's finger writes survive, body pose's bones survive, and
    phase.bones overrides any axis it explicitly sets."""
    scene = _build_finger_scene()
    doc = _minimal_doc()
    doc["phases"][0]["pose"] = "v_arms_up"  # writes head, chest, upper_arm_*
    doc["phases"][0]["hand_R"] = "fist_R"   # writes R-hand finger curls
    doc["phases"][0]["bones"] = {
        "head": {"x_rad": 0.0},  # override pose's head x_rad
    }
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    # Head's x_rad overridden to 0 → quat[0] ≈ 0
    np.testing.assert_allclose(
        scene.find("head").transform.rotation[0], 0.0, atol=1e-3,
    )
    # R-hand fist curl on Index1 still applied
    assert scene.find("J_Bip_R_Index1").transform.rotation[0] > 0.5


class _StubCamera:
    """Minimal Camera-shaped stub for declarative camera tests.

    Captures position / target / fov_degrees writes so assertions can
    verify what the runtime wrote without needing a real Camera class
    from posecascade.render which pulls in numpy + Vec3 plumbing.
    """

    def __init__(self) -> None:
        self.position = None
        self.target = None
        self.fov_degrees = 60.0


def test_camera_keyframes_lerp_position_target_at_midpoint() -> None:
    """Two camera keys at t=0 and t=2; at t=1 the runtime writes the
    midpoint of position / target / fov."""
    scene = _build_minimal_scene()
    camera = _StubCamera()
    doc = _minimal_doc()
    doc["loop_sec"] = 4.0
    doc["phases"][0]["duration_sec"] = 4.0
    doc["camera"] = [
        {"at_sec": 0.0, "position": [0, 0, 0], "target": [0, 0, 0], "fov": 50.0},
        {"at_sec": 2.0, "position": [10, 5, -2], "target": [1, 1, 1], "fov": 70.0},
    ]
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 1.0,
        camera_api=camera,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    np.testing.assert_allclose(camera.position, [5.0, 2.5, -1.0], atol=1e-3)
    np.testing.assert_allclose(camera.target, [0.5, 0.5, 0.5], atol=1e-3)
    assert camera.fov_degrees == pytest.approx(60.0, abs=1e-3)


def test_camera_holds_at_first_key_before_window() -> None:
    """Times before the first keyframe snap to its values — same idea as
    "establishing shot held for N seconds before the camera starts moving"."""
    scene = _build_minimal_scene()
    camera = _StubCamera()
    doc = _minimal_doc()
    doc["loop_sec"] = 4.0
    doc["phases"][0]["duration_sec"] = 4.0
    doc["camera"] = [
        {"at_sec": 1.0, "position": [3, 0, 0], "target": [0, 0, 0]},
        {"at_sec": 2.0, "position": [6, 0, 0], "target": [0, 0, 0]},
    ]
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        camera_api=camera,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    np.testing.assert_allclose(camera.position, [3, 0, 0], atol=1e-3)


def test_camera_at_beat_resolves_via_bpm() -> None:
    """Keyframes can be authored in beats when the document has a bpm."""
    scene = _build_minimal_scene()
    camera = _StubCamera()
    doc = _minimal_doc()
    doc["bpm"] = 120.0
    doc["loop_sec"] = 4.0
    doc["phases"][0]["duration_sec"] = 4.0
    # 4 beats @ 120 bpm = 2.0 sec → keyframe at t = 2.0
    doc["camera"] = [
        {"at_beat": 0, "position": [0, 0, 0], "target": [0, 0, 0]},
        {"at_beat": 4, "position": [4, 0, 0], "target": [0, 0, 0]},
    ]
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 1.0,
        camera_api=camera,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    np.testing.assert_allclose(camera.position, [2.0, 0.0, 0.0], atol=1e-3)


def test_camera_untouched_when_no_keyframes() -> None:
    """Documents with no camera array leave the Camera object alone."""
    scene = _build_minimal_scene()
    camera = _StubCamera()
    camera.position = "untouched_sentinel"  # type: ignore[assignment]
    doc = _minimal_doc()  # no camera field
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        camera_api=camera,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    assert camera.position == "untouched_sentinel"


def test_pose_preset_applies_builtin_bones() -> None:
    """Setting ``pose: 'v_arms_up'`` writes the built-in preset's bone
    rotations through the runtime — verified by sampling the head's
    x_rad against the preset table."""
    from posecascade.scripting.pose_library import BUILTIN_POSES  # noqa: PLC0415
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["phases"][0]["pose"] = "v_arms_up"
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    head_rot = scene.find("head").transform.rotation
    expected_x = BUILTIN_POSES["v_arms_up"]["head"]["x_rad"]
    # X rotation by expected_x → quat[0] = sin(expected_x/2)
    np.testing.assert_allclose(
        head_rot[0], math.sin(expected_x / 2.0), atol=1e-3,
    )


def test_pose_user_library_overrides_builtin() -> None:
    """A document-level ``pose_library`` entry with the same name as
    a built-in preset replaces it."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["pose_library"] = {
        "v_arms_up": {"head": {"x_rad": 0.5}},
    }
    doc["phases"][0]["pose"] = "v_arms_up"
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    head_rot = scene.find("head").transform.rotation
    np.testing.assert_allclose(head_rot[0], math.sin(0.25), atol=1e-3)


def test_pose_weight_scales_preset_rotations() -> None:
    """Object form ``pose: {name, weight: 0.5}`` scales the preset's
    rotations by 0.5 each frame — useful for easing in/out of a pose."""
    from posecascade.scripting.pose_library import BUILTIN_POSES  # noqa: PLC0415
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["phases"][0]["pose"] = {"name": "v_arms_up", "weight": 0.5}
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    head_rot = scene.find("head").transform.rotation
    half_x = BUILTIN_POSES["v_arms_up"]["head"]["x_rad"] * 0.5
    np.testing.assert_allclose(head_rot[0], math.sin(half_x / 2.0), atol=1e-3)


def test_pose_phase_bones_override_preset_per_axis() -> None:
    """``phase.bones[bone][axis]`` wins over the preset's same axis;
    other axes from the preset survive untouched."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["phases"][0]["pose"] = "v_arms_up"
    # v_arms_up's head has x_rad: -0.18. Override only x_rad to 0.0.
    doc["phases"][0]["bones"] = {"head": {"x_rad": 0.0}}
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    head_rot = scene.find("head").transform.rotation
    # x_rad now 0 → quat[0] ≈ 0.
    np.testing.assert_allclose(head_rot[0], 0.0, atol=1e-3)


def test_new_pose_presets_register_in_builtin_library() -> None:
    """The arm-clipping fix added three presets. Pin their names so a
    rename doesn't silently break example scripts that reference them."""
    from posecascade.scripting.pose_library import BUILTIN_POSES  # noqa: PLC0415
    for name in ("reach_R_soft", "reach_L_soft", "arms_open"):
        assert name in BUILTIN_POSES, f"missing built-in preset {name!r}"
    # reach_R_soft / reach_L_soft are mirror — same magnitude, opposite
    # z_rad sign on the lifted arm so neither hand clips into the body.
    r = BUILTIN_POSES["reach_R_soft"]
    l_ = BUILTIN_POSES["reach_L_soft"]
    assert r["upper_arm_R"]["x_rad"] == pytest.approx(l_["upper_arm_L"]["x_rad"])
    assert r["upper_arm_R"]["z_rad"] == pytest.approx(-l_["upper_arm_L"]["z_rad"])


class _StubClothHost:
    """In-memory ClothHost-shaped stub for the declarative cloth tests.

    Captures add_cloth_for_node + add_collider calls so assertions can
    verify what the runtime asked for without spinning up the real PBD
    solver. Mirrors the surface :class:`DeclarativeRuntime` actually
    consumes — nothing else.
    """

    def __init__(self) -> None:
        self.cloth_calls: list[dict] = []
        self.colliders: list = []
        # Mirrors the real ClothHost surface — the declarative runtime
        # writes ``floor_y`` from the animation's ``ground`` block so the
        # cloth solver clamps verts to the same plane the foot planter
        # uses for IK. ``None`` is the no-clamp default.
        self.floor_y: float | None = None

    def add_cloth_for_node(self, node, **kwargs) -> None:  # noqa: ANN001
        self.cloth_calls.append({"node": node.name, **kwargs})

    def add_collider(self, collider) -> None:  # noqa: ANN001
        self.colliders.append(collider)


def test_hand_ik_drives_wrist_toward_target() -> None:
    """Declaring ``rig.arm_chain_l`` + ``ik.hand_l_target`` should make
    the runtime drive the wrist node to the world target each frame.

    Builds a vertical 3-bone arm chain (shoulder at Y=2 → elbow at
    Y=1.5 → wrist at Y=1), declares a target at (1, 1.2, 0), and
    asserts the wrist ends up within a small tolerance of the target
    after one update tick. Loose tolerance — CCD converges in 8
    iterations to within centimetres, not microns, and a future
    swap to the analytic solver would still satisfy 0.05 m."""
    from posecascade.utils.math3d import vec3  # noqa: PLC0415

    scene = Scene(name="test")
    char_root = _node("char")
    # Hierarchical arm chain: shoulder owns elbow owns wrist. The
    # IK solver walks the parent chain to find world-space orientation,
    # so the bones need to be nested, not flat siblings.
    shoulder = _node("upper_arm_L")
    shoulder.transform.set_translation(vec3(0.0, 2.0, 0.0))
    elbow = _node("lower_arm_L")
    elbow.transform.set_translation(vec3(0.0, -0.5, 0.0))
    wrist = _node("hand_L")
    wrist.transform.set_translation(vec3(0.0, -0.5, 0.0))
    elbow.add_child(wrist)
    shoulder.add_child(elbow)
    char_root.add_child(shoulder)
    scene.root.add_child(char_root)

    doc = _minimal_doc()
    doc["rig"]["character_root"] = "char"
    doc["rig"]["arm_chain_l"] = ["upper_arm_L", "lower_arm_L", "hand_L"]
    # Target within reach (~0.6 m from shoulder; arm length is 1.0 m).
    doc["phases"][0]["ik"] = {"hand_l_target": [0.4, 1.6, 0.0]}
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)

    from posecascade.animation.ik import _world_position  # noqa: PLC0415
    wrist_world = np.asarray(_world_position(wrist), dtype=np.float32)
    target = np.array([0.4, 1.6, 0.0], dtype=np.float32)
    distance = float(np.linalg.norm(wrist_world - target))
    # Within 10 cm is a comfortable bar for 8 iterations of CCD on a
    # 2-link chain. The point isn't sub-mm precision — it's "the IK
    # actually drove the bone toward the target", which 10 cm proves.
    assert distance < 0.10, f"wrist landed at {wrist_world}, target {target}, distance {distance}"


def test_hand_ik_disabled_without_arm_chain_or_target() -> None:
    """Hand IK should no-op cleanly when either the rig has no
    ``arm_chain_*`` OR the phase has no ``ik.hand_*_target``.

    Regression guard: a typo in the rig block shouldn't crash the
    runtime — it should just leave the arms at rest (matching
    pre-IK behaviour) so existing animations keep working when
    only one side of the wiring is present."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    # Arm chain declared but no target — should no-op.
    doc["rig"]["arm_chain_l"] = ["upper_arm_L", "lower_arm_L", "hand_L"]
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(animation=parsed, scene=scene, time=lambda: 0.0)
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)  # must not raise


def test_auto_clamp_discovers_skinned_meshes_under_flat_ground() -> None:
    """With ``ground.kind == 'flat'`` and the default
    ``auto_clamp_skinned_to_ground = True``, the runtime walks the scene
    at start and registers every SkinRefComponent node as a passive
    collision_deform piece — so the engine's cloth-floor clamp covers
    every clothing / hair / accessory mesh without per-document
    enumeration.

    The minimal test scene has three skinned nodes (chest / upper_arm_L /
    upper_leg_L per ``_build_minimal_scene``); the test asserts every
    one of them ends up as a cloth piece named ``auto_clamp_<node>``."""
    from posecascade.scene.component import SkinRefComponent  # noqa: PLC0415
    scene = _build_minimal_scene()
    # Tag the three skinned nodes with SkinRefComponent so the discovery
    # walk sees them. The minimal scene's bones don't carry a skin ref
    # by default — this stamps the marker the cloth host looks for.
    for name in ("chest", "upper_arm_L", "upper_leg_L"):
        node = scene.find(name)
        node.components = list(node.components) + [SkinRefComponent()]
    cloth_host = _StubClothHost()
    doc = _minimal_doc()
    doc["ground"] = {"kind": "flat", "y": 0.0}
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        cloth_host=cloth_host,
    )
    runtime.hooks()["start"]()
    auto_names = {c["cloth_name"] for c in cloth_host.cloth_calls}
    assert "auto_clamp_chest" in auto_names
    assert "auto_clamp_upper_arm_L" in auto_names
    assert "auto_clamp_upper_leg_L" in auto_names


def test_auto_clamp_skips_explicit_collision_deform_entries() -> None:
    """Names already in ``collision_deform_meshes`` aren't re-discovered.

    Otherwise the auto path would register a duplicate piece for every
    explicit entry — the cloth host would happily build two solver
    pieces for the same mesh and the renderer would draw whichever's
    state was synced last. Pin the dedupe behaviour so a future
    refactor can't silently regress it."""
    from posecascade.scene.component import SkinRefComponent  # noqa: PLC0415
    scene = _build_minimal_scene()
    chest = scene.find("chest")
    chest.components = list(chest.components) + [SkinRefComponent()]
    cloth_host = _StubClothHost()
    doc = _minimal_doc()
    doc["ground"] = {"kind": "flat", "y": 0.0}
    doc["collision_deform_meshes"] = ["chest"]  # explicit entry
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        cloth_host=cloth_host,
    )
    runtime.hooks()["start"]()
    cloth_names = [c["cloth_name"] for c in cloth_host.cloth_calls]
    # Exactly one registration — the explicit one — for chest.
    assert cloth_names.count("collision_deform_chest") == 1
    assert "auto_clamp_chest" not in cloth_names


def test_auto_clamp_opt_out_via_flag() -> None:
    """Setting ``auto_clamp_skinned_to_ground = false`` disables discovery.

    Useful for abstract / underground sequences where the cloth should
    be free to drape below the floor plane (or for non-humanoid rigs
    where the auto-discovered passive-skin overhead isn't worth it)."""
    from posecascade.scene.component import SkinRefComponent  # noqa: PLC0415
    scene = _build_minimal_scene()
    scene.find("chest").components = list(scene.find("chest").components) + [
        SkinRefComponent(),
    ]
    cloth_host = _StubClothHost()
    doc = _minimal_doc()
    doc["ground"] = {"kind": "flat", "y": 0.0}
    doc["auto_clamp_skinned_to_ground"] = False
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        cloth_host=cloth_host,
    )
    runtime.hooks()["start"]()
    # No cloth pieces registered — the opt-out switched off the whole
    # auto path even though ground.kind == flat.
    assert not any(
        c["cloth_name"].startswith("auto_clamp_") for c in cloth_host.cloth_calls
    )


def test_flat_ground_sets_cloth_host_floor_y() -> None:
    """Declaring ``ground: {kind: flat, y: N}`` should propagate N into
    ``cloth_host.floor_y`` so cloth verts stop at the same plane the
    foot planter uses for IK.

    Regression guard: before the cloth solver gained ``ground_y``, the
    skirt clipped through the floor in any pose that lowered the hips
    (squat, kneel, crawl). The fix is engine-wide — every animation
    with a flat ground gets the clamp for free; this test pins that
    contract so a future refactor can't silently disable it."""
    scene = _build_minimal_scene()
    cloth_host = _StubClothHost()
    doc = _minimal_doc()
    doc["ground"] = {"kind": "flat", "y": -0.05}
    # A cloth piece is required because the wiring lives in
    # ``_setup_cloth_and_colliders`` — without one, that hook short-
    # circuits before reaching the ground-clamp wiring.
    doc["cloth"] = [{"mesh_node": "chest"}]
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        cloth_host=cloth_host,
    )
    runtime.hooks()["start"]()
    assert cloth_host.floor_y == pytest.approx(-0.05)


def test_stairs_ground_leaves_cloth_floor_unset() -> None:
    """Only ``ground.kind == 'flat'`` maps to ``cloth_host.floor_y``.

    Stairs intentionally don't, because the cloth has no notion of
    which step a vert should rest on without per-step colliders, and
    clamping to a single plane would either let cloth sink through
    the upper step or hover above the lower one. Better to leave the
    clamp off so each scene's authoring decides what to do."""
    scene = _build_minimal_scene()
    cloth_host = _StubClothHost()
    doc = _minimal_doc()
    doc["ground"] = {
        "kind": "stairs", "base_z": 0.0, "step_depth": 0.3, "step_rise": 0.15, "count": 4,
    }
    doc["cloth"] = [{"mesh_node": "chest"}]
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        cloth_host=cloth_host,
    )
    runtime.hooks()["start"]()
    assert cloth_host.floor_y is None


def test_cloth_pieces_register_with_cloth_host_on_start() -> None:
    """Each entry in document-root 'cloth' calls
    ``cloth_host.add_cloth_for_node`` with the named scene node + the
    spec's parameters."""
    scene = _build_minimal_scene()
    cloth_host = _StubClothHost()
    doc = _minimal_doc()
    doc["cloth"] = [
        {
            "mesh_node": "chest",  # arbitrary existing node in minimal scene
            "structural_stiffness": 0.7,
            "bend_stiffness": 0.2,
            "anchor_fraction": 0.20,
        },
    ]
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        cloth_host=cloth_host,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    assert len(cloth_host.cloth_calls) == 1
    call = cloth_host.cloth_calls[0]
    assert call["node"] == "chest"
    assert call["structural_stiffness"] == pytest.approx(0.7)
    assert call["bend_stiffness"] == pytest.approx(0.2)
    assert call["anchor_fraction"] == pytest.approx(0.20)


def test_colliders_register_and_track_bones_per_frame() -> None:
    """Sphere + capsule colliders are registered with the cloth host
    at start, and their geometry is mutated each frame to follow the
    named bones' world positions."""
    scene = _build_minimal_scene()
    cloth_host = _StubClothHost()
    doc = _minimal_doc()
    doc["colliders"] = [
        {"kind": "sphere", "follow_bone": "head", "radius": 0.05},
        {
            "kind": "capsule",
            # Use two bones that DO exist in the minimal test scene —
            # the geometry doesn't matter for what this test checks
            # (collider registers + tracks both endpoints).
            "follow_bone": "upper_arm_L",
            "end_bone": "upper_leg_L",
            "radius": 0.04,
        },
    ]
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        cloth_host=cloth_host,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    assert len(cloth_host.colliders) == 2
    sphere, capsule = cloth_host.colliders
    # Sphere has center; capsule has a + b. Initial positions match
    # the bones' world positions captured at start.
    assert hasattr(sphere, "center")
    assert hasattr(capsule, "a") and hasattr(capsule, "b")
    initial_center = np.array(sphere.center, dtype=np.float32).copy()
    # Move the head bone — collider should follow on next update.
    head = scene.find("head")
    head.transform.set_translation(vec3(1.5, 0.0, 0.0))
    hooks["update"](0.0)
    moved_center = np.array(sphere.center, dtype=np.float32)
    assert not np.allclose(initial_center, moved_center, atol=1e-3), (
        f"sphere collider should track bone movement; was {initial_center}, "
        f"now {moved_center}"
    )
    np.testing.assert_allclose(moved_center[0], 1.5, atol=1e-3)


def test_cloth_rest_positions_track_bone_full_transform() -> None:
    """ALL cloth verts follow the track_bone's full transform each tick.

    Tracking only the anchors and letting free verts simulate around
    their init-world rest position visually TORE the cloth in half when
    the body yaw flipped between gait phases: anchors snapped 180 deg
    while free verts stayed at the original orientation, and structural
    edges spanned impossible distances. The fix tracks the bone's full
    transform (rotation + translation) and applies it to every vert in
    ``positions`` / ``prev_positions`` / ``rest_positions`` each tick,
    so the whole cloth moves as one rigid frame between simulator steps.
    """
    from posecascade.animation.cloth_host import ClothHost  # noqa: PLC0415
    from posecascade.assets.types import ImportedScene, Mesh  # noqa: PLC0415
    from posecascade.scene.component import (  # noqa: PLC0415
        ClothComponent,
        MeshRefComponent,
    )
    from posecascade.utils.math3d import quat_from_axis_angle  # noqa: PLC0415

    positions = np.array(
        [
            [-0.1,  0.1, 0.0], [0.1,  0.1, 0.0],
            [-0.1,  0.0, 0.0], [0.1,  0.0, 0.0],
            [-0.1, -0.1, 0.0], [0.1, -0.1, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0,2,1, 1,2,3, 2,4,3, 3,4,5], dtype=np.uint32)
    mesh = Mesh(name="m", positions=positions, indices=indices)

    root = _node("Sketchfab_model")
    bone = _node("hip")
    cloth_node = _node("cloth_node")
    root.add_child(bone)
    root.add_child(cloth_node)
    cloth_node.add_component(MeshRefComponent(mesh_indices=(0,)))
    cloth_node.add_component(
        ClothComponent(cloth_name="c", mesh_index=0, anchor_fraction=0.34),
    )
    scene = Scene(name="t")
    scene.root.add_child(root)
    imported = ImportedScene(meshes=(mesh,), textures=(), skins=(), scene=scene)

    doc = _minimal_doc()
    doc["cloth"] = [
        {"mesh_node": "cloth_node", "track_bone": "hip", "anchor_fraction": 0.34},
    ]
    parsed = parse_animation(doc)

    host = ClothHost()
    host.register_imported_scene(imported)

    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0, cloth_host=host,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    piece = host.find_piece("cloth_node")
    rest_before = piece.rest_positions.copy()
    anchor_idx = np.flatnonzero(piece.inverse_masses == 0.0)
    free_idx = np.flatnonzero(piece.inverse_masses > 0.0)
    assert anchor_idx.size > 0 and free_idx.size > 0

    bone.transform.set_rotation(quat_from_axis_angle(vec3(0.0, 1.0, 0.0), 0.5))
    # Anchor update now lives on the cloth host and runs at the start of each tick.
    host.tick(1.0 / 60.0)

    # Anchor band rest moved.
    assert not np.allclose(
        rest_before[anchor_idx], piece.rest_positions[anchor_idx], atol=1e-4,
    ), "anchor-vert rest should track the bone"
    # Free band rest ALSO moved — the bone's rotation propagates through
    # the whole cloth so structural edges between anchors and free verts
    # stay at their rest length when the body yaws.
    assert not np.allclose(
        rest_before[free_idx], piece.rest_positions[free_idx], atol=1e-4,
    ), "free-vert rest should track the bone's full transform too"


def test_collider_unknown_bone_logged_and_skipped() -> None:
    """A collider with a follow_bone that doesn't exist is logged +
    skipped so the rest of the dance still loads."""
    scene = _build_minimal_scene()
    cloth_host = _StubClothHost()
    doc = _minimal_doc()
    doc["colliders"] = [
        {"kind": "sphere", "follow_bone": "no_such_bone", "radius": 0.05},
    ]
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        cloth_host=cloth_host,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    assert cloth_host.colliders == []


def test_auto_body_colliders_emitted_when_cloth_present() -> None:
    """A document declaring cloth but no explicit colliders gets the
    standard humanoid body set auto-attached from ``scene.bone_aliases``.
    Engine-layer concern: per-character collider wiring shouldn't be
    re-written on every script."""
    scene = _build_minimal_scene()
    # Wire bone aliases: canonical names happen to MATCH the bone-node
    # names in this minimal scene, but the alias map is what the
    # auto-emitter consumes, so set it explicitly.
    for canonical in (
        "hip",
        "upper_leg_L", "lower_leg_L", "foot_L",
        "upper_leg_R", "lower_leg_R", "foot_R",
    ):
        scene.bone_aliases[canonical] = scene.find(canonical)
    cloth_host = _StubClothHost()
    doc = _minimal_doc()
    doc["cloth"] = [{"mesh_node": "chest"}]
    # No "colliders" key — auto-emit is the only source.
    parsed = parse_animation(doc)
    assert parsed.auto_body_colliders is True
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        cloth_host=cloth_host,
    )
    runtime.hooks()["start"]()
    # 1 hip sphere + 2 thigh + 2 shin capsules = 5
    assert len(cloth_host.colliders) == 5
    kinds = sorted(type(c).__name__ for c in cloth_host.colliders)
    assert kinds == [
        "CapsuleCollider", "CapsuleCollider", "CapsuleCollider",
        "CapsuleCollider", "SphereCollider",
    ]


def test_auto_body_colliders_disabled_by_flag() -> None:
    """Setting ``"auto_body_colliders": false`` on the document opts out
    even when cloth is declared. Lets non-humanoid rigs use cloth
    without inheriting the humanoid leg colliders."""
    scene = _build_minimal_scene()
    for canonical in ("hip", "upper_leg_L", "lower_leg_L"):
        scene.bone_aliases[canonical] = scene.find(canonical)
    cloth_host = _StubClothHost()
    doc = _minimal_doc()
    doc["cloth"] = [{"mesh_node": "chest"}]
    doc["auto_body_colliders"] = False
    parsed = parse_animation(doc)
    assert parsed.auto_body_colliders is False
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        cloth_host=cloth_host,
    )
    runtime.hooks()["start"]()
    assert cloth_host.colliders == []


def test_auto_body_colliders_merge_with_explicit() -> None:
    """Explicit colliders win on duplicate bone keys; auto-emit only
    fills gaps the JSON didn't already cover."""
    scene = _build_minimal_scene()
    for canonical in (
        "hip", "upper_leg_L", "lower_leg_L", "upper_leg_R", "lower_leg_R",
    ):
        scene.bone_aliases[canonical] = scene.find(canonical)
    cloth_host = _StubClothHost()
    doc = _minimal_doc()
    doc["cloth"] = [{"mesh_node": "chest"}]
    # Explicit hip sphere with a wider radius than the default 0.10.
    doc["colliders"] = [
        {"kind": "sphere", "follow_bone": "hip", "radius": 0.18},
    ]
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
        cloth_host=cloth_host,
    )
    runtime.hooks()["start"]()
    # Explicit hip (0.18) + 2 auto thigh capsules — auto hip is suppressed
    # because the explicit one already covers (kind, follow_bone, end_bone).
    spheres = [c for c in cloth_host.colliders if type(c).__name__ == "SphereCollider"]
    assert len(spheres) == 1
    assert spheres[0].radius == pytest.approx(0.18)
    capsules = [
        c for c in cloth_host.colliders if type(c).__name__ == "CapsuleCollider"
    ]
    assert len(capsules) == 2


def test_hide_detaches_named_nodes_from_scene() -> None:
    """Document-root 'hide' detaches each named node from its parent at
    start. Useful for asset bundles that ship props (Stairs, room,
    lights) the dance doesn't want."""
    scene = _build_minimal_scene()
    # Add a prop sibling to Sketchfab_model so we have something to hide.
    scene.root.add_child(_node("Stairs"))
    assert scene.find("Stairs") is not None
    doc = _minimal_doc()
    doc["hide"] = ["Stairs"]
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    assert scene.find("Stairs") is None
    # Sketchfab_model still present — only the named node is hidden.
    assert scene.find("Sketchfab_model") is not None


def test_hide_unknown_node_is_skipped_quietly() -> None:
    """An unknown name in 'hide' doesn't raise — logged + skipped."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["hide"] = ["NoSuchNode"]
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()  # must not raise


def test_pose_weight_slerps_from_gait_baseline_not_rest() -> None:
    """The new pose_blends pathway slerps from the bone's CURRENT
    rotation (gait baseline) toward the pose target by weight. So at
    weight 0.5 with gait writing arm_hang and pose writing arms_up,
    the arm should land halfway between hanging and overhead — NOT
    at half-arms-up-from-T-pose (the old magnitude-scale semantic)."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["phases"][0]["gait"] = {
        "kind": "walking", "step_cycle_sec": 1.0,
        "leg_swing_amplitude": 0.0, "knee_bend": 0.0,
        "arm_swing_amplitude": 0.0, "arm_hang_rad": -1.45,
    }
    doc["phases"][0]["pose"] = {"name": "v_arms_up", "weight": 0.5}
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    arm_l = scene.find("upper_arm_L").transform.rotation
    # At weight 0.5, the arm's rotation should still carry significant
    # Z component from gait's arm_hang (the old magnitude-scale
    # semantic would have lost most of the hang because it scaled the
    # pose's rotation FROM rest, ignoring gait). |z| > 0.2 means hang
    # is still meaningfully there alongside the pose's X contribution.
    assert abs(arm_l[2]) > 0.2, (
        f"arm_l should retain gait hang at weight=0.5; got {arm_l}"
    )
    # And the pose's X contribution is also clearly there (otherwise
    # we're just running pure gait).
    assert abs(arm_l[0]) > 0.2, (
        f"arm_l should also carry pose contribution at weight=0.5; got {arm_l}"
    )


def test_pose_zero_weight_does_not_override_gait() -> None:
    """A pose with weight=0 must NOT clobber the gait's bone writes back
    to identity — that would snap arms to the rig's T-pose silhouette,
    visibly breaking the dance for one frame at the start of any
    weight-from-zero ramp. Verify by running gait + a weight-0 pose
    and checking the upper_arm bone keeps its gait-written rotation."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["phases"][0]["gait"] = {
        "kind": "walking", "step_cycle_sec": 1.0,
        "leg_swing_amplitude": 0.0, "knee_bend": 0.0,
        "arm_swing_amplitude": 0.0, "arm_hang_rad": -1.45,
    }
    doc["phases"][0]["pose"] = {"name": "v_arms_up", "weight": 0.0}
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    arm_l = scene.find("upper_arm_L").transform.rotation
    # arm_hang_rad of -1.45 produces a non-trivial Z-axis rotation on
    # the arm (the "hang" component); arm should NOT be at identity.
    assert abs(arm_l[2]) > 0.5, (
        f"arm_l should keep gait's arm_hang rotation when pose weight=0; "
        f"got {arm_l} (T-pose suspect)"
    )


def test_crossfade_one_sided_bone_keeps_full_value() -> None:
    """A bone written by only ONE of the two crossfading phases must be
    emitted at full strength — slerping toward identity would flash the
    bone to its rest pose (T-pose for VRoid arms) through the blend
    window, the symptom this fix targets."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["loop_sec"] = 2.0
    doc["phases"] = [
        {
            "name": "a",
            "duration_sec": 1.0,
            "blend_out_sec": 0.4,
            # NO bones — this phase relies on gait alone.
            "gait": {
                "kind": "walking", "step_cycle_sec": 1.0,
                "leg_swing_amplitude": 0.0, "knee_bend": 0.0,
                "arm_swing_amplitude": 0.0, "arm_hang_rad": -1.45,
            },
        },
        {
            "name": "b",
            "duration_sec": 1.0,
            "blend_in_sec": 0.4,
            "bones": {"head": {"x_rad": 1.0}},
        },
    ]
    parsed = parse_animation(doc)
    t_now = [0.8]  # mid-blend
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    # B's head x_rad=1.0 → quat[0] = sin(0.5). Slerp-with-identity would
    # give sin(0.25) at t=0.5 (visibly different + briefly T-pose).
    head_rot = scene.find("head").transform.rotation
    np.testing.assert_allclose(head_rot[0], math.sin(0.5), atol=1e-3)


def test_pose_unknown_name_logged_and_skipped() -> None:
    """An unknown pose name doesn't raise — it's logged and the
    runtime continues with no preset contribution."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["phases"][0]["pose"] = "no_such_pose"
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)  # must not raise
    # Head stays at rest.
    head_rot = scene.find("head").transform.rotation
    np.testing.assert_allclose(head_rot, [0, 0, 0, 1], atol=1e-3)


def test_crossfade_blends_body_yaw_at_boundary() -> None:
    """Body yaw at the cross-fade window's midpoint is the lerp midpoint
    of the two phases' yaw curves. This is the most direct test of the
    blend math — gait and bones add complexity but body fields are
    pure scalar lerps."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["loop_sec"] = 2.0
    doc["phases"] = [
        {
            "name": "a",
            "duration_sec": 1.0,
            "blend_out_sec": 0.4,
            "body": {"yaw_rad": 0.0},
        },
        {
            "name": "b",
            "duration_sec": 1.0,
            "blend_in_sec": 0.4,
            "body": {"yaw_rad": "pi"},
        },
    ]
    parsed = parse_animation(doc)
    t_now = [0.8]  # 0.2 s remaining in phase a, overlap 0.4 → blend_t = 0.5
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    root = scene.find("Sketchfab_model")
    # Yaw lerps 0 → π so at blend_t = 0.5 the yaw is π/2.
    # Quaternion for Y-axis π/2 → (0, sin(π/4), 0, cos(π/4)).
    np.testing.assert_allclose(
        root.transform.rotation[1], math.sin(math.pi / 4), atol=1e-3,
    )


def test_crossfade_blends_bones_with_slerp_at_boundary() -> None:
    """Bone rotation in the cross-fade midpoint matches the slerp
    midpoint of the two phases' computed deltas — slerp not lerp,
    so the rotation stays on the unit hypersphere."""
    from posecascade.scripting.declarative import (  # noqa: PLC0415
        _euler_zyx_quat,
    )
    from posecascade.utils.math3d import quat_slerp  # noqa: PLC0415
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["loop_sec"] = 2.0
    doc["phases"] = [
        {
            "name": "a",
            "duration_sec": 1.0,
            "blend_out_sec": 0.4,
            "bones": {"head": {"x_rad": 0.0}},
        },
        {
            "name": "b",
            "duration_sec": 1.0,
            "blend_in_sec": 0.4,
            "bones": {"head": {"x_rad": 1.0}},
        },
    ]
    parsed = parse_animation(doc)
    t_now = [0.8]  # blend midpoint
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    head_rot = scene.find("head").transform.rotation
    # Expected = slerp midpoint between identity and X(1.0).
    expected = quat_slerp(
        _euler_zyx_quat(0.0, 0.0, 0.0),
        _euler_zyx_quat(1.0, 0.0, 0.0),
        0.5,
    )
    np.testing.assert_allclose(head_rot, expected, atol=1e-3)


def test_crossfade_blends_morphs_at_boundary() -> None:
    """Morph weights lerp during the cross-fade window."""
    from posecascade.scripting.morph_api import MorphApi  # noqa: PLC0415
    scene = _build_minimal_scene()
    morph_api = MorphApi()
    doc = _minimal_doc()
    doc["loop_sec"] = 2.0
    doc["phases"] = [
        {
            "name": "a",
            "duration_sec": 1.0,
            "blend_out_sec": 0.4,
            "morphs": {"smile": 0.0},
        },
        {
            "name": "b",
            "duration_sec": 1.0,
            "blend_in_sec": 0.4,
            "morphs": {"smile": 1.0},
        },
    ]
    parsed = parse_animation(doc)
    t_now = [0.8]
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
        morph_api=morph_api,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    weights = dict(morph_api.current_weights())
    assert weights["smile"] == pytest.approx(0.5, abs=1e-3)


def test_crossfade_only_when_both_phases_consent() -> None:
    """If either blend_in_sec or blend_out_sec is 0, no blending happens
    — the boundary is a hard cut. Authors opt in by setting both."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["loop_sec"] = 2.0
    doc["phases"] = [
        {
            "name": "a",
            "duration_sec": 1.0,
            "blend_out_sec": 0.4,
            "body": {"yaw_rad": 0.0},
        },
        {
            "name": "b",
            "duration_sec": 1.0,
            # No blend_in_sec → blending suppressed.
            "body": {"yaw_rad": "pi"},
        },
    ]
    parsed = parse_animation(doc)
    t_now = [0.999]  # last frame of phase a, no blend → still phase a's yaw
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    root = scene.find("Sketchfab_model")
    # Phase a's yaw is 0 → identity → quat[1] ≈ 0.
    np.testing.assert_allclose(root.transform.rotation[1], 0.0, atol=1e-3)


def test_crossfade_outside_window_only_current_phase_visible() -> None:
    """At elapsed times BEFORE the blend_out_sec window, only the
    current phase's output is visible — no premature blending."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["loop_sec"] = 2.0
    doc["phases"] = [
        {
            "name": "a",
            "duration_sec": 1.0,
            "blend_out_sec": 0.4,
            "body": {"yaw_rad": 0.0},
        },
        {
            "name": "b",
            "duration_sec": 1.0,
            "blend_in_sec": 0.4,
            "body": {"yaw_rad": "pi"},
        },
    ]
    parsed = parse_animation(doc)
    t_now = [0.5]  # Mid phase a, well before blend window starts at t=0.6
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    root = scene.find("Sketchfab_model")
    # Pure phase a → yaw = 0 → identity.
    np.testing.assert_allclose(root.transform.rotation[1], 0.0, atol=1e-3)


def test_parse_bpm_and_duration_beats_resolve_to_seconds() -> None:
    """A phase declared in beats with bpm=120 has duration_sec = beats / 2."""
    doc = _minimal_doc()
    doc["bpm"] = 120.0
    doc["loop_sec"] = 4.0
    doc["phases"][0].pop("duration_sec", None)
    doc["phases"][0]["duration_beats"] = 8.0  # 8 beats @ 120 bpm = 4.0 s
    parsed = parse_animation(doc)
    assert parsed.bpm == pytest.approx(120.0)
    assert parsed.phases[0].duration_sec == pytest.approx(4.0)


def test_parse_rejects_duration_beats_without_bpm() -> None:
    """duration_beats requires document-level bpm > 0."""
    doc = _minimal_doc()
    doc["phases"][0].pop("duration_sec", None)
    doc["phases"][0]["duration_beats"] = 4.0
    with pytest.raises(DeclarativeAnimationError, match="bpm"):
        parse_animation(doc)


def test_parse_rejects_both_duration_sec_and_beats() -> None:
    """Specifying both is ambiguous and rejected at parse time."""
    doc = _minimal_doc()
    doc["bpm"] = 120.0
    doc["phases"][0]["duration_beats"] = 4.0  # already has duration_sec
    with pytest.raises(DeclarativeAnimationError, match="exactly one"):
        parse_animation(doc)


def test_runtime_beat_variable_in_expression_scope() -> None:
    """Expressions can reference ``beat`` (= elapsed * bpm / 60) and
    ``phase_beat`` (= phase_elapsed * bpm / 60); both come from the
    document-level bpm."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["bpm"] = 120.0
    doc["loop_sec"] = 4.0
    doc["phases"][0]["duration_sec"] = 4.0
    doc["phases"][0]["body"] = {
        # At elapsed=1.0s with bpm=120, beat=2.0; sin(2 * tau / 4) = sin(π) = 0
        "yaw_rad": "sin(beat * tau / 4)",
        "translation": {"x": "phase_beat / 4.0", "y": 0.0, "z": 0.0},
    }
    parsed = parse_animation(doc)
    t_now = [1.0]  # beat = 2.0, phase_beat = 2.0
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    root = scene.find("Sketchfab_model")
    # yaw = sin(π) ≈ 0 → quaternion W ≈ 1, others ≈ 0
    np.testing.assert_allclose(
        root.transform.rotation[1], 0.0, atol=1e-3,
    )
    # x translation = phase_beat / 4 = 0.5
    np.testing.assert_allclose(
        root.transform.translation[0], 0.5, atol=1e-3,
    )


def test_runtime_beat_zero_when_bpm_unset() -> None:
    """Documents without bpm get beat=0 in scope so legacy expressions
    referencing it (unlikely but possible) get a defined value rather
    than a NameError."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["phases"][0]["body"] = {
        "yaw_rad": "beat * pi",  # bpm unset → beat is 0 → yaw is 0
    }
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 1.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    root = scene.find("Sketchfab_model")
    np.testing.assert_allclose(root.transform.rotation[1], 0.0, atol=1e-3)


def test_curve_step_returns_from_then_to_at_threshold() -> None:
    """``step`` is the discrete-jump curve: ``from`` strictly below
    the ``at`` threshold, ``to`` at or above it."""
    from posecascade.scripting.declarative import _resolve_value_curve  # noqa: PLC0415
    spec = {"kind": "step", "from": -1.0, "to": 1.0, "at": 0.4}
    assert _resolve_value_curve(spec, 0.0) == pytest.approx(-1.0)
    assert _resolve_value_curve(spec, 0.39) == pytest.approx(-1.0)
    assert _resolve_value_curve(spec, 0.40) == pytest.approx(1.0)
    assert _resolve_value_curve(spec, 1.0) == pytest.approx(1.0)
    # Default threshold is 0.5.
    spec_default = {"kind": "step", "from": 0.0, "to": 1.0}
    assert _resolve_value_curve(spec_default, 0.49) == pytest.approx(0.0)
    assert _resolve_value_curve(spec_default, 0.50) == pytest.approx(1.0)


def test_curve_quad_and_cubic_in_out_endpoints_and_midpoint() -> None:
    """Quadratic and cubic ease-in/-out curves anchor at the endpoints
    and produce the expected midpoint values (0.25 / 0.75 for quad,
    0.125 / 0.875 for cubic)."""
    from posecascade.scripting.declarative import _resolve_value_curve  # noqa: PLC0415
    base = {"from": 0.0, "to": 1.0}
    cases = (
        ({"kind": "quad-in", **base},   0.25),
        ({"kind": "quad-out", **base},  0.75),
        ({"kind": "cubic-in", **base},  0.125),
        ({"kind": "cubic-out", **base}, 0.875),
    )
    for spec, mid in cases:
        assert _resolve_value_curve(spec, 0.0) == pytest.approx(0.0, abs=1e-6), spec
        assert _resolve_value_curve(spec, 1.0) == pytest.approx(1.0, abs=1e-6), spec
        assert _resolve_value_curve(spec, 0.5) == pytest.approx(mid, abs=1e-6), spec


def test_curve_back_out_overshoots_then_lands_on_target() -> None:
    """``back-out`` lands exactly on ``to`` at t=1 and overshoots somewhere
    inside the [0,1] interval (Penner's classic ease-out-back).
    """
    from posecascade.scripting.declarative import _resolve_value_curve  # noqa: PLC0415
    spec = {"kind": "back-out", "from": 0.0, "to": 1.0}
    assert _resolve_value_curve(spec, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert _resolve_value_curve(spec, 1.0) == pytest.approx(1.0, abs=1e-6)
    # Somewhere in the upper half the value briefly exceeds 1.0 — that
    # IS the visible overshoot. Use a coarse scan to find at least one.
    samples = [_resolve_value_curve(spec, i / 100.0) for i in range(101)]
    assert max(samples) > 1.01, (
        f"back-out should overshoot above 1.0 with default coefficient; max={max(samples)}"
    )
    # Larger overshoot → more visible kick.
    big = {"kind": "back-out", "from": 0.0, "to": 1.0, "overshoot": 4.0}
    big_samples = [_resolve_value_curve(big, i / 100.0) for i in range(101)]
    assert max(big_samples) > max(samples)


def test_curve_pulse_zero_outside_window_and_peaks_at_center() -> None:
    """``pulse`` returns ``from`` outside the window, ``to`` at the
    window's centre, and the half-sine bell in between."""
    from posecascade.scripting.declarative import _resolve_value_curve  # noqa: PLC0415
    spec = {"kind": "pulse", "from": 0.0, "to": 2.0, "center": 0.5, "width": 0.4}
    # Outside [0.3, 0.7] → from
    assert _resolve_value_curve(spec, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert _resolve_value_curve(spec, 0.29) == pytest.approx(0.0, abs=1e-6)
    assert _resolve_value_curve(spec, 0.71) == pytest.approx(0.0, abs=1e-6)
    # At centre → to
    assert _resolve_value_curve(spec, 0.5) == pytest.approx(2.0, abs=1e-3)
    # Width 0 degenerate → always from (no division by zero).
    degenerate = {"kind": "pulse", "from": 0.0, "to": 1.0, "width": 0.0}
    assert _resolve_value_curve(degenerate, 0.5) == pytest.approx(0.0)


def test_curve_unknown_kind_lists_supported_in_error() -> None:
    """Typo'd ``kind`` raises with the supported list so the author
    sees the new options without scrolling docs."""
    from posecascade.scripting.declarative import _resolve_value_curve  # noqa: PLC0415
    with pytest.raises(DeclarativeAnimationError, match="back-out"):
        _resolve_value_curve({"kind": "qaud-in"}, 0.5)


def test_parse_bones_section_accepts_per_axis_curves() -> None:
    """Minimal ``bones`` block parses; unknown axes raise."""
    doc = _minimal_doc()
    doc["phases"][0]["bones"] = {
        "head": {"y_rad": 0.5, "x_rad": "0.1 * sin(elapsed * tau)"},
    }
    parsed = parse_animation(doc)
    assert "head" in parsed.phases[0].bones
    assert parsed.phases[0].bones["head"]["y_rad"] == 0.5

    doc_bad = _minimal_doc()
    doc_bad["phases"][0]["bones"] = {"head": {"x_red": 0.5}}  # typo
    with pytest.raises(DeclarativeAnimationError, match="unknown axis"):
        parse_animation(doc_bad)


def test_parse_bones_rejects_non_dict_entry() -> None:
    """``bones[name]`` must itself be an object — not a scalar."""
    doc = _minimal_doc()
    doc["phases"][0]["bones"] = {"head": 0.5}
    with pytest.raises(DeclarativeAnimationError, match="must be an object"):
        parse_animation(doc)


def test_runtime_bones_drives_x_rotation_on_named_bone() -> None:
    """A constant ``x_rad`` curve produces a matching X-axis rotation on the bone."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["phases"][0]["bones"] = {"head": {"x_rad": 0.5}}
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    head_rot = scene.find("head").transform.rotation
    # Quaternion (sin(0.25), 0, 0, cos(0.25)) for X rotation by 0.5 rad.
    expected_x = math.sin(0.25)
    np.testing.assert_allclose(head_rot[0], expected_x, atol=1e-3)
    np.testing.assert_allclose(head_rot[1], 0.0, atol=1e-3)
    np.testing.assert_allclose(head_rot[2], 0.0, atol=1e-3)


def test_runtime_bones_axes_compose_in_zyx_order() -> None:
    """Multi-axis curves compose as ``Rz · Ry · Rx``.

    Verified by checking against ``_euler_zyx_quat`` directly — that
    helper IS the spec, and the runtime must apply it identically so
    authors can predict the resulting orientation.
    """
    from posecascade.scripting.declarative import (  # noqa: PLC0415
        _euler_zyx_quat,
    )
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["phases"][0]["bones"] = {
        "head": {"x_rad": 0.3, "y_rad": 0.2, "z_rad": 0.1},
    }
    parsed = parse_animation(doc)
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    expected = _euler_zyx_quat(0.3, 0.2, 0.1)
    actual = scene.find("head").transform.rotation
    np.testing.assert_allclose(actual, expected, atol=1e-3)


def test_runtime_bones_overrides_gait_when_same_bone() -> None:
    """When both gait and bones write the same bone, bones wins.

    Walking gait swings ``upper_arm_L`` per its amplitude; an explicit
    ``bones`` entry on the same bone with a constant curve must end up
    on the bone after the frame, not the gait's swing. This is the core
    "hold-pose-while-others-walk" use case.
    """
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["phases"][0]["gait"] = {
        "kind": "walking", "step_cycle_sec": 1.0,
        "leg_swing_amplitude": 0.0, "knee_bend": 0.0,
        "arm_swing_amplitude": 0.6, "arm_hang_rad": 0.0,
    }
    doc["phases"][0]["bones"] = {
        "upper_arm_L": {"x_rad": 1.2},  # held overhead
    }
    parsed = parse_animation(doc)
    t_now = [0.25]  # peak swing time — gait would normally write a big delta here
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    arm_l = scene.find("upper_arm_L").transform.rotation
    # Bones wrote pure X rotation by 1.2 rad → q = (sin(0.6), 0, 0, cos(0.6)).
    np.testing.assert_allclose(arm_l[0], math.sin(0.6), atol=1e-3)
    np.testing.assert_allclose(arm_l[2], 0.0, atol=1e-3)


def test_runtime_bones_curve_uses_frame_scope_variables() -> None:
    """Bone-axis curves are evaluated against the per-frame scope so
    expressions referencing ``elapsed`` / ``phase_t`` work."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()
    doc["loop_sec"] = 1.0
    doc["phases"][0]["duration_sec"] = 1.0
    doc["phases"][0]["bones"] = {
        "head": {"y_rad": "phase_t * pi"},
    }
    parsed = parse_animation(doc)
    t_now = [0.5]  # phase_t = 0.5 → angle = π/2
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: t_now[0],
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)
    head_rot = scene.find("head").transform.rotation
    # Y-axis rotation by π/2 → q = (0, sin(π/4), 0, cos(π/4))
    np.testing.assert_allclose(head_rot[1], math.sin(math.pi / 4), atol=1e-3)


def test_runtime_bones_yaw_conjugation_invariant_to_root_yaw() -> None:
    """The same body-frame ``x_rad`` curve produces the same bone-local
    rotation under any root yaw — the conjugation pass cancels yaw out
    in body frame, mirroring the gait's invariance."""
    def _head_local_rotation(yaw_value: object) -> np.ndarray:
        scene = _build_minimal_scene()
        doc = _minimal_doc()
        doc["loop_sec"] = 1.0
        doc["phases"][0]["duration_sec"] = 1.0
        doc["phases"][0]["body"]["yaw_rad"] = yaw_value
        doc["phases"][0]["bones"] = {"head": {"x_rad": 0.4}}
        parsed = parse_animation(doc)
        runtime = DeclarativeRuntime(
            animation=parsed, scene=scene, time=lambda: 0.0,
        )
        hooks = runtime.hooks()
        hooks["start"]()
        hooks["update"](0.0)
        return scene.find("head").transform.rotation.copy()

    rot_zero = _head_local_rotation(0.0)
    rot_pi = _head_local_rotation("pi")
    np.testing.assert_allclose(rot_zero, rot_pi, atol=1e-3)


def test_runtime_phase_without_bones_is_backward_compatible() -> None:
    """An old document with no ``bones`` field still parses and ticks
    cleanly — the field defaults to an empty dict and the runtime skips
    ``_apply_bones`` entirely."""
    scene = _build_minimal_scene()
    doc = _minimal_doc()  # no bones key
    parsed = parse_animation(doc)
    assert parsed.phases[0].bones == {}
    runtime = DeclarativeRuntime(
        animation=parsed, scene=scene, time=lambda: 0.0,
    )
    hooks = runtime.hooks()
    hooks["start"]()
    hooks["update"](0.0)  # must not raise


def test_walk_example_runs_full_loop_without_errors() -> None:
    """The shipped walk JSON parses, binds, and runs every frame of its
    loop without raising. Guards the per-phase expression DSL inside
    body translation / yaw / lean / gait fields so a typo in the
    example would surface as a test failure rather than silently
    breaking the demo."""
    from pathlib import Path  # noqa: PLC0415

    scene = _build_minimal_scene()
    t_now = [0.0]
    api = {
        "scene": scene,
        "time": lambda: t_now[0],
    }
    walk_path = (
        Path(__file__).parent.parent / "examples" / "scripts" / "walk.json"
    )
    source = walk_path.read_text(encoding="utf-8")
    # Pass the full path so ``extends`` resolves against the example's
    # directory; the bare filename would search CWD and miss the
    # ``_herta_profile.json`` sibling.
    hooks = load_animation(source, str(walk_path), api)
    hooks["start"]()
    fps = 30
    import json as _json  # noqa: PLC0415
    total_seconds = float(_json.loads(source)["loop_sec"])
    for i in range(int(total_seconds * fps)):
        t_now[0] = i / fps
        hooks["update"](1.0 / fps)


# ---------------------------------------------------------------------------
# extends — JSON profile inheritance.
# ---------------------------------------------------------------------------


def test_extends_merges_profile_into_child(tmp_path) -> None:
    """A child JSON's ``extends`` pulls the parent's top-level keys in.

    Child overrides parent on a per-key basis at the top level; parent
    values pass through untouched for any key the child omits.
    """
    from posecascade.scripting.declarative import resolve_extends  # noqa: PLC0415

    profile = {
        "schema_version": 1,
        "rig": {"character_root": "Root"},
        "physics_chains": {"hair_L": {"stiffness": 1.0}},
        "wind": {"speed": 0.05},
    }
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    child = {
        "schema_version": 1,
        "extends": "profile.json",
        "name": "child",
        "wind": {"speed": 0.20},  # overrides parent's speed
        "phases": [{"name": "p", "duration_sec": 1.0}],
    }
    merged = resolve_extends(child, tmp_path)
    # Inherited verbatim:
    assert merged["rig"]["character_root"] == "Root"
    assert merged["physics_chains"]["hair_L"]["stiffness"] == 1.0
    # Overridden:
    assert merged["wind"]["speed"] == 0.20
    # ``extends`` field is consumed.
    assert "extends" not in merged


def test_extends_rejects_traversal_outside_source_dir(tmp_path) -> None:
    """``extends`` paths that try to escape via ``..`` raise loudly."""
    from posecascade.scripting.declarative import resolve_extends  # noqa: PLC0415

    child = {"schema_version": 1, "extends": "../escape.json"}
    with pytest.raises(DeclarativeAnimationError, match="rejected"):
        resolve_extends(child, tmp_path)


def test_extends_rejects_cycle(tmp_path) -> None:
    """A → B → A cycle raises rather than looping forever."""
    from posecascade.scripting.declarative import resolve_extends  # noqa: PLC0415

    (tmp_path / "a.json").write_text(
        json.dumps({"schema_version": 1, "extends": "b.json"}),
    )
    (tmp_path / "b.json").write_text(
        json.dumps({"schema_version": 1, "extends": "a.json"}),
    )
    child = {"schema_version": 1, "extends": "a.json"}
    with pytest.raises(DeclarativeAnimationError, match="cycle"):
        resolve_extends(child, tmp_path)


def test_extends_never_inherits_parent_phases(tmp_path) -> None:
    """Phases are choreography, never inherited — even if the parent has them."""
    from posecascade.scripting.declarative import resolve_extends  # noqa: PLC0415

    parent_with_phases = {
        "schema_version": 1,
        "phases": [{"name": "parent_phase", "duration_sec": 5.0}],
    }
    (tmp_path / "parent.json").write_text(json.dumps(parent_with_phases))
    child = {
        "schema_version": 1,
        "extends": "parent.json",
        "phases": [{"name": "child_phase", "duration_sec": 1.0}],
    }
    merged = resolve_extends(child, tmp_path)
    assert [p["name"] for p in merged["phases"]] == ["child_phase"]


def test_extends_pose_library_merges_per_preset(tmp_path) -> None:
    """``pose_library`` and ``hand_library`` are the only top-level keys that
    deep-merge — a child can add a new pose without redeclaring every
    inherited one."""
    from posecascade.scripting.declarative import resolve_extends  # noqa: PLC0415

    profile = {
        "schema_version": 1,
        "pose_library": {"a": {"head": {"x_rad": 0.1}}},
    }
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    child = {
        "schema_version": 1,
        "extends": "profile.json",
        "pose_library": {"b": {"head": {"x_rad": 0.2}}},
    }
    merged = resolve_extends(child, tmp_path)
    # Both presets present — parent's "a" survived alongside child's "b".
    assert set(merged["pose_library"]) == {"a", "b"}


# ---------------------------------------------------------------------------
# Shorthand syntax — [from, to] curves, [x, y, z] translation, bone axis aliases.
# ---------------------------------------------------------------------------


def test_curve_array_shorthand_resolves_as_linear() -> None:
    """``[from, to]`` arrays evaluate the same as ``{kind: 'linear', from, to}``."""
    from posecascade.scripting.declarative import _resolve_value_curve  # noqa: PLC0415

    # phase_t at the midpoint should land halfway between endpoints.
    assert _resolve_value_curve([2.0, 8.0], 0.5) == 5.0
    # At endpoints, exact:
    assert _resolve_value_curve([2.0, 8.0], 0.0) == 2.0
    assert _resolve_value_curve([2.0, 8.0], 1.0) == 8.0


def test_translation_array_shorthand_unpacks_per_axis() -> None:
    """``[x, y, z]`` translation array evaluates per-axis the same as the dict form."""
    from posecascade.scripting.declarative import _resolve_translation  # noqa: PLC0415

    # Static origin via shorthand.
    out = _resolve_translation([0.0, 1.5, -2.0], 0.5, {})
    assert out == (0.0, 1.5, -2.0)
    # Each axis is itself a value curve — animated middle axis works.
    out = _resolve_translation([0.0, [0.0, 4.0], 0.0], 0.25, {})
    assert out[0] == 0.0
    assert out[1] == 1.0   # linear 0→4 at t=0.25
    assert out[2] == 0.0


def test_translation_array_rejects_wrong_length() -> None:
    """A 2- or 4-element translation array is an authoring error."""
    from posecascade.scripting.declarative import _resolve_translation  # noqa: PLC0415

    with pytest.raises(DeclarativeAnimationError, match="3 entries"):
        _resolve_translation([0.0, 1.0], 0.0, {})


def test_bone_axis_aliases_accept_short_form() -> None:
    """``{x: ...}`` reads as ``{x_rad: ...}`` and mixing the two raises."""
    doc = _minimal_doc()
    doc["phases"][0]["bones"] = {"head": {"x": 0.5}}
    parsed = parse_animation(doc)
    assert parsed.phases[0].bones["head"]["x_rad"] == 0.5

    doc_mix = _minimal_doc()
    doc_mix["phases"][0]["bones"] = {"head": {"x": 0.5, "x_rad": 0.5}}
    with pytest.raises(DeclarativeAnimationError, match="twice"):
        parse_animation(doc_mix)

