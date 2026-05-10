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


def test_hide_detaches_named_nodes_from_scene() -> None:
    """Document-root 'hide' detaches each named node from its parent at
    start. Useful for character.glb files that bundle props (Stairs,
    room, lights) the dance doesn't want."""
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
    with pytest.raises(DeclarativeAnimationError, match="unknown axes"):
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


def test_dance_example_runs_full_loop_without_errors() -> None:
    """The shipped MMD-style dance JSON parses, binds, and runs every
    frame of its 16 s loop without raising. Guards the per-phase
    expression DSL inside body translation / yaw / lean / morph fields
    so a typo in the example would surface as a test failure rather
    than silently breaking the demo."""
    from pathlib import Path  # noqa: PLC0415

    from posecascade.scripting.morph_api import MorphApi  # noqa: PLC0415
    scene = _build_minimal_scene()
    morph_api = MorphApi()
    api = {
        "scene": scene,
        "time": lambda: t_now[0],
        "morphs": morph_api,
    }
    source = (
        Path(__file__).parent.parent / "examples" / "scripts" / "dance.json"
    ).read_text(encoding="utf-8")
    hooks = load_animation(source, "dance.json", api)
    t_now = [0.0]
    hooks["start"]()
    fps = 30
    # Cover the full loop length so every phase + every cross-fade
    # window gets a frame, and the final wrap back to phase 0 also
    # ticks. Re-fetched from the parsed document so the test doesn't
    # hard-code a length the example may grow past.
    import json as _json  # noqa: PLC0415
    total_seconds = float(_json.loads(source)["loop_sec"])
    for i in range(int(total_seconds * fps)):
        t_now[0] = i / fps
        hooks["update"](1.0 / fps)
    weights = dict(morph_api.current_weights())
    assert "smile" in weights
    # Smile is always written by at least one phase — exact final
    # value depends on how the dance choreography evolves, so assert
    # it stays in [0, 1] not at a fixed endpoint.
    assert 0.0 <= weights["smile"] <= 1.0
