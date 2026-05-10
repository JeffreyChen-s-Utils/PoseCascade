"""Declarative animation runtime — JSON-driven script alternative.

Walks a JSON document describing an animation timeline (phases, body
trajectory, bone drivers, rig bindings, ground geometry, springs) and
exposes ``start`` / ``update`` / ``on_event`` hooks compatible with
:class:`~posecascade.scripting.host.ScriptHost`. Lets non-programmer
users author demos as data instead of Python.

Schema overview (see ``schema_v1`` below for full JSON-schema):

.. code-block:: json

    {
      "schema_version": 1,
      "name": "stair_walk",
      "loop_sec": 16.0,
      "rig": {
        "character_root": "Sketchfab_model",
        "leg_chain_l": ["upper_leg_L", "lower_leg_L", "foot_L"],
        "leg_chain_r": ["upper_leg_R", "lower_leg_R", "foot_R"],
        "knee_limit_min": [-2.4, 0, 0],
        "knee_limit_max": [0.1, 0, 0]
      },
      "ground": {"kind": "stairs",
                 "base_z": -0.20, "step_depth": 0.04, "step_rise": 0.02,
                 "count": 5, "forward_sign": -1},
      "phases": [
        {"name": "walk_forward", "duration_sec": 4.0,
         "body": {"yaw_rad": "pi",
                  "translation": {"z": {"kind": "linear",
                                        "from": 0.0, "to": -0.20}}},
         "gait": {"kind": "walking", "step_cycle_sec": 1.0,
                  "leg_swing_amplitude": 0.50, "knee_bend": -0.30,
                  "arm_swing_amplitude": 0.40}}
      ]
    }

Covered: phase timing, body trajectory (incl. stair shortcut), walking
+ stride gaits with body-yaw conjugation and T-pose→hang composition,
ground binding + auto-bound foot planter, physics chain tunings, wind
setup, per-phase morph curves, lock-target IK on the trailing foot,
and an inline expression DSL for value curves.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from posecascade.errors import PoseCascadeError, ScriptSecurityError
from posecascade.scripting.expressions import (
    ExpressionError,
    evaluate_expression,
    looks_like_expression,
)
from posecascade.utils.logging import get_logger
from posecascade.utils.math3d import (
    quat_from_axis_angle,
    quat_inverse,
    quat_mul,
    quat_slerp,
    vec3,
)

quat_axis_angle = quat_from_axis_angle

_log = get_logger(__name__)

_TAU = math.tau
_TWO = 2.0
_HALF = 0.5
_DECLARATIVE_SCHEMA_VERSION = 1
_VEC3_LEN = 3
_YAW_NEGLIGIBLE = 1e-4
# Identity rotation for cross-fade blending: a bone present in only one
# of the two phases being blended is implicitly at rest in the other.
_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
# Default Z-tuck that pulls arms from T-pose down to a vertical hang.
# Slightly less than ±π/2 so arms angle a few degrees out from the torso
# instead of clipping the rib cage. Matches walk.py's ARM_HANG.
_DEFAULT_ARM_HANG_RAD = -1.45
# Frames over which a released lock target decays its IK pull. ~6
# frames at 60 FPS = 0.1 s — long enough that the trailing foot
# tracks the body softly across the step boundary, short enough
# that it's free to lift before the new stride's leading-leg envelope
# starts ramping at step_t ≈ 0.10.
_LOCK_RELEASE_FRAMES = 6


class DeclarativeAnimationError(PoseCascadeError):
    """Raised when the JSON document fails schema validation."""


# --- Scalar / value resolution ----------------------------------------------


_SYMBOLIC_FLOATS = {
    "pi": math.pi,
    "-pi": -math.pi,
    "2pi": 2 * math.pi,
    "tau": math.tau,
    "pi/2": math.pi / 2,
    "-pi/2": -math.pi / 2,
}


def _resolve_scalar(value: Any, scope: dict[str, float] | None = None) -> float:
    """Resolve a scalar that may be a number, symbolic constant, or expression.

    ``scope`` is the per-frame variable bag (``elapsed``, ``phase_t``,
    ``phase_elapsed``); pass ``None`` for parse-time resolution where
    only :data:`_SYMBOLIC_FLOATS` are accepted. Scalars containing
    arithmetic operators or function calls are routed through the
    expression DSL so animation authors can write
    ``"amplitude * sin(elapsed * tau)"`` directly in the JSON value.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value in _SYMBOLIC_FLOATS:
            return _SYMBOLIC_FLOATS[value]
        if looks_like_expression(value):
            try:
                return evaluate_expression(value, scope or {})
            except ExpressionError as err:
                raise DeclarativeAnimationError(str(err)) from err
        try:
            return float(value)
        except ValueError as err:
            raise DeclarativeAnimationError(
                f"unrecognised scalar string {value!r}; expected a number, "
                f"an expression, or one of {sorted(_SYMBOLIC_FLOATS)}",
            ) from err
    raise DeclarativeAnimationError(
        f"expected scalar, got {type(value).__name__}: {value!r}",
    )


# Penner ease-out-back default overshoot — the visible "snap past the
# target before settling" coefficient. 1.70158 is the canonical value
# from Robert Penner's easing equations and matches MMD / After Effects
# defaults so authors get the look they expect without tuning.
_BACK_OUT_DEFAULT_OVERSHOOT = 1.70158


def _from_to(spec: dict[str, Any], scope: dict[str, float]) -> tuple[float, float]:
    """Resolve a curve's ``from`` / ``to`` endpoints.

    Extracted because every interpolating curve kind reads exactly
    these two fields the same way; without the helper the rule against
    duplicated 3-statement blocks bites once we have ten curve kinds.
    """
    a = _resolve_scalar(spec.get("from", 0.0), scope)
    b = _resolve_scalar(spec.get("to", 0.0), scope)
    return a, b


def _curve_constant(spec: dict[str, Any], scope: dict[str, float], _t: float) -> float:
    return _resolve_scalar(spec.get("value", 0.0), scope)


def _curve_linear(spec: dict[str, Any], scope: dict[str, float], t: float) -> float:
    a, b = _from_to(spec, scope)
    return a + (b - a) * t


def _curve_ease(spec: dict[str, Any], scope: dict[str, float], t: float) -> float:
    a, b = _from_to(spec, scope)
    eased = _HALF - _HALF * math.cos(t * math.pi)
    return a + (b - a) * eased


def _curve_expression(
    spec: dict[str, Any], scope: dict[str, float], _t: float,
) -> float:
    source = spec.get("source", "0")
    if not isinstance(source, str):
        raise DeclarativeAnimationError(
            f"expression curve 'source' must be str, got {type(source).__name__}",
        )
    try:
        return evaluate_expression(source, scope)
    except ExpressionError as err:
        raise DeclarativeAnimationError(str(err)) from err


def _curve_step(spec: dict[str, Any], scope: dict[str, float], t: float) -> float:
    """Discrete jump from ``from`` to ``to`` at ``at`` (default 0.5).

    Useful for cymbal-crash accent moves where the bone teleports to a
    new pose on a beat instead of easing into it.
    """
    a, b = _from_to(spec, scope)
    at = _resolve_scalar(spec.get("at", _HALF), scope)
    return b if t >= at else a


def _curve_quad_in(spec: dict[str, Any], scope: dict[str, float], t: float) -> float:
    a, b = _from_to(spec, scope)
    return a + (b - a) * (t * t)


def _curve_quad_out(spec: dict[str, Any], scope: dict[str, float], t: float) -> float:
    a, b = _from_to(spec, scope)
    return a + (b - a) * (1.0 - (1.0 - t) ** 2)


def _curve_cubic_in(spec: dict[str, Any], scope: dict[str, float], t: float) -> float:
    a, b = _from_to(spec, scope)
    return a + (b - a) * (t ** 3)


def _curve_cubic_out(spec: dict[str, Any], scope: dict[str, float], t: float) -> float:
    a, b = _from_to(spec, scope)
    return a + (b - a) * (1.0 - (1.0 - t) ** 3)


def _curve_back_out(spec: dict[str, Any], scope: dict[str, float], t: float) -> float:
    """Penner ease-out-back: overshoot the target then settle on it.

    The classic "snap past then ease back" feel of MMD pose hits. At
    ``t=1`` the curve always lands exactly on ``to``; the overshoot is
    visible somewhere around ``t≈0.7``. Larger ``overshoot`` → more
    visible kick.
    """
    a, b = _from_to(spec, scope)
    c1 = _resolve_scalar(spec.get("overshoot", _BACK_OUT_DEFAULT_OVERSHOOT), scope)
    c2 = c1 + 1.0
    eased = 1.0 + c2 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2
    return a + (b - a) * eased


def _curve_pulse(spec: dict[str, Any], scope: dict[str, float], t: float) -> float:
    """Bell-shaped excursion away from ``from`` toward ``to`` and back.

    Output is ``from`` outside the window ``[center − width/2,
    center + width/2]`` and reaches ``to`` at the window's centre.
    The half-sine bell makes the excursion smooth on both sides — drop
    a ``pulse`` on a beat to get a "thump" without manually authoring
    two ease curves back to back.
    """
    a, b = _from_to(spec, scope)
    center = _resolve_scalar(spec.get("center", _HALF), scope)
    width = _resolve_scalar(spec.get("width", _HALF), scope)
    half = width * _HALF
    lo = center - half
    if t <= lo or t >= center + half or width <= 0.0:
        return a
    progress = (t - lo) / width
    bell = math.sin(progress * math.pi)
    return a + (b - a) * bell


_CURVE_HANDLERS: dict[str, Callable[[dict[str, Any], dict[str, float], float], float]] = {
    "constant": _curve_constant,
    "linear": _curve_linear,
    "ease": _curve_ease,
    "expression": _curve_expression,
    "step": _curve_step,
    "quad-in": _curve_quad_in,
    "quad-out": _curve_quad_out,
    "cubic-in": _curve_cubic_in,
    "cubic-out": _curve_cubic_out,
    "back-out": _curve_back_out,
    "pulse": _curve_pulse,
}


def _resolve_value_curve(
    spec: Any, phase_t: float, scope: dict[str, float] | None = None,
) -> float:
    """Evaluate a per-phase value at normalised phase time ``phase_t`` ∈ [0,1].

    ``spec`` is either a scalar (number / symbolic constant / expression
    string) or a dict with a ``kind`` field. Supported kinds are listed
    in :data:`_CURVE_HANDLERS`; each handler is a small pure function so
    the central dispatcher stays under the cyclomatic-complexity bound.

    Linear / ease use ``from`` and ``to``. Expression takes a ``source``
    string and evaluates it via the safe AST DSL with access to
    ``elapsed`` / ``phase_t`` / ``phase_elapsed`` / math helpers. The
    snappier curves (``quad-in/out``, ``cubic-in/out``, ``back-out``,
    ``pulse``, ``step``) are pure-math interpolators authored for sharp
    MMD-style accent hits — see each handler's docstring.
    """
    eval_scope = dict(scope) if scope else {}
    eval_scope.setdefault("phase_t", float(phase_t))
    if isinstance(spec, (int, float, str)):
        return _resolve_scalar(spec, eval_scope)
    if not isinstance(spec, dict):
        raise DeclarativeAnimationError(
            f"value curve must be scalar or dict, got {type(spec).__name__}",
        )
    kind = spec.get("kind", "constant")
    handler = _CURVE_HANDLERS.get(kind)
    if handler is None:
        raise DeclarativeAnimationError(
            f"unknown value-curve kind {kind!r}; expected one of "
            f"{sorted(_CURVE_HANDLERS)}",
        )
    return handler(spec, eval_scope, float(phase_t))


# --- Schema parsing ---------------------------------------------------------


@dataclass(frozen=True)
class RigBindings:
    character_root: str
    leg_chain_l: tuple[str, str, str] | None
    leg_chain_r: tuple[str, str, str] | None
    knee_limit_min: tuple[float, float, float] | None
    knee_limit_max: tuple[float, float, float] | None
    body_bones: dict[str, str]


@dataclass(frozen=True)
class GroundSpec:
    kind: str
    params: dict[str, Any]


@dataclass(frozen=True)
class Phase:
    name: str
    duration_sec: float
    body_yaw_rad: Any  # scalar or curve spec
    body_lean_x_rad: Any
    body_translation: dict[str, Any]
    gait: dict[str, Any] | None
    morphs: dict[str, Any]  # name → value-curve spec
    # bone_key → {"x_rad"?: curve, "y_rad"?: curve, "z_rad"?: curve}.
    # Composed AFTER gait so authors can override a gait-driven bone with
    # a custom curve (e.g. hold the arm overhead during a finale phase
    # while the walking gait would otherwise swing it).
    bones: dict[str, dict[str, Any]]
    # Cross-fade windows in seconds. When > 0 AND the next phase's
    # ``blend_in_sec`` is also > 0, the runtime evaluates BOTH phases'
    # body / bones / morphs outputs in the overlap window (using the
    # mutual minimum) and lerps between them. Gait is NOT blended —
    # only the current phase's gait runs at any given time, since
    # blending two step-based gaits is ill-defined.
    blend_in_sec: float
    blend_out_sec: float


@dataclass(frozen=True)
class DeclarativeAnimation:
    name: str
    loop_sec: float
    rig: RigBindings
    ground: GroundSpec | None
    phases: tuple[Phase, ...]
    physics_chains: dict[str, dict[str, float]]
    wind: dict[str, Any] | None
    # Beats per minute. Used for the ``beat`` / ``phase_beat`` expression-DSL
    # variables and for resolving any phase that declared ``duration_beats``
    # instead of ``duration_sec``. ``0.0`` means the document has no tempo
    # (durations are seconds and ``beat`` evaluates to 0 in expressions).
    bpm: float


@dataclass
class PhaseOutput:
    """Computed (not yet applied) per-frame output of a single phase.

    Used by the cross-fade path to blend two phases at boundaries
    without writing intermediate state into the scene. ``yaw`` /
    ``lean`` / ``translation`` are scalars / triples, ``bones`` maps
    bone keys to body-frame quaternion deltas (pre yaw-conjugation),
    ``morphs`` maps morph names to weights.
    """

    yaw: float
    lean: float
    translation: tuple[float, float, float]
    bones: dict[str, np.ndarray] = field(default_factory=dict)
    morphs: dict[str, float] = field(default_factory=dict)


def _lerp_translation(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _blend_phase_outputs(a: PhaseOutput, b: PhaseOutput, t: float) -> PhaseOutput:
    """Linearly blend two :class:`PhaseOutput` into one.

    Body fields use scalar lerp. Bone deltas use quaternion slerp so
    intermediate rotations stay on the unit hypersphere (component lerp
    drifts off the manifold and the resulting bone wobble is visible).
    Morph weights use scalar lerp. Bones / morphs that appear in only
    one of the two outputs are blended against the implicit identity:
    a missing bone means "rest pose" (identity quaternion); a missing
    morph means weight 0.
    """
    out = PhaseOutput(
        yaw=a.yaw + (b.yaw - a.yaw) * t,
        lean=a.lean + (b.lean - a.lean) * t,
        translation=_lerp_translation(a.translation, b.translation, t),
    )
    bone_keys = set(a.bones) | set(b.bones)
    for key in bone_keys:
        qa = a.bones.get(key, _IDENTITY_QUAT)
        qb = b.bones.get(key, _IDENTITY_QUAT)
        out.bones[key] = quat_slerp(qa, qb, t)
    morph_keys = set(a.morphs) | set(b.morphs)
    for key in morph_keys:
        wa = a.morphs.get(key, 0.0)
        wb = b.morphs.get(key, 0.0)
        out.morphs[key] = wa + (wb - wa) * t
    return out


def _parse_rig(raw: dict[str, Any]) -> RigBindings:
    if not isinstance(raw, dict):
        raise DeclarativeAnimationError("'rig' must be an object")
    leg_l = raw.get("leg_chain_l")
    leg_r = raw.get("leg_chain_r")
    return RigBindings(
        character_root=str(raw.get("character_root", "")),
        leg_chain_l=tuple(leg_l) if leg_l else None,
        leg_chain_r=tuple(leg_r) if leg_r else None,
        knee_limit_min=(
            tuple(_resolve_scalar(v) for v in raw["knee_limit_min"])
            if "knee_limit_min" in raw else None
        ),
        knee_limit_max=(
            tuple(_resolve_scalar(v) for v in raw["knee_limit_max"])
            if "knee_limit_max" in raw else None
        ),
        body_bones=dict(raw.get("body_bones", {})),
    )


def _parse_ground(raw: Any) -> GroundSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise DeclarativeAnimationError("'ground' must be an object or null")
    kind = raw.get("kind")
    if kind not in {"flat", "stairs"}:
        raise DeclarativeAnimationError(
            f"ground kind {kind!r} not supported; expected 'flat' or 'stairs'",
        )
    return GroundSpec(kind=str(kind), params={k: v for k, v in raw.items() if k != "kind"})


_SECONDS_PER_MINUTE = 60.0


def _resolve_phase_duration(raw: dict[str, Any], bpm: float) -> float:
    """Resolve a phase's duration into seconds.

    Authors may write ``duration_sec`` (always valid) OR ``duration_beats``
    (requires the document-level ``bpm`` to be > 0). Mixing both in the
    same phase is rejected so the source of truth is unambiguous; pure
    backward compat for documents that only ever wrote ``duration_sec``.
    """
    has_sec = "duration_sec" in raw
    has_beats = "duration_beats" in raw
    if has_sec and has_beats:
        raise DeclarativeAnimationError(
            "phase has both 'duration_sec' and 'duration_beats'; "
            "specify exactly one",
        )
    if has_beats:
        if bpm <= 0.0:
            raise DeclarativeAnimationError(
                "phase uses 'duration_beats' but the document has no "
                "positive 'bpm' to convert it",
            )
        return float(raw["duration_beats"]) * _SECONDS_PER_MINUTE / bpm
    return float(raw.get("duration_sec", 0.0))


def _parse_phase(raw: dict[str, Any], bpm: float) -> Phase:
    if not isinstance(raw, dict):
        raise DeclarativeAnimationError("each phase must be an object")
    body = raw.get("body", {})
    if not isinstance(body, dict):
        raise DeclarativeAnimationError("phase 'body' must be an object")
    morphs = raw.get("morphs", {})
    if not isinstance(morphs, dict):
        raise DeclarativeAnimationError("phase 'morphs' must be an object")
    bones = _parse_bones(raw.get("bones", {}))
    blend_in = float(raw.get("blend_in_sec", 0.0))
    blend_out = float(raw.get("blend_out_sec", 0.0))
    if blend_in < 0.0 or blend_out < 0.0:
        raise DeclarativeAnimationError(
            "blend_in_sec / blend_out_sec must be non-negative",
        )
    return Phase(
        name=str(raw.get("name", "")),
        duration_sec=_resolve_phase_duration(raw, bpm),
        body_yaw_rad=body.get("yaw_rad", 0.0),
        body_lean_x_rad=body.get("lean_x_rad", 0.0),
        body_translation=body.get("translation", {}),
        gait=raw.get("gait"),
        morphs=morphs,
        bones=bones,
        blend_in_sec=blend_in,
        blend_out_sec=blend_out,
    )


_BONE_AXES = ("x_rad", "y_rad", "z_rad")


def _parse_bones(raw: Any) -> dict[str, dict[str, Any]]:
    """Validate the per-phase ``bones`` block.

    Each entry is ``{bone_name: {x_rad?: curve, y_rad?: curve, z_rad?: curve}}``.
    Unknown axes are rejected loudly so a typo (``x_red``) surfaces at parse
    time instead of silently producing a still pose at runtime.
    """
    if not isinstance(raw, dict):
        raise DeclarativeAnimationError("phase 'bones' must be an object")
    out: dict[str, dict[str, Any]] = {}
    for bone_name, axes in raw.items():
        if not isinstance(axes, dict):
            raise DeclarativeAnimationError(
                f"bones[{bone_name!r}] must be an object of axis curves",
            )
        unknown = set(axes) - set(_BONE_AXES)
        if unknown:
            raise DeclarativeAnimationError(
                f"bones[{bone_name!r}] has unknown axes {sorted(unknown)}; "
                f"expected any of {list(_BONE_AXES)}",
            )
        out[str(bone_name)] = dict(axes)
    return out


def parse_animation(document: dict[str, Any]) -> DeclarativeAnimation:
    """Validate and parse a declarative-animation document."""
    if not isinstance(document, dict):
        raise DeclarativeAnimationError("document root must be an object")
    version = document.get("schema_version", _DECLARATIVE_SCHEMA_VERSION)
    if version != _DECLARATIVE_SCHEMA_VERSION:
        raise DeclarativeAnimationError(
            f"schema_version {version} not supported; expected "
            f"{_DECLARATIVE_SCHEMA_VERSION}",
        )
    phases_raw = document.get("phases", [])
    if not isinstance(phases_raw, list):
        raise DeclarativeAnimationError("'phases' must be an array")
    bpm = float(document.get("bpm", 0.0))
    if bpm < 0.0:
        raise DeclarativeAnimationError(
            f"'bpm' must be non-negative, got {bpm}",
        )
    phases = tuple(_parse_phase(p, bpm) for p in phases_raw)
    if not phases:
        raise DeclarativeAnimationError("animation must have at least one phase")
    return DeclarativeAnimation(
        name=str(document.get("name", "unnamed")),
        loop_sec=float(document.get("loop_sec", sum(p.duration_sec for p in phases))),
        rig=_parse_rig(document.get("rig", {})),
        ground=_parse_ground(document.get("ground")),
        phases=phases,
        physics_chains=_parse_physics_chains(document.get("physics_chains", {})),
        wind=_parse_wind(document.get("wind")),
        bpm=bpm,
    )


def _parse_physics_chains(raw: Any) -> dict[str, dict[str, float]]:
    if not isinstance(raw, dict):
        raise DeclarativeAnimationError("'physics_chains' must be an object")
    out: dict[str, dict[str, float]] = {}
    for chain_name, params in raw.items():
        if not isinstance(params, dict):
            raise DeclarativeAnimationError(
                f"physics_chains[{chain_name!r}] must be an object",
            )
        out[str(chain_name)] = {k: _resolve_scalar(v) for k, v in params.items()}
    return out


def _parse_wind(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise DeclarativeAnimationError("'wind' must be an object")
    return raw


# --- Runtime ----------------------------------------------------------------


@dataclass
class _BoneDrive:
    node: Any
    rest_rotation: np.ndarray


@dataclass
class DeclarativeRuntime:
    """Per-frame runner for a parsed :class:`DeclarativeAnimation`."""

    animation: DeclarativeAnimation
    scene: Any
    time: Callable[[], float]
    floor_api: Any | None = None
    physics_lite: Any | None = None
    morph_api: Any | None = None
    _root_drive: _BoneDrive | None = None
    _bone_drives: dict[str, _BoneDrive] = field(default_factory=dict)
    _phase_starts: list[float] = field(default_factory=list)
    _last_step_idx: dict[str, int] = field(default_factory=dict)
    _foot_lock: dict[str, np.ndarray] = field(default_factory=dict)
    # Side → (release_target_world, frames_remaining). When the leading /
    # trailing parity flips at a step boundary, the previously-pinned
    # foot doesn't snap free immediately — it stays IK'd toward its old
    # world position for a few frames, decaying to no IK. This smooths
    # the visible "feet swap" jump where the trailing foot otherwise
    # teleports from its old stair to a rest pose under the freshly-
    # translated body.
    _foot_release: dict[str, tuple[np.ndarray, int]] = field(default_factory=dict)
    _leg_chain_nodes: dict[str, tuple[Any, Any, Any]] = field(default_factory=dict)

    # ----- Hook surface (matches ScriptHost expectations) -------------------
    def hooks(self) -> dict[str, Callable[..., Any]]:
        return {
            "start": self._start,
            "update": self._update,
            "on_event": self._on_event,
        }

    # ----- start ------------------------------------------------------------
    def _start(self) -> None:
        # Find the character root + cache its rest pose.
        root_name = self.animation.rig.character_root
        if root_name:
            root_node = self.scene.find(root_name)
            if root_node is not None:
                self._root_drive = _BoneDrive(
                    node=root_node,
                    rest_rotation=np.asarray(
                        root_node.transform.rotation, dtype=np.float32,
                    ).copy(),
                )
        # Cache rest rotations for any bones referenced by gait drivers OR
        # the per-phase ``bones`` block. Both paths use ``_set_bone`` which
        # composes a delta against the cached rest rotation.
        for phase in self.animation.phases:
            for bone_name in _phase_target_bones(phase, self.animation.rig):
                if bone_name in self._bone_drives:
                    continue
                node = self.scene.find(bone_name)
                if node is None:
                    continue
                self._bone_drives[bone_name] = _BoneDrive(
                    node=node,
                    rest_rotation=np.asarray(
                        node.transform.rotation, dtype=np.float32,
                    ).copy(),
                )
        # Build phase-start cumulative table once.
        cumulative = 0.0
        self._phase_starts = []
        for phase in self.animation.phases:
            self._phase_starts.append(cumulative)
            cumulative += phase.duration_sec
        # Engage foot planter + ground binding if both are described.
        if (
            self.floor_api is not None
            and self.animation.ground is not None
            and self.animation.rig.leg_chain_l
            and self.animation.rig.leg_chain_r
        ):
            self._bind_foot_planter()
        # Apply physics chain tunings + wind setup if declared.
        self._apply_physics_setup()
        # Cache leg-chain nodes for stride lock-target IK.
        for side, chain_names in (
            ("L", self.animation.rig.leg_chain_l),
            ("R", self.animation.rig.leg_chain_r),
        ):
            if chain_names is None:
                continue
            nodes = tuple(self.scene.find(n) for n in chain_names)
            if all(n is not None for n in nodes):
                self._leg_chain_nodes[side] = nodes

    def _apply_physics_setup(self) -> None:
        if self.physics_lite is None:
            return
        for chain_name, params in self.animation.physics_chains.items():
            chain = self.physics_lite.get_chain(chain_name)
            if chain is None:
                _log.debug("declarative: physics chain %r not found", chain_name)
                continue
            if "stiffness" in params:
                chain.stiffness = params["stiffness"]
            if "damping" in params:
                chain.damping = params["damping"]
            if "inertia" in params:
                chain.set_inertia(params["inertia"])
        if self.animation.wind is not None:
            wind = self.animation.wind
            self.physics_lite.add_wind(
                direction=vec3(*[
                    _resolve_scalar(c) for c in wind.get("direction", (1.0, 0.0, 0.0))
                ]),
                speed=_resolve_scalar(wind.get("speed", 0.0)),
                turbulence_amplitude=_resolve_scalar(wind.get("turbulence_amplitude", 0.0)),
                turbulence_frequency_hz=_resolve_scalar(wind.get("turbulence_frequency_hz", 0.0)),
            )

    def _bind_foot_planter(self) -> None:
        ground = self._build_ground_provider()
        if ground is None:
            return
        self.floor_api.clear()
        for chain_names in (self.animation.rig.leg_chain_l, self.animation.rig.leg_chain_r):
            nodes = tuple(self.scene.find(n) for n in chain_names)
            if any(n is None for n in nodes):
                _log.warning("declarative: leg chain bone missing — skipping foot bind")
                continue
            self.floor_api.bind_foot(
                nodes[0], nodes[1], nodes[2], ground,
                knee_limit_min=self.animation.rig.knee_limit_min,
                knee_limit_max=self.animation.rig.knee_limit_max,
            )

    def _build_ground_provider(self) -> Any | None:
        ground = self.animation.ground
        if ground is None:
            return None
        if ground.kind == "flat":
            return self.floor_api.flat(_resolve_scalar(ground.params.get("y", 0.0)))
        if ground.kind == "stairs":
            return self.floor_api.stairs(
                base_z=_resolve_scalar(ground.params.get("base_z", 0.0)),
                step_depth=_resolve_scalar(ground.params.get("step_depth", 0.0)),
                step_rise=_resolve_scalar(ground.params.get("step_rise", 0.0)),
                count=int(ground.params.get("count", 1)),
                base_y=_resolve_scalar(ground.params.get("base_y", 0.0)),
                forward_sign=int(ground.params.get("forward_sign", -1)),
            )
        return None

    # ----- update -----------------------------------------------------------
    def _update(self, _dt: float) -> None:
        elapsed = self.time() % self.animation.loop_sec
        phase_idx, phase_t, phase_elapsed = self._phase_index_at(elapsed)
        phase = self.animation.phases[phase_idx]
        scope = self._build_scope(elapsed, phase_t, phase_elapsed)
        output = self._compute_phase_output(phase, scope, phase_t)
        # Cross-fade with the next phase if both phases consent. The
        # actual overlap is the mutual minimum of the two consents so
        # neither phase blends further than it asked. Only happens at
        # the END of the current phase, so the boundary lands at "100%
        # next" with no jump as the runtime advances to the next phase.
        next_idx = phase_idx + 1
        if (
            next_idx < len(self.animation.phases)
            and phase.blend_out_sec > 0.0
        ):
            next_phase = self.animation.phases[next_idx]
            overlap = min(phase.blend_out_sec, next_phase.blend_in_sec)
            remaining = phase.duration_sec - phase_elapsed
            if overlap > 0.0 and 0.0 <= remaining < overlap:
                blend_t = 1.0 - remaining / overlap
                next_scope = self._build_scope(elapsed, 0.0, 0.0)
                next_output = self._compute_phase_output(
                    next_phase, next_scope, 0.0,
                )
                output = _blend_phase_outputs(output, next_output, blend_t)
        # Snapshot lock targets BEFORE the body translates this frame —
        # we want the trailing foot pinned to where it was at the end
        # of the previous frame (when it was actually on a stair),
        # not to its REST position under the freshly-translated body.
        if phase.gait is not None:
            self._maybe_refresh_lock_targets(phase.gait, phase_t)
        self._apply_root(output.translation, output.yaw, output.lean)
        # Reset stable bones (head / hip / chest / feet) to their rest
        # rotations before the gait runs. The previous frame's foot
        # planter or analytical IK can leave residual rotations on these
        # bones — without an explicit reset they accumulate and the
        # foot eventually points sideways or backward (the "腳反過來"
        # symptom). walk.py does the same by writing identity-deltas to
        # these bones every frame.
        self._reset_idle_bones()
        # Gait runs from the CURRENT phase only — blending two step-based
        # gaits is ill-defined, and authors typically want gait to be
        # continuous across short crossfades anyway.
        if phase.gait is not None:
            self._apply_gait(phase.gait, phase_elapsed, phase_t, output.yaw)
        # Bones override gait (applied after) — same rule as before, but
        # now sourced from the (possibly blended) computed output.
        for bone_key, body_delta in output.bones.items():
            self._set_bone(bone_key, _yaw_to_world(body_delta, output.yaw))
        if output.morphs and self.morph_api is not None:
            for name, weight in output.morphs.items():
                self.morph_api.set(str(name), float(weight))
        # Tell the engine foot planter which way the body is facing so
        # its post-tick toe-twist alignment doesn't fight the gait.
        if self.floor_api is not None:
            self.floor_api.set_body_forward(
                (math.sin(output.yaw), 0.0, math.cos(output.yaw)),
            )

    def _build_scope(
        self, elapsed: float, phase_t: float, phase_elapsed: float,
    ) -> dict[str, float]:
        bpm = self.animation.bpm
        beat = elapsed * bpm / _SECONDS_PER_MINUTE if bpm > 0.0 else 0.0
        phase_beat = (
            phase_elapsed * bpm / _SECONDS_PER_MINUTE if bpm > 0.0 else 0.0
        )
        return {
            "elapsed": float(elapsed),
            "phase_t": float(phase_t),
            "phase_elapsed": float(phase_elapsed),
            "beat": float(beat),
            "phase_beat": float(phase_beat),
        }

    def _compute_phase_output(
        self, phase: Phase, scope: dict[str, float], phase_t: float,
    ) -> PhaseOutput:
        """Evaluate body / bones / morphs curves into a :class:`PhaseOutput`.

        Pure (no scene writes); cheap to call twice per frame for the
        cross-fade path. Per-bone evaluation failures are logged and
        skipped rather than aborting the whole frame so a single bad
        expression doesn't freeze the timeline.
        """
        yaw = _resolve_value_curve(phase.body_yaw_rad, phase_t, scope)
        lean = _resolve_value_curve(phase.body_lean_x_rad, phase_t, scope)
        translation = _resolve_translation(phase.body_translation, phase_t, scope)
        output = PhaseOutput(yaw=yaw, lean=lean, translation=translation)
        for bone_key, axes in phase.bones.items():
            try:
                x = _resolve_value_curve(axes.get("x_rad", 0.0), phase_t, scope)
                y = _resolve_value_curve(axes.get("y_rad", 0.0), phase_t, scope)
                z = _resolve_value_curve(axes.get("z_rad", 0.0), phase_t, scope)
            except DeclarativeAnimationError as err:
                _log.warning(
                    "declarative: bones[%r] failed to evaluate: %s", bone_key, err,
                )
                continue
            output.bones[bone_key] = _euler_zyx_quat(x, y, z)
        for name, spec in phase.morphs.items():
            try:
                weight = _resolve_value_curve(spec, phase_t, scope)
            except DeclarativeAnimationError as err:
                _log.warning(
                    "declarative: morph %r failed to evaluate: %s", name, err,
                )
                continue
            output.morphs[str(name)] = float(weight)
        return output

    def _maybe_refresh_lock_targets(
        self, gait: dict[str, Any], phase_t: float,
    ) -> None:
        """Snapshot trailing-foot world position at each step boundary.

        Runs at the TOP of the frame, before ``_apply_root`` translates
        the body. Captures where the foot is sitting *now* (= end of
        previous frame, after that frame's IK / planter ran) so the
        lock target is on the stair surface, not on a rest-pose foot
        position floating ahead of the body.
        """
        if gait.get("kind") != "stride":
            self._foot_lock.clear()
            self._foot_release.clear()
            self._last_step_idx.clear()
            return
        step_count = int(gait.get("step_count", 1))
        if step_count <= 0:
            return
        step_pos = phase_t * step_count
        step_idx = min(int(step_pos), step_count - 1)
        leading_l = (step_idx % 2 == 0)
        self._refresh_lock_target(gait, step_idx, leading_l)

    def _reset_idle_bones(self) -> None:
        """Snap every cached bone back to its rest local rotation.

        Bones the gait writes to are overwritten immediately afterward
        — the reset is a no-op for them. Bones the gait does NOT write
        to (foot_L/R, head, hip, chest in the default vocabulary) get
        their previous-frame planter / IK rotations cleared so they
        track the lower-leg (and parent-chain yaw) cleanly.
        """
        for drive in self._bone_drives.values():
            drive.node.transform.set_rotation(drive.rest_rotation)

    def _phase_index_at(self, elapsed: float) -> tuple[int, float, float]:
        """Return ``(phase_index, phase_t, phase_elapsed)`` for ``elapsed``.

        The cross-fade path needs the index (so it can peek at the next
        phase), so the per-phase scan returns it directly. Falls back
        to the last phase at full progress when ``elapsed`` exceeds the
        cumulative duration — same behaviour as the previous
        ``_phase_for`` so existing tests' end-of-loop assertions hold.
        """
        for i, phase in enumerate(self.animation.phases):
            start = self._phase_starts[i]
            end = start + phase.duration_sec
            if elapsed < end:
                local = elapsed - start
                phase_t = local / phase.duration_sec if phase.duration_sec > 0 else 0.0
                return i, phase_t, local
        last = self.animation.phases[-1]
        return len(self.animation.phases) - 1, 1.0, last.duration_sec

    def _apply_root(
        self, translation: tuple[float, float, float], yaw: float, lean: float,
    ) -> None:
        if self._root_drive is None:
            return
        yaw_q = quat_axis_angle(vec3(0.0, 1.0, 0.0), yaw)
        composed = quat_mul(yaw_q, self._root_drive.rest_rotation)
        if abs(lean) > 0.0:
            lean_q = quat_axis_angle(vec3(1.0, 0.0, 0.0), lean)
            composed = quat_mul(lean_q, composed)
        self._root_drive.node.transform.set_rotation(composed)
        self._root_drive.node.transform.set_translation(
            vec3(*translation),
        )

    def _apply_gait(
        self, gait: dict[str, Any], phase_elapsed: float, phase_t: float, yaw: float,
    ) -> None:
        kind = gait.get("kind", "")
        if kind == "walking":
            self._apply_walking_gait(gait, phase_elapsed, yaw)
        elif kind == "stride":
            self._apply_stride_gait(gait, phase_t, yaw)
        else:
            _log.debug("declarative: unknown gait kind %r", kind)

    def _refresh_lock_target(
        self, gait: dict[str, Any], step_idx: int, leading_l: bool,
    ) -> None:
        """Snapshot the trailing foot's world position at each step boundary.

        ``stride`` gaits naturally pin the leg that's NOT swinging — the
        trailing foot stays planted on its stair while the leading leg
        moves. Snapshotting at each ``step_idx`` boundary gives us a
        per-stride world target the IK pass can then drive the trailing
        leg toward, on top of the engine's own collision planter.
        Without this lock the trailing foot drifts as the body translates
        and the engine planter is the only thing preventing visible
        sliding — fine for clip-prevention, less natural-looking.
        """
        if not gait.get("lock_trailing_foot", True):
            return
        # Detect step boundary on a per-side basis so each foot's lock
        # is refreshed only when IT is the trailing one.
        side = "R" if leading_l else "L"
        last = self._last_step_idx.get(side, -1)
        if last == step_idx:
            return
        self._last_step_idx[side] = step_idx
        chain = self._leg_chain_nodes.get(side)
        if chain is None:
            return
        foot = chain[2]
        self._foot_lock[side] = _world_position(foot)
        # The OTHER side is leading now. Don't pop its lock outright —
        # park it in the release queue so its IK decays over a few
        # frames instead of snapping the foot from its old stair to
        # the rest pose under the new body position.
        opposite = "L" if side == "R" else "R"
        old_lock = self._foot_lock.pop(opposite, None)
        if old_lock is not None:
            self._foot_release[opposite] = (old_lock, _LOCK_RELEASE_FRAMES)
        # Fresh hard lock on the new trailing side overrides any
        # in-flight release on the same side.
        self._foot_release.pop(side, None)

    def _apply_lock_targets(self) -> None:
        """Run CCD 2-bone IK on each foot with an active lock or release.

        Uses CCD (with the rig's knee hinge limits) instead of the
        analytical solve so the knee bends on its anatomical hinge axis
        — the analytical solver picks an arbitrary perpendicular axis
        when the current knee is colinear with hip→target, which can
        twist the knee sideways or backward and produce the "腳反過來"
        symptom on a stair stride. Mirrors walk.py's lock-target IK.

        Hard locks (``_foot_lock``) pull the trailing foot fully toward
        its frozen world position. Soft releases (``_foot_release``)
        carry over a decaying number of frames — the foot blends from
        its old lock toward the natural rest position, so the leading
        leg doesn't snap off the previous stair the instant parity
        flips at a step boundary.
        """
        if not self._leg_chain_nodes:
            return
        if not self._foot_lock and not self._foot_release:
            return
        try:
            from posecascade.animation.ik import (  # noqa: PLC0415
                solve_two_bone,
            )
        except ImportError:
            return
        knee_min = self.animation.rig.knee_limit_min
        knee_max = self.animation.rig.knee_limit_max
        for side, target_world in list(self._foot_lock.items()):
            chain = self._leg_chain_nodes.get(side)
            if chain is None:
                continue
            root, mid, end = chain
            solve_two_bone(
                root, mid, end, target_world.astype(np.float32),
                iterations=8,
                step_radian=0.6,
                mid_limit_min=knee_min,
                mid_limit_max=knee_max,
            )
        for side, (old_target, frames_left) in list(self._foot_release.items()):
            chain = self._leg_chain_nodes.get(side)
            if chain is None:
                self._foot_release.pop(side, None)
                continue
            root, mid, end = chain
            current = _world_position(end)
            # Blend toward the natural (no-IK) foot position over the
            # release window. weight=1 at start, 0 at end.
            weight = float(frames_left) / float(_LOCK_RELEASE_FRAMES)
            blended = old_target * weight + current * (1.0 - weight)
            solve_two_bone(
                root, mid, end, blended.astype(np.float32),
                iterations=8,
                step_radian=0.6,
                mid_limit_min=knee_min,
                mid_limit_max=knee_max,
            )
            new_frames = frames_left - 1
            if new_frames <= 0:
                self._foot_release.pop(side, None)
            else:
                self._foot_release[side] = (old_target, new_frames)

    def _apply_stride_gait(
        self, gait: dict[str, Any], phase_t: float, yaw: float,
    ) -> None:
        """Step-based stair stride: leading leg lifts forward, trailing back-pedals.

        Maps phase_t into ``step_count`` discrete strides; within each stride
        the gait blends through bell envelopes to drive upper-leg / knee /
        arm angles. Cross-body coordination flips per stride (parity of
        step index). Body-frame deltas are conjugated by ``yaw`` so the
        sign convention stays consistent across yaw-flipped phases (e.g.
        the same ``leading_lift_rad`` value reads as "body-forward" both
        when facing -Z and when facing +Z).
        """
        step_count = int(gait.get("step_count", 1))
        if step_count <= 0:
            return
        leading_lift = _resolve_scalar(gait.get("leading_lift_rad", 0.0))
        trailing_back = _resolve_scalar(gait.get("trailing_back_rad", 0.0))
        knee_bend = _resolve_scalar(gait.get("knee_bend_rad", 0.0))
        arm_amp = _resolve_scalar(gait.get("arm_swing_amplitude_rad", 0.0))
        arm_hang = _resolve_scalar(
            gait.get("arm_hang_rad", _DEFAULT_ARM_HANG_RAD),
        )
        knee_bell = gait.get("knee_bell", [0.10, 0.65])
        forward_bell = gait.get("forward_bell", [0.10, 0.65])
        step_pos = phase_t * step_count
        step_idx = min(int(step_pos), step_count - 1)
        step_t = step_pos - step_idx
        leading_l = (step_idx % 2 == 0)
        # Lock-target refresh happens earlier in ``_update`` (before
        # ``_apply_root`` translates the body) — see
        # ``_maybe_refresh_lock_targets``.
        knee_env = _bell(step_t, float(knee_bell[0]), float(knee_bell[1]))
        forward_env = _bell(step_t, float(forward_bell[0]), float(forward_bell[1]))
        leading_upper = leading_lift * forward_env
        trailing_upper = trailing_back * forward_env
        leading_knee = knee_bend * knee_env
        trailing_knee = 0.0
        if leading_l:
            upper_l, upper_r = leading_upper, trailing_upper
            knee_l, knee_r = leading_knee, trailing_knee
        else:
            upper_l, upper_r = trailing_upper, leading_upper
            knee_l, knee_r = trailing_knee, leading_knee
        arm_swing = (+arm_amp if leading_l else -arm_amp) * forward_env
        self._set_body_x_delta("upper_leg_L", upper_l, yaw)
        self._set_body_x_delta("upper_leg_R", upper_r, yaw)
        self._set_body_x_delta("lower_leg_L", knee_l, yaw)
        self._set_body_x_delta("lower_leg_R", knee_r, yaw)
        self._set_arm("upper_arm_L", arm_swing, +1, arm_hang, yaw)
        self._set_arm("upper_arm_R", arm_swing, -1, arm_hang, yaw)
        # AFTER gait writes the leg poses, drag the trailing foot back
        # to its locked world position via analytical IK. Engine's
        # foot_planter still runs (post-script) as the safety net.
        self._apply_lock_targets()

    def _apply_walking_gait(
        self, gait: dict[str, Any], phase_elapsed: float, yaw: float,
    ) -> None:
        cycle_sec = float(gait.get("step_cycle_sec", 1.0))
        if cycle_sec <= 0.0:
            return
        leg_amp = _resolve_scalar(gait.get("leg_swing_amplitude", 0.0))
        knee_bend = _resolve_scalar(gait.get("knee_bend", 0.0))
        arm_amp = _resolve_scalar(gait.get("arm_swing_amplitude", 0.0))
        arm_hang = _resolve_scalar(
            gait.get("arm_hang_rad", _DEFAULT_ARM_HANG_RAD),
        )
        phase = phase_elapsed * _TAU / cycle_sec
        cycle = math.sin(phase)
        cos_phase = math.cos(phase)
        knee_l = knee_bend * _HALF * (1.0 - cos_phase)
        knee_r = knee_bend * _HALF * (1.0 + cos_phase)
        leg_l = leg_amp * cycle
        leg_r = -leg_amp * cycle
        # Cross-body: when L leg is body-back (cycle=+1), L arm swings
        # body-forward — that's NEGATIVE swing on the post-hang arm.
        # The L/R sign is handled by ``_set_arm``'s ``side`` argument.
        arm_swing = -arm_amp * cycle
        self._set_body_x_delta("upper_leg_L", leg_l, yaw)
        self._set_body_x_delta("upper_leg_R", leg_r, yaw)
        self._set_body_x_delta("lower_leg_L", knee_l, yaw)
        self._set_body_x_delta("lower_leg_R", knee_r, yaw)
        self._set_arm("upper_arm_L", arm_swing, +1, arm_hang, yaw)
        self._set_arm("upper_arm_R", arm_swing, -1, arm_hang, yaw)

    def _set_body_x_delta(self, bone_key: str, angle: float, yaw: float) -> None:
        body_delta = quat_axis_angle(vec3(1.0, 0.0, 0.0), angle)
        self._set_bone(bone_key, _yaw_to_world(body_delta, yaw))

    def _set_arm(
        self,
        bone_key: str,
        swing: float,
        side: int,
        arm_hang: float,
        yaw: float,
    ) -> None:
        """Compose the T-pose→hang Z-tuck with a forward/back X swing.

        Galaxia (and most VRoid rigs) rest in T-pose along world ±X. The
        Z-tuck rotates each arm down to a vertical hang at -Y, after
        which a rotation around X is the natural pendulum swing. ``side``
        flips the Z and X signs so the L/R arms mirror without needing
        per-side amplitudes. The composed body-frame delta is then
        yaw-conjugated to world before going through ``_set_bone``.
        """
        body_delta = quat_mul(
            quat_axis_angle(vec3(1.0, 0.0, 0.0), side * swing),
            quat_axis_angle(vec3(0.0, 0.0, 1.0), side * arm_hang),
        )
        self._set_bone(bone_key, _yaw_to_world(body_delta, yaw))

    def _set_bone(self, bone_key: str, delta_world: np.ndarray) -> None:
        """Apply ``delta_world`` to the named bone, conjugated to parent-local.

        World-space deltas don't compose directly with a bone whose parent
        chain is itself rotated — the same world delta would mean different
        body-frame motion depending on the parent's accumulated rotation.
        Walking up the parent chain to compute the parent's WORLD rotation
        and conjugating into parent-local frame keeps the runtime's "+x =
        world-forward" sign convention robust against rigs whose parents
        carry baked rest rotations (VRoid hip 180°-Z, root Y-up→Z-up).
        """
        bone_name = self.animation.rig.body_bones.get(bone_key, bone_key)
        drive = self._bone_drives.get(bone_name)
        if drive is None:
            return
        parent_world = quat_axis_angle(vec3(1.0, 0.0, 0.0), 0.0)
        chain: list[np.ndarray] = []
        cur = drive.node.parent
        while cur is not None:
            chain.append(cur.transform.rotation)
            cur = cur.parent
        for r in reversed(chain):
            parent_world = quat_mul(parent_world, r)
        parent_world_inv = quat_inverse(parent_world)
        delta_local = quat_mul(
            quat_mul(parent_world_inv, delta_world),
            parent_world,
        )
        drive.node.transform.set_rotation(
            quat_mul(delta_local, drive.rest_rotation).astype(np.float32, copy=False),
        )

    # ----- on_event ---------------------------------------------------------
    def _on_event(self, name: str, _payload: object) -> None:
        if name != "reset":
            return
        if self.physics_lite is not None:
            for chain in self.physics_lite.chains():
                chain.reset()


def _gait_target_bones(gait: dict[str, Any], rig: RigBindings) -> tuple[str, ...]:
    """List bone names the runtime caches rest rotations for.

    Includes both the bones the gait actively drives (legs / arms) AND
    the "stable" bones that get reset to rest each frame (feet / head /
    hip / chest) so previous-frame IK / planter rotations don't leak
    into the next frame.
    """
    bones: set[str] = set()
    kind = gait.get("kind", "")
    if kind in ("walking", "stride"):
        for key in (
            "upper_leg_L", "upper_leg_R",
            "lower_leg_L", "lower_leg_R",
            "upper_arm_L", "upper_arm_R",
            "foot_L", "foot_R",
            "head", "hip", "chest",
        ):
            bones.add(rig.body_bones.get(key, key))
    return tuple(bones)


def _phase_target_bones(phase: Phase, rig: RigBindings) -> tuple[str, ...]:
    """All bone names a phase needs rest rotations cached for.

    Union of the gait's reset-set and any bone keyed in ``phase.bones``.
    Both go through the rig's alias map so authors can rename bones in
    one place without touching every phase.
    """
    names: set[str] = set()
    if phase.gait is not None:
        names.update(_gait_target_bones(phase.gait, rig))
    for bone_key in phase.bones:
        names.add(rig.body_bones.get(bone_key, bone_key))
    return tuple(names)


def _euler_zyx_quat(x: float, y: float, z: float) -> np.ndarray:
    """Compose ``Rz · Ry · Rx`` as a single quaternion.

    Extrinsic XYZ Euler order (== intrinsic ZYX). Matches the natural
    "first pitch, then yaw, then roll in the same fixed frame" mental
    model authors use when describing a pose. Identity if all three
    angles are zero (no allocation cost difference — quat_axis_angle
    handles the 0-angle case cleanly).
    """
    qx = quat_axis_angle(vec3(1.0, 0.0, 0.0), x)
    qy = quat_axis_angle(vec3(0.0, 1.0, 0.0), y)
    qz = quat_axis_angle(vec3(0.0, 0.0, 1.0), z)
    return quat_mul(qz, quat_mul(qy, qx))


def _bell(t: float, start: float, end: float) -> float:
    """Half-sine bell — 0 outside ``[start, end]``, 1 at the midpoint."""
    if t <= start or t >= end or end <= start:
        return 0.0
    return math.sin(((t - start) / (end - start)) * math.pi)


def _yaw_to_world(body_delta: np.ndarray, yaw: float) -> np.ndarray:
    """Conjugate a body-frame rotation by ``yaw`` to get a world delta.

    When the character root is rotated by ``yaw`` around world Y, a
    bone delta authored in body-frame coordinates does not directly
    apply as a world delta — Y-axis rotations commute (no change),
    but X- and Z-axis rotations flip sign under yaw=π. Conjugating
    keeps the runtime's "+X = body-forward" sign convention robust
    across yaw-flipped phases.
    """
    if abs(yaw) < _YAW_NEGLIGIBLE:
        return body_delta
    yaw_q = quat_axis_angle(vec3(0.0, 1.0, 0.0), yaw)
    yaw_inv = quat_inverse(yaw_q)
    return quat_mul(yaw_q, quat_mul(body_delta, yaw_inv))


def _world_position(node: Any) -> np.ndarray:
    """Compose ``node``'s parent chain to get its world-space origin."""
    matrix = node.transform.to_matrix()
    parent = node.parent
    while parent is not None:
        matrix = parent.transform.to_matrix() @ matrix
        parent = parent.parent
    return np.array([matrix[0, 3], matrix[1, 3], matrix[2, 3]], dtype=np.float64)


def _resolve_translation(
    spec: dict[str, Any], phase_t: float, scope: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    if "stair" in spec:
        return _resolve_stair_translation(spec["stair"], phase_t)
    return (
        _resolve_value_curve(spec.get("x", 0.0), phase_t, scope),
        _resolve_value_curve(spec.get("y", 0.0), phase_t, scope),
        _resolve_value_curve(spec.get("z", 0.0), phase_t, scope),
    )


def _resolve_stair_translation(
    spec: dict[str, Any], phase_t: float,
) -> tuple[float, float, float]:
    """Body Y / Z trajectory for stair traversal.

    Spec fields:
    - ``base_z``: world Z of the staircase's front edge (entrance).
    - ``rise``: total height climbed (Y).
    - ``forward``: total horizontal travel (along Z).
    - ``step_count``: number of steps spanned.
    - ``ascending``: ``True`` for going up (low → high), False for descending.
    - ``rise_window``: ``[t0, t1]`` within each per-stride normalised time
      where the body's Y eases from base step height to next step height.
      Outside this window the body sits on the current step's height.
    - ``forward_sign``: sign convention matching the ground provider's.
    """
    base_z = _resolve_scalar(spec.get("base_z", 0.0))
    rise = _resolve_scalar(spec.get("rise", 0.0))
    forward = _resolve_scalar(spec.get("forward", 0.0))
    step_count = max(1, int(spec.get("step_count", 1)))
    ascending = bool(spec.get("ascending", True))
    rise_window = spec.get("rise_window", [0.65, 0.95])
    forward_sign = int(spec.get("forward_sign", -1))
    rise_per_step = rise / step_count
    z_offset = forward * float(phase_t)
    if ascending:
        body_z = base_z + forward_sign * z_offset
    else:
        body_z = base_z + forward_sign * (forward - z_offset)
    step_pos = phase_t * step_count
    step_idx = min(int(step_pos), step_count - 1)
    step_t = step_pos - step_idx
    if ascending:
        base_y = step_idx * rise_per_step
        next_y = (step_idx + 1) * rise_per_step
    else:
        base_y = (step_count - step_idx) * rise_per_step
        next_y = (step_count - step_idx - 1) * rise_per_step
    rise_start = float(rise_window[0])
    rise_end = float(rise_window[1])
    if step_t < rise_start:
        body_y = base_y
    elif step_t < rise_end:
        progress = (step_t - rise_start) / (rise_end - rise_start)
        eased = _HALF - _HALF * math.cos(progress * math.pi)
        body_y = base_y + (next_y - base_y) * eased
    else:
        body_y = next_y
    return (0.0, float(body_y), float(body_z))


# --- Loader (matches sandbox.load_script's signature shape) -----------------


def load_animation(
    source: str,
    filename: str,
    api: dict[str, Any],
) -> dict[str, Callable[..., Any]]:
    """Compile a JSON document into ScriptHost-compatible hooks.

    Mirrors :func:`posecascade.scripting.sandbox.load_script` so the
    ScriptHost can attach declarative animations the same way it
    attaches Python scripts.
    """
    if not isinstance(source, str):
        raise ScriptSecurityError("animation source must be str")
    try:
        document = json.loads(source)
    except json.JSONDecodeError as err:
        raise DeclarativeAnimationError(
            f"failed to parse {filename}: {err.msg} at line {err.lineno}",
        ) from err
    parsed = parse_animation(document)
    runtime = DeclarativeRuntime(
        animation=parsed,
        scene=api["scene"],
        time=api["time"],
        floor_api=api.get("floor"),
        physics_lite=api.get("physics_lite"),
        morph_api=api.get("morphs"),
    )
    return runtime.hooks()


__all__ = [
    "DeclarativeAnimation",
    "DeclarativeAnimationError",
    "DeclarativeRuntime",
    "GroundSpec",
    "Phase",
    "RigBindings",
    "load_animation",
    "parse_animation",
]
