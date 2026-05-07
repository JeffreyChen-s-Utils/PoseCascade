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

Stage 1 (this module): phase timing, body trajectory, walking gait,
stair_ground binding, foot planter auto-binding. Stride / lock-target
gait, physics chain tunings, and morph timeline are stage 2 — declared
in schema for forward-compat but currently parsed-and-ignored with a
warning.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from posecascade.errors import PoseCascadeError, ScriptSecurityError
from posecascade.utils.logging import get_logger
from posecascade.utils.math3d import (
    quat_from_axis_angle,
    quat_mul,
    vec3,
)

quat_axis_angle = quat_from_axis_angle

_log = get_logger(__name__)

_TAU = math.tau
_TWO = 2.0
_HALF = 0.5
_DECLARATIVE_SCHEMA_VERSION = 1
_VEC3_LEN = 3


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


def _resolve_scalar(value: Any) -> float:
    """Resolve a scalar that may be a number or a symbolic constant string."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value in _SYMBOLIC_FLOATS:
            return _SYMBOLIC_FLOATS[value]
        try:
            return float(value)
        except ValueError as err:
            raise DeclarativeAnimationError(
                f"unrecognised scalar string {value!r}; expected a number "
                f"or one of {sorted(_SYMBOLIC_FLOATS)}",
            ) from err
    raise DeclarativeAnimationError(
        f"expected scalar, got {type(value).__name__}: {value!r}",
    )


def _resolve_value_curve(spec: Any, phase_t: float) -> float:
    """Evaluate a per-phase value at normalised phase time ``phase_t`` ∈ [0,1].

    ``spec`` is either a scalar (constant) or a dict with a ``kind`` field
    naming one of the supported curves: ``constant``, ``linear``, ``ease``.
    """
    if isinstance(spec, (int, float, str)):
        return _resolve_scalar(spec)
    if not isinstance(spec, dict):
        raise DeclarativeAnimationError(
            f"value curve must be scalar or dict, got {type(spec).__name__}",
        )
    kind = spec.get("kind", "constant")
    if kind == "constant":
        return _resolve_scalar(spec.get("value", 0.0))
    if kind == "linear":
        a = _resolve_scalar(spec.get("from", 0.0))
        b = _resolve_scalar(spec.get("to", 0.0))
        return a + (b - a) * float(phase_t)
    if kind == "ease":
        a = _resolve_scalar(spec.get("from", 0.0))
        b = _resolve_scalar(spec.get("to", 0.0))
        t = float(phase_t)
        eased = _HALF - _HALF * math.cos(t * math.pi)
        return a + (b - a) * eased
    raise DeclarativeAnimationError(
        f"unknown value-curve kind {kind!r}; expected one of "
        "constant / linear / ease",
    )


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


@dataclass(frozen=True)
class DeclarativeAnimation:
    name: str
    loop_sec: float
    rig: RigBindings
    ground: GroundSpec | None
    phases: tuple[Phase, ...]
    physics_chains: dict[str, dict[str, float]]
    wind: dict[str, Any] | None


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


def _parse_phase(raw: dict[str, Any]) -> Phase:
    if not isinstance(raw, dict):
        raise DeclarativeAnimationError("each phase must be an object")
    body = raw.get("body", {})
    if not isinstance(body, dict):
        raise DeclarativeAnimationError("phase 'body' must be an object")
    return Phase(
        name=str(raw.get("name", "")),
        duration_sec=float(raw.get("duration_sec", 0.0)),
        body_yaw_rad=body.get("yaw_rad", 0.0),
        body_lean_x_rad=body.get("lean_x_rad", 0.0),
        body_translation=body.get("translation", {}),
        gait=raw.get("gait"),
    )


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
    phases = tuple(_parse_phase(p) for p in phases_raw)
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
    _root_drive: _BoneDrive | None = None
    _bone_drives: dict[str, _BoneDrive] = field(default_factory=dict)
    _phase_starts: list[float] = field(default_factory=list)

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
        # Cache rest rotations for any bones referenced by gait drivers.
        for phase in self.animation.phases:
            if phase.gait is None:
                continue
            for bone_name in _gait_target_bones(phase.gait, self.animation.rig):
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
        phase, phase_t, phase_elapsed = self._phase_for(elapsed)
        yaw = _resolve_value_curve(phase.body_yaw_rad, phase_t)
        lean = _resolve_value_curve(phase.body_lean_x_rad, phase_t)
        translation = _resolve_translation(phase.body_translation, phase_t)
        self._apply_root(translation, yaw, lean)
        if phase.gait is not None:
            self._apply_gait(phase.gait, phase_elapsed, phase_t)

    def _phase_for(self, elapsed: float) -> tuple[Phase, float, float]:
        for i, phase in enumerate(self.animation.phases):
            start = self._phase_starts[i]
            end = start + phase.duration_sec
            if elapsed < end:
                local = elapsed - start
                phase_t = local / phase.duration_sec if phase.duration_sec > 0 else 0.0
                return phase, phase_t, local
        last = self.animation.phases[-1]
        return last, 1.0, last.duration_sec

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

    def _apply_gait(self, gait: dict[str, Any], phase_elapsed: float, phase_t: float) -> None:
        kind = gait.get("kind", "")
        if kind == "walking":
            self._apply_walking_gait(gait, phase_elapsed)
        elif kind == "stride":
            self._apply_stride_gait(gait, phase_t)
        else:
            _log.debug("declarative: unknown gait kind %r", kind)

    def _apply_stride_gait(self, gait: dict[str, Any], phase_t: float) -> None:
        """Step-based stair stride: leading leg lifts forward, trailing back-pedals.

        Maps phase_t into ``step_count`` discrete strides; within each stride
        the gait blends through bell envelopes to drive upper-leg / knee /
        arm angles. Cross-body coordination flips per stride (parity of
        step index).
        """
        step_count = int(gait.get("step_count", 1))
        if step_count <= 0:
            return
        leading_lift = _resolve_scalar(gait.get("leading_lift_rad", 0.0))
        trailing_back = _resolve_scalar(gait.get("trailing_back_rad", 0.0))
        knee_bend = _resolve_scalar(gait.get("knee_bend_rad", 0.0))
        arm_amp = _resolve_scalar(gait.get("arm_swing_amplitude_rad", 0.0))
        knee_bell = gait.get("knee_bell", [0.10, 0.65])
        forward_bell = gait.get("forward_bell", [0.10, 0.65])
        step_pos = phase_t * step_count
        step_idx = min(int(step_pos), step_count - 1)
        step_t = step_pos - step_idx
        leading_l = (step_idx % 2 == 0)
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
        self._set_world_x_delta("upper_leg_L", upper_l)
        self._set_world_x_delta("upper_leg_R", upper_r)
        self._set_world_x_delta("lower_leg_L", knee_l)
        self._set_world_x_delta("lower_leg_R", knee_r)
        self._set_world_y_delta("upper_arm_L", arm_swing)
        self._set_world_y_delta("upper_arm_R", arm_swing)

    def _apply_walking_gait(self, gait: dict[str, Any], phase_elapsed: float) -> None:
        cycle_sec = float(gait.get("step_cycle_sec", 1.0))
        if cycle_sec <= 0.0:
            return
        leg_amp = _resolve_scalar(gait.get("leg_swing_amplitude", 0.0))
        knee_bend = _resolve_scalar(gait.get("knee_bend", 0.0))
        arm_amp = _resolve_scalar(gait.get("arm_swing_amplitude", 0.0))
        phase = phase_elapsed * _TAU / cycle_sec
        cycle = math.sin(phase)
        cos_phase = math.cos(phase)
        knee_l = knee_bend * _HALF * (1.0 - cos_phase)
        knee_r = knee_bend * _HALF * (1.0 + cos_phase)
        leg_l = leg_amp * cycle
        leg_r = -leg_amp * cycle
        arm_swing = arm_amp * cycle
        self._set_world_x_delta("upper_leg_L", leg_l)
        self._set_world_x_delta("upper_leg_R", leg_r)
        self._set_world_x_delta("lower_leg_L", knee_l)
        self._set_world_x_delta("lower_leg_R", knee_r)
        self._set_world_y_delta("upper_arm_L", arm_swing)
        self._set_world_y_delta("upper_arm_R", arm_swing)

    def _set_world_x_delta(self, bone_key: str, angle: float) -> None:
        delta = quat_axis_angle(vec3(1.0, 0.0, 0.0), angle)
        self._set_bone(bone_key, delta)

    def _set_world_y_delta(self, bone_key: str, angle: float) -> None:
        delta = quat_axis_angle(vec3(0.0, 1.0, 0.0), angle)
        self._set_bone(bone_key, delta)

    def _set_bone(self, bone_key: str, delta_world: np.ndarray) -> None:
        bone_name = self.animation.rig.body_bones.get(bone_key, bone_key)
        drive = self._bone_drives.get(bone_name)
        if drive is None:
            return
        drive.node.transform.set_rotation(
            quat_mul(delta_world, drive.rest_rotation).astype(np.float32, copy=False),
        )

    # ----- on_event ---------------------------------------------------------
    def _on_event(self, name: str, _payload: object) -> None:
        if name != "reset":
            return
        if self.physics_lite is not None:
            for chain in self.physics_lite.chains():
                chain.reset()


def _gait_target_bones(gait: dict[str, Any], rig: RigBindings) -> tuple[str, ...]:
    """List bone names that a gait's drivers will write to."""
    bones: set[str] = set()
    kind = gait.get("kind", "")
    if kind in ("walking", "stride"):
        # Default leg / arm names — overridable via rig.body_bones[bone_key].
        for key in (
            "upper_leg_L", "upper_leg_R",
            "lower_leg_L", "lower_leg_R",
            "upper_arm_L", "upper_arm_R",
        ):
            bones.add(rig.body_bones.get(key, key))
    return tuple(bones)


def _bell(t: float, start: float, end: float) -> float:
    """Half-sine bell — 0 outside ``[start, end]``, 1 at the midpoint."""
    if t <= start or t >= end or end <= start:
        return 0.0
    return math.sin(((t - start) / (end - start)) * math.pi)


def _resolve_translation(
    spec: dict[str, Any], phase_t: float,
) -> tuple[float, float, float]:
    if "stair" in spec:
        return _resolve_stair_translation(spec["stair"], phase_t)
    return (
        _resolve_value_curve(spec.get("x", 0.0), phase_t),
        _resolve_value_curve(spec.get("y", 0.0), phase_t),
        _resolve_value_curve(spec.get("z", 0.0), phase_t),
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
