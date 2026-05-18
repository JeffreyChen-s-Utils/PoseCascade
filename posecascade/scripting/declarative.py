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
from pathlib import Path
from typing import Any

import numpy as np

from posecascade.assets.path_safety import resolve_safe
from posecascade.errors import PoseCascadeError, ScriptSecurityError, UnsafePathError
from posecascade.scripting.expressions import (
    ExpressionError,
    evaluate_expression,
    looks_like_expression,
)
from posecascade.scripting.hand_library import (
    merge_libraries as merge_hand_libraries,
)
from posecascade.scripting.pose_library import (
    PoseSpec,
)
from posecascade.scripting.pose_library import (
    merge_libraries as merge_pose_libraries,
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
# Quaternion components below this magnitude count as "no rotation".
# Used to skip applying a bone delta that's effectively identity —
# avoids overriding the gait's arm_hang back to T-pose when a pose
# preset's weight curve is ramping through zero. 1e-3 ≈ 0.06° error,
# below visible threshold.
_IDENTITY_QUAT_TOL = 1e-3
# Shared (x, y, z, w) identity quaternion — handed back from the
# per-frame parent-world cache for orphan / root nodes, and a starting
# point for chain composition. Never mutated. The component order
# matches :func:`posecascade.utils.math3d.quat_mul`.
_IDENTITY_QUAT: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _is_identity_quat(q: np.ndarray, tol: float = _IDENTITY_QUAT_TOL) -> bool:
    """True if ``q`` is effectively the identity rotation.

    Identity is ``(0, 0, 0, ±1)``. The W sign is canonicalised by
    taking ``abs`` since both ``q`` and ``-q`` represent the same
    rotation.
    """
    return (
        abs(float(q[0])) < tol
        and abs(float(q[1])) < tol
        and abs(float(q[2])) < tol
        and abs(abs(float(q[3])) - 1.0) < tol
    )
# Default Z-tuck applied to arms when the gait doesn't override it. The
# old value of -1.45 rad (~83°) hung the arms vertically against the
# torso — fine on a stick-figure rig, but on most stylised characters
# the wide sleeve cuffs / dress flares at hip level got intersected by
# the wrist. -0.50 rad (~29°) keeps the arms in a gentle A-pose with
# the hands clearly OUTSIDE the body silhouette; matches the
# ``rest_arms`` pose magnitude in :mod:`posecascade.scripting.pose_library`
# so swapping between gait-driven and pose-driven arm states reads
# continuously.
_DEFAULT_ARM_HANG_RAD = -0.50
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
    # ``_build_scope`` already seeds ``phase_t`` and the handlers
    # treat scope as read-only, so we skip the defensive ``dict(scope)``
    # copy every curve evaluation used to pay. The ``scope or {}`` keeps
    # the API compatible with callers passing ``None`` (parse-time
    # validation paths) without re-allocating an empty dict per call.
    eval_scope = scope if scope is not None else _EMPTY_SCOPE
    if isinstance(spec, (int, float, str)):
        return _resolve_scalar(spec, eval_scope)
    # Two-element ``[from, to]`` arrays are author-friendly shorthand for a
    # linear curve — by far the most common parametric shape. The dict
    # form ``{"kind": "linear", "from": A, "to": B}`` keeps working.
    if isinstance(spec, list) and len(spec) == _SHORTHAND_LINEAR_LEN:
        a = _resolve_scalar(spec[0], eval_scope)
        b = _resolve_scalar(spec[1], eval_scope)
        return a + (b - a) * float(phase_t)
    if not isinstance(spec, dict):
        raise DeclarativeAnimationError(
            f"value curve must be scalar, [from, to], or dict — "
            f"got {type(spec).__name__}",
        )
    kind = spec.get("kind", "constant")
    handler = _CURVE_HANDLERS.get(kind)
    if handler is None:
        raise DeclarativeAnimationError(
            f"unknown value-curve kind {kind!r}; expected one of "
            f"{sorted(_CURVE_HANDLERS)}",
        )
    return handler(spec, eval_scope, float(phase_t))


_SHORTHAND_LINEAR_LEN = 2


# Sentinel used as the read-only "no scope" value for parse-time
# validation calls into :func:`_resolve_value_curve`. The handlers only
# read from scope; sharing one frozen dict avoids the per-call ``{}``
# allocation when no caller scope is provided.
_EMPTY_SCOPE: dict[str, float] = {}


# --- Schema parsing ---------------------------------------------------------


@dataclass(frozen=True)
class RigBindings:
    character_root: str
    leg_chain_l: tuple[str, str, str] | None
    leg_chain_r: tuple[str, str, str] | None
    knee_limit_min: tuple[float, float, float] | None
    knee_limit_max: tuple[float, float, float] | None
    body_bones: dict[str, str]
    # Three-bone chains for hand IK: shoulder → elbow → wrist. When set
    # AND a phase declares ``ik.hand_l_target`` / ``ik.hand_r_target``,
    # the runtime drives the arm to put the wrist at the target world
    # position each frame. ``None`` (the default) disables hand IK.
    # Mirrors ``leg_chain_l/r``; bone names go through the alias map.
    arm_chain_l: tuple[str, str, str] | None = None
    arm_chain_r: tuple[str, str, str] | None = None
    # Optional XYZ-Euler clamps on the elbow's local rotation so the
    # IK solver doesn't bend the arm sideways. Same shape +
    # interpretation as ``knee_limit_min/max``. ``None`` = no clamp.
    elbow_limit_min: tuple[float, float, float] | None = None
    elbow_limit_max: tuple[float, float, float] | None = None


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
    body_translation: Any  # dict[str, Any] | list[Any]; shorthand for [x, y, z]
    gait: dict[str, Any] | None
    morphs: dict[str, Any]  # name → value-curve spec
    # bone_key → {"x_rad"?: curve, "y_rad"?: curve, "z_rad"?: curve}.
    # Composed AFTER gait so authors can override a gait-driven bone with
    # a custom curve (e.g. hold the arm overhead during a finale phase
    # while the walking gait would otherwise swing it).
    bones: dict[str, dict[str, Any]]
    # Same shape as ``bones`` but interpreted as bone-LOCAL basis
    # rotations rather than world-frame deltas: x_rad/y_rad/z_rad are
    # Blender-style intrinsic XYZ Euler angles in the bone's own basis
    # frame (matching ``pose_bone.rotation_euler`` with rotation_mode
    # 'XYZ'). The runtime writes ``rotation = basis_quat @ rest_rotation``
    # directly without the world-frame conjugation ``bones:`` does,
    # because Blender's ``matrix_basis`` is already in the correct
    # frame for direct concatenation.
    #
    # Use this when importing a pose authored in Blender (or any DCC
    # that exports bone rotations in the bone's local frame). Don't use
    # it with the gait system — gait writes world deltas and they don't
    # compose with local basis rotations cleanly.
    bones_local: dict[str, dict[str, Any]]
    # Cross-fade windows in seconds. When > 0 AND the next phase's
    # ``blend_in_sec`` is also > 0, the runtime evaluates BOTH phases'
    # body / bones / morphs outputs in the overlap window (using the
    # mutual minimum) and lerps between them. Gait is NOT blended —
    # only the current phase's gait runs at any given time, since
    # blending two step-based gaits is ill-defined.
    blend_in_sec: float
    blend_out_sec: float
    # Optional pose preset name to compose UNDER the phase's bones —
    # the preset's per-axis values become the starting silhouette,
    # then ``bones`` overrides any axis the phase explicitly authors.
    # ``pose_weight`` is an optional value-curve scaling the preset's
    # values per frame (0 = preset disabled, 1 = preset at full
    # strength). Lets a phase ease in / out of a preset.
    pose: str | None
    pose_weight: Any  # value-curve spec; defaults to 1.0
    # Optional finger / hand presets per side. Resolved against the
    # document's hand library at runtime; identical machinery to body
    # ``pose`` but a separate library namespace so finger poses don't
    # collide with body silhouette presets.
    hand_l: str | None
    hand_r: str | None
    # Optional world-space target positions for IK-driving the wrist
    # bones. When both ``hand_l_target`` and ``rig.arm_chain_l`` (or
    # the right-side equivalents) are non-None, the runtime runs
    # two-bone IK on the arm chain each frame to land the wrist at
    # the target. Each target is a 3-tuple of value curves (so the
    # target can move per frame); ``None`` per side leaves that arm
    # alone. Used to pin hands to a fixed point on the floor for
    # quadruped poses, or to track a moving prop. Parsed from a phase
    # ``ik`` block: ``{ik: {hand_l_target: [x, y, z], ...}}``.
    hand_l_target: tuple[Any, Any, Any] | None
    hand_r_target: tuple[Any, Any, Any] | None
    # Foot IK targets — same semantics as ``hand_l/r_target`` but resolved
    # against ``rig.leg_chain_l/r``. Used by quadruped poses to plant the
    # ankle bone AT a chosen world position (typically on the floor) so
    # the cloth/skin clamp doesn't have to push the foot mesh up from an
    # underground bone. Parsed from the same ``ik`` block:
    # ``{ik: {foot_l_target: [x, y, z], foot_r_target: [...]}}``.
    foot_l_target: tuple[Any, Any, Any] | None = None
    foot_r_target: tuple[Any, Any, Any] | None = None
    # Bone-name list whose end of frame world rotation should be aligned
    # so the bone's local +Y axis points world -Y — i.e. the bone's
    # 'top face' is up, 'bottom face' (sole / palm) is down. Useful for
    # planting feet flat on the floor without authoring per-axis foot
    # rotations by hand. Parsed from a phase ``floor_align`` array:
    # ``"floor_align": ["foot_L", "foot_R"]``. Empty list = no aligner.
    floor_align: tuple[str, ...] = ()


@dataclass(frozen=True)
class CameraKey:
    """One camera animation keyframe."""

    at_sec: float
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    fov_degrees: float | None = None


@dataclass(frozen=True)
class LyricLine:
    """One lyric line with absolute start / end times in seconds."""

    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class ClothPieceSpec:
    """Declarative cloth attachment.

    ``mesh_node`` is the scene-tree node name whose mesh should be
    simulated as cloth (the importer leaves a node per primitive — for
    a Galaxia / VRoid avatar the skirt mesh is the one named after its
    material like ``F00_001_01_Bottoms_01_CLOTH``). The other fields
    forward straight into ``ClothHost.add_cloth_for_node`` — defaults
    suit a draping skirt anchored at its hip-side top edge.

    ``track_bone`` is optional. When set, the runtime mutates each
    anchored vertex's position each frame to follow that bone's world
    transform — so a skirt anchored to the hip waistband stays glued
    to the hip even when the dance rotates the hip bone (which the
    cloth's static world-frame would otherwise miss because the mesh
    node isn't parented to that bone in the imported scene tree).
    """

    mesh_node: str
    structural_stiffness: float = 0.85
    bend_stiffness: float = 0.12
    linear_damping: float = 0.985
    rest_pull: float = 4.0
    anchor_axis: int = 1
    anchor_fraction: float = 0.15
    anchor_mode: str = "top_axis"
    iterations: int = 8
    substeps: int = 2
    track_bone: str | None = None


@dataclass(frozen=True)
class ColliderSpec:
    """Bone-following collider that pushes cloth vertices outside.

    ``kind`` is ``"sphere"`` or ``"capsule"``. Sphere colliders track
    one bone (``follow_bone``) — the collider's centre is mutated to
    the bone's world position each frame. Capsule colliders need
    ``end_bone`` too; the capsule spans the two bones' world
    positions. ``radius`` plus the cloth's per-piece skin offset
    determine the keep-out distance.
    """

    kind: str
    follow_bone: str
    radius: float
    end_bone: str | None = None
    skin_offset: float = 0.005


@dataclass(frozen=True)
class AudioSpec:
    """Audio attachment for a declarative animation.

    ``path`` is the WAV path as written in the JSON; the runtime
    resolves it relative to the .json's directory at load time.
    ``offset_sec`` shifts the audio's clock by a constant so the dance
    can start before / after the audio file's t=0. ``sync_clock`` swaps
    the runtime's wall-clock time provider for the audio player's
    playback position so the entire animation drifts with the music.
    """

    path: str
    offset_sec: float = 0.0
    sync_clock: bool = False


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
    # Pose presets keyed by name — built-ins overlaid with whatever the
    # document declared in ``pose_library`` (user entries win). Phases
    # reference these via ``pose: "name"``.
    pose_library: dict[str, PoseSpec]
    # Hand / finger preset library keyed by name (built-ins:
    # peace_L/R, fist_L/R, point_L/R, open_palm_L/R, thumbs_up_L/R).
    # Phases reference these via ``hand_L: "name"`` / ``hand_R: "name"``.
    hand_library: dict[str, PoseSpec]
    # Camera keyframes (sorted by absolute time in seconds). Each entry
    # carries position / target / fov_degrees. The runtime lerps
    # between bracketing keyframes per frame and writes the result to
    # the viewport's Camera (passed in via ``api['camera']``). Empty
    # tuple → camera is left untouched.
    camera_keys: tuple[CameraKey, ...]
    # Optional audio attachment. ``None`` keeps the runtime silent and
    # avoids loading the audio backend entirely.
    audio: AudioSpec | None
    # Optional karaoke-style lyric lines. Each frame the runtime finds
    # the active line (start_sec ≤ elapsed < end_sec) and pushes its
    # text through ``api['overlay']``. Empty tuple → overlay never
    # touched (legacy / no-lyrics docs).
    lyrics: tuple[LyricLine, ...]
    # Scene node names to detach from the scene tree at start. Useful
    # when the loaded .glb bundles props (stairs, room, lights) that
    # the dance shouldn't include. Names go through ``scene.find`` so
    # any descendant of the root with that name is detached from its
    # parent. Empty tuple → no detachment.
    hide_nodes: tuple[str, ...]
    # Cloth pieces to register at start. Each entry binds a scene
    # mesh node into the cloth solver so it sims as cloth (skirt /
    # cape / tie / sleeve). Empty tuple → no cloth registration.
    cloth_pieces: tuple[ClothPieceSpec, ...]
    # Bone-following colliders for the cloth solver. Each entry tracks
    # one (sphere) or two (capsule) bones — the collider geometry is
    # mutated each frame to match the bones' world positions, so cloth
    # naturally keeps clear of the body / arms / hands as the dance
    # moves them. Empty tuple → no colliders.
    colliders: tuple[ColliderSpec, ...]
    # Auto-emit a standard humanoid body collider set (hip sphere +
    # thigh + shin capsules) from ``scene.bone_aliases`` at start when
    # ``cloth_pieces`` is non-empty. Saves authors from re-specifying
    # the same collider list on every per-character animation. Default
    # ``True``; set ``"auto_body_colliders": false`` in the document to
    # opt out (e.g. for non-humanoid rigs or when the explicit list is
    # already comprehensive).
    auto_body_colliders: bool = True
    # Mesh nodes to register for post-skin collision push-out. Each
    # listed mesh is driven by its bone skinning every tick and then
    # projected against the registered body colliders — stops dress /
    # sleeve / hair-fringe verts from clipping into the torso during
    # animation without re-rigging the asset. Strings are scene node
    # names; the mesh primitive index can be supplied as a (name,
    # mesh_index) tuple if a node carries multiple primitives.
    collision_deform_meshes: tuple[Any, ...] = ()
    # When ``ground.kind == "flat"`` and this is ``True`` (default), the
    # runtime walks the scene at start, finds every SkinRefComponent
    # node, and registers each one as a passive collision_deform piece
    # so the engine's cloth-floor clamp covers every clothing / hair /
    # accessory mesh — not just the ones the document enumerated by
    # hand. Existing explicit entries are kept; auto-discovered names
    # are added only if not already listed. Set to ``False`` to opt out
    # for rigs whose meshes should be free to drape below the floor
    # plane (e.g. underground sequences, abstract scenes).
    auto_clamp_skinned_to_ground: bool = True


@dataclass
class PhaseOutput:
    """Computed (not yet applied) per-frame output of a single phase.

    Used by the cross-fade path to blend two phases at boundaries
    without writing intermediate state into the scene. ``yaw`` /
    ``lean`` / ``translation`` are scalars / triples, ``bones`` maps
    bone keys to body-frame quaternion deltas (pre yaw-conjugation),
    ``morphs`` maps morph names to weights.

    ``pose_blends`` is the gait-aware path: each entry is the FULL
    body-frame target rotation for that bone PLUS a weight in [0,1]
    saying how much to slerp from gait's current rotation toward that
    target. weight = 0 leaves gait untouched (arm hangs naturally);
    weight = 1 snaps to the pose target; in between produces a real
    "slow rise from hang to pose" instead of the rest-pose-scaled
    intermediate that a magnitude-scale weight would give.
    """

    yaw: float
    lean: float
    translation: tuple[float, float, float]
    bones: dict[str, np.ndarray] = field(default_factory=dict)
    # Bone-LOCAL basis quaternions (Blender-style). Applied without the
    # world-frame conjugation that ``bones`` goes through — the runtime
    # writes ``node.transform.rotation = q @ rest_rotation`` directly.
    bones_local: dict[str, np.ndarray] = field(default_factory=dict)
    morphs: dict[str, float] = field(default_factory=dict)
    pose_blends: dict[str, tuple[np.ndarray, float]] = field(default_factory=dict)


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
    Morph weights use scalar lerp.

    **One-sided bones** (a bone present in only one of the two phases)
    are emitted at the present phase's full strength rather than being
    slerp'd toward the identity quaternion. Identity here means "rest
    pose", which on a VRoid rig is a T-pose for arms — slerping toward
    it produces a visible T-pose flash through the cross-fade window.
    Keeping the present phase's value full means the gait's arm_hang
    keeps showing on the side that doesn't drive that bone, and the
    transition is one-shot at the boundary instead of two passes
    through identity. Morphs absent from one side fall back to weight 0
    (the natural neutral for a morph).
    """
    out = PhaseOutput(
        yaw=a.yaw + (b.yaw - a.yaw) * t,
        lean=a.lean + (b.lean - a.lean) * t,
        translation=_lerp_translation(a.translation, b.translation, t),
    )
    bone_keys = set(a.bones) | set(b.bones)
    for key in bone_keys:
        qa = a.bones.get(key)
        qb = b.bones.get(key)
        if qa is not None and qb is not None:
            out.bones[key] = quat_slerp(qa, qb, t)
        elif qa is not None:
            out.bones[key] = qa
        else:
            out.bones[key] = qb
    morph_keys = set(a.morphs) | set(b.morphs)
    for key in morph_keys:
        wa = a.morphs.get(key, 0.0)
        wb = b.morphs.get(key, 0.0)
        out.morphs[key] = wa + (wb - wa) * t
    pose_keys = set(a.pose_blends) | set(b.pose_blends)
    for key in pose_keys:
        if key in a.pose_blends and key in b.pose_blends:
            target_a, weight_a = a.pose_blends[key]
            target_b, weight_b = b.pose_blends[key]
            blended_target = quat_slerp(target_a, target_b, t)
            blended_weight = weight_a + (weight_b - weight_a) * t
        elif key in a.pose_blends:
            target_a, weight_a = a.pose_blends[key]
            # A's pose fades out as the cross-fade progresses; at t=1
            # the bone is fully back to gait baseline. This produces a
            # smooth "lower the arm" exit when the next phase doesn't
            # touch this bone.
            blended_target = target_a
            blended_weight = weight_a * (1.0 - t)
        else:
            target_b, weight_b = b.pose_blends[key]
            # Symmetric: B's pose fades in as the cross-fade progresses.
            # At t=0 the bone is at gait baseline; at t=1 fully posed.
            # This is what produces the "slow rise" the user wants when
            # a sway phase blends into a reach phase.
            blended_target = target_b
            blended_weight = weight_b * t
        out.pose_blends[key] = (blended_target, blended_weight)
    return out


def _parse_rig(raw: dict[str, Any]) -> RigBindings:
    if not isinstance(raw, dict):
        raise DeclarativeAnimationError("'rig' must be an object")
    leg_l = raw.get("leg_chain_l")
    leg_r = raw.get("leg_chain_r")
    arm_l = raw.get("arm_chain_l")
    arm_r = raw.get("arm_chain_r")
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
        arm_chain_l=tuple(arm_l) if arm_l else None,
        arm_chain_r=tuple(arm_r) if arm_r else None,
        elbow_limit_min=(
            tuple(_resolve_scalar(v) for v in raw["elbow_limit_min"])
            if "elbow_limit_min" in raw else None
        ),
        elbow_limit_max=(
            tuple(_resolve_scalar(v) for v in raw["elbow_limit_max"])
            if "elbow_limit_max" in raw else None
        ),
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
    bones_local = _parse_bones(raw.get("bones_local", {}))
    blend_in = float(raw.get("blend_in_sec", 0.0))
    blend_out = float(raw.get("blend_out_sec", 0.0))
    if blend_in < 0.0 or blend_out < 0.0:
        raise DeclarativeAnimationError(
            "blend_in_sec / blend_out_sec must be non-negative",
        )
    pose, pose_weight = _parse_pose(raw.get("pose"))
    hand_l = _parse_hand_field(raw.get("hand_L"), "hand_L")
    hand_r = _parse_hand_field(raw.get("hand_R"), "hand_R")
    hand_l_target, hand_r_target, foot_l_target, foot_r_target = _parse_ik_block(
        raw.get("ik"),
    )
    floor_align_raw = raw.get("floor_align", ())
    if not isinstance(floor_align_raw, (list, tuple)):
        raise DeclarativeAnimationError(
            "phase 'floor_align' must be a list of bone names",
        )
    floor_align = tuple(str(b) for b in floor_align_raw)
    return Phase(
        name=str(raw.get("name", "")),
        duration_sec=_resolve_phase_duration(raw, bpm),
        body_yaw_rad=body.get("yaw_rad", 0.0),
        body_lean_x_rad=body.get("lean_x_rad", 0.0),
        body_translation=body.get("translation", {}),
        gait=raw.get("gait"),
        morphs=morphs,
        bones=bones,
        bones_local=bones_local,
        blend_in_sec=blend_in,
        blend_out_sec=blend_out,
        pose=pose,
        pose_weight=pose_weight,
        hand_l=hand_l,
        hand_r=hand_r,
        hand_l_target=hand_l_target,
        hand_r_target=hand_r_target,
        foot_l_target=foot_l_target,
        foot_r_target=foot_r_target,
        floor_align=floor_align,
    )


def _parse_ik_block(raw: Any) -> tuple[
    tuple[Any, Any, Any] | None, tuple[Any, Any, Any] | None,
    tuple[Any, Any, Any] | None, tuple[Any, Any, Any] | None,
]:
    """Parse the optional per-phase ``ik`` block.

    Returns ``(hand_l, hand_r, foot_l, foot_r)`` — each is a 3-tuple of
    value-curve specs or ``None`` if the side wasn't authored. Missing
    block = all ``None``.
    """
    if raw is None:
        return None, None, None, None
    if not isinstance(raw, dict):
        raise DeclarativeAnimationError(
            f"phase 'ik' must be an object, got {type(raw).__name__}",
        )
    return (
        _parse_hand_target(raw.get("hand_l_target"), "hand_l_target"),
        _parse_hand_target(raw.get("hand_r_target"), "hand_r_target"),
        _parse_hand_target(raw.get("foot_l_target"), "foot_l_target"),
        _parse_hand_target(raw.get("foot_r_target"), "foot_r_target"),
    )


_HAND_TARGET_AXIS_COUNT = 3


def _parse_hand_target(raw: Any, field_name: str) -> tuple[Any, Any, Any] | None:
    """Validate a single hand-IK target — a 3-element list of value curves."""
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != _HAND_TARGET_AXIS_COUNT:
        raise DeclarativeAnimationError(
            f"ik.{field_name} must be a 3-element [x, y, z] list",
        )
    return (raw[0], raw[1], raw[2])


def _parse_hand_field(raw: Any, field_name: str) -> str | None:
    """Validate a phase's ``hand_L`` / ``hand_R`` field — string preset name."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise DeclarativeAnimationError(
            f"phase '{field_name}' must be a preset name string, got "
            f"{type(raw).__name__}",
        )
    return raw


def _parse_pose(raw: Any) -> tuple[str | None, Any]:
    """Validate the ``pose`` field on a phase.

    Two shapes:
    - ``"name"`` — preset at full weight.
    - ``{"name": "name", "weight": curve}`` — preset with a per-frame
      weight curve. Resolution against the actual library happens at
      runtime so the user can omit a preset name and we just no-op.
    """
    if raw is None:
        return None, 1.0
    if isinstance(raw, str):
        return raw, 1.0
    if isinstance(raw, dict):
        name = raw.get("name")
        if not isinstance(name, str):
            raise DeclarativeAnimationError(
                "phase 'pose' object must have a string 'name' field",
            )
        return name, raw.get("weight", 1.0)
    raise DeclarativeAnimationError(
        f"phase 'pose' must be a string or object, got {type(raw).__name__}",
    )


_BONE_AXES = ("x_rad", "y_rad", "z_rad")
# Author-friendly aliases — ``"x"`` reads as ``"x_rad"``, etc. The
# verbose form is kept canonical for backwards compatibility with
# every existing JSON; the short form simply rewrites in
# :func:`_parse_bones` before the canonical axes check fires.
_BONE_AXIS_ALIASES = {"x": "x_rad", "y": "y_rad", "z": "z_rad"}


def _parse_bones(raw: Any) -> dict[str, dict[str, Any]]:
    """Validate the per-phase ``bones`` block.

    Each entry is ``{bone_name: {x_rad?: curve, y_rad?: curve, z_rad?: curve}}``.
    Short-form axis names (``"x"`` / ``"y"`` / ``"z"``) are accepted as
    aliases for ``"x_rad"`` / ``"y_rad"`` / ``"z_rad"``; mixing the two
    forms on the same bone (``{"x": ..., "x_rad": ...}``) raises so an
    author who refactors halfway gets a clear error instead of silently
    losing one of the writes. Unknown axes are rejected loudly so a typo
    (``x_red``) surfaces at parse time instead of silently producing a
    still pose at runtime.
    """
    if not isinstance(raw, dict):
        raise DeclarativeAnimationError("phase 'bones' must be an object")
    out: dict[str, dict[str, Any]] = {}
    for bone_name, axes in raw.items():
        if not isinstance(axes, dict):
            raise DeclarativeAnimationError(
                f"bones[{bone_name!r}] must be an object of axis curves",
            )
        normalised: dict[str, Any] = {}
        for axis_key, curve in axes.items():
            canonical = _BONE_AXIS_ALIASES.get(axis_key, axis_key)
            if canonical not in _BONE_AXES:
                raise DeclarativeAnimationError(
                    f"bones[{bone_name!r}] has unknown axis {axis_key!r}; "
                    f"expected any of {list(_BONE_AXES)} "
                    f"or short aliases {sorted(_BONE_AXIS_ALIASES)}",
                )
            if canonical in normalised:
                raise DeclarativeAnimationError(
                    f"bones[{bone_name!r}] writes axis {canonical!r} twice — "
                    f"don't mix short ({axis_key!r}) and long forms",
                )
            normalised[canonical] = curve
        out[str(bone_name)] = normalised
    return out


def resolve_extends(
    document: dict[str, Any],
    source_dir: Path | None,
    _seen: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Deep-merge any ``extends`` chain into ``document``, returning the result.

    The ``extends`` field is the file-system path (relative to
    ``source_dir``) of another animation JSON to inherit from. Common
    rig / physics_chains / colliders / ground / wind / pose_library /
    hand_library / cloth boilerplate can live in a shared "profile"
    JSON; the child's keys override the parent on a per-key basis. The
    merge is:

    * Dicts: recursive shallow merge (child wins per-key).
    * Lists / scalars: child replaces parent wholesale.
    * ``phases``: never merged from parent — phases are always
      authored in the child file.

    Merge is **shallow at the top level**: each child key replaces the
    parent's value entirely. This avoids Frankenstein dicts (mixing
    ``ground: {kind: flat}`` with ``ground: {kind: stairs}`` would
    otherwise leak ``y`` from the flat default into the stair spec).
    To tune only part of a section (e.g. one physics chain's stiffness)
    the author copies the whole section in their child — the
    boilerplate that motivated ``extends`` is rarely that fine-grained,
    and the simpler rule pays back in fewer surprises.

    Two top-level keys do recurse one level: ``pose_library`` and
    ``hand_library`` merge per-preset, so a child can add or override
    individual poses without redeclaring the whole library.

    Cycles are rejected. Each ``extends`` reference goes through
    :func:`resolve_safe` so a profile file cannot point outside its
    source directory. Maximum chain depth is bounded by ``_MAX_EXTENDS``
    so a maliciously deep chain cannot tie up the parser.
    """
    extends_ref = document.get("extends")
    if extends_ref is None:
        return document
    if not isinstance(extends_ref, str):
        raise DeclarativeAnimationError(
            f"'extends' must be a string path, got {type(extends_ref).__name__}",
        )
    if source_dir is None:
        raise DeclarativeAnimationError(
            "'extends' requires a source directory — declarative loader "
            "must be invoked through load_animation() with a filename",
        )
    if len(_seen) >= _MAX_EXTENDS:
        raise DeclarativeAnimationError(
            f"'extends' chain exceeds maximum depth of {_MAX_EXTENDS}",
        )
    try:
        parent_path = resolve_safe(source_dir, extends_ref)
    except UnsafePathError as err:
        raise DeclarativeAnimationError(
            f"'extends' path {extends_ref!r} rejected: {err}",
        ) from err
    if parent_path in _seen:
        cycle = " → ".join(str(p) for p in (*_seen, parent_path))
        raise DeclarativeAnimationError(f"'extends' cycle detected: {cycle}")
    try:
        parent_source = parent_path.read_text(encoding="utf-8")
    except OSError as err:
        raise DeclarativeAnimationError(
            f"failed to read extended profile {extends_ref!r}: {err}",
        ) from err
    try:
        parent_doc = json.loads(parent_source)
    except json.JSONDecodeError as err:
        raise DeclarativeAnimationError(
            f"failed to parse extended profile {extends_ref!r}: "
            f"{err.msg} at line {err.lineno}",
        ) from err
    parent_resolved = resolve_extends(
        parent_doc, parent_path.parent, (*_seen, parent_path),
    )
    # Strip the ``extends`` field from the child copy; it has been
    # consumed and propagating it would re-trigger inheritance on a
    # later call. ``parent_resolved`` carried no ``extends`` (already
    # resolved recursively) so the merge result is also clean.
    child = {k: v for k, v in document.items() if k != "extends"}
    return _merge_extends(parent_resolved, child)


def _merge_extends(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Shallow top-level merge with pose/hand-library recursion.

    Child keys replace parent keys outright at the top level. The two
    exceptions are :data:`_EXTENDS_DEEP_MERGE_KEYS` (``pose_library``
    and ``hand_library``) where per-preset merging is the natural
    pattern — a child can add a new pose without redeclaring every
    inherited one. Phases are NEVER inherited: even if the parent
    declares a ``phases`` list, the child's value wins outright, and
    a child without ``phases`` does not pull the parent's in
    (validated downstream in :func:`parse_animation`).
    """
    # Exclude ``phases`` from the parent unconditionally — a profile
    # that happens to include phases (typically a complete animation
    # the child is just extending for tuning) should not have those
    # phases bleed through when the child redefines its own timeline.
    out: dict[str, Any] = {k: v for k, v in parent.items() if k != "phases"}
    for key, value in child.items():
        existing = out.get(key)
        if (
            key in _EXTENDS_DEEP_MERGE_KEYS
            and isinstance(existing, dict)
            and isinstance(value, dict)
        ):
            merged: dict[str, Any] = dict(existing)
            merged.update(value)
            out[key] = merged
        else:
            out[key] = value
    return out


_EXTENDS_DEEP_MERGE_KEYS = frozenset({"pose_library", "hand_library"})


# Hard cap on inheritance depth. A chain of more than 4 profiles is
# almost certainly an authoring mistake (e.g. circular reference that
# only diverges late) and should be rejected fast.
_MAX_EXTENDS = 4


def parse_animation(document: dict[str, Any]) -> DeclarativeAnimation:
    """Validate and parse a declarative-animation document.

    Call :func:`resolve_extends` first if the document might use the
    ``extends`` inheritance mechanism; ``parse_animation`` itself does
    not touch the filesystem, so passing a raw extends-bearing dict here
    silently ignores the inheritance.
    """
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
        raise DeclarativeAnimationError(
            "animation must declare at least one 'phases' entry",
        )
    return DeclarativeAnimation(
        name=str(document.get("name", "unnamed")),
        loop_sec=float(document.get("loop_sec", sum(p.duration_sec for p in phases))),
        rig=_parse_rig(document.get("rig", {})),
        ground=_parse_ground(document.get("ground")),
        phases=phases,
        physics_chains=_parse_physics_chains(document.get("physics_chains", {})),
        wind=_parse_wind(document.get("wind")),
        bpm=bpm,
        pose_library=_parse_pose_library(document.get("pose_library")),
        hand_library=_parse_hand_library(document.get("hand_library")),
        camera_keys=_parse_camera_keys(document.get("camera"), bpm),
        audio=_parse_audio(document.get("audio")),
        lyrics=_parse_lyrics(document.get("lyrics"), bpm),
        hide_nodes=_parse_hide(document.get("hide")),
        cloth_pieces=_parse_cloth(document.get("cloth")),
        colliders=_parse_colliders(document.get("colliders")),
        auto_body_colliders=bool(document.get("auto_body_colliders", True)),
        auto_clamp_skinned_to_ground=bool(
            document.get("auto_clamp_skinned_to_ground", True),
        ),
        collision_deform_meshes=_parse_collision_deform_meshes(
            document.get("collision_deform_meshes"),
        ),
    )


def _parse_collision_deform_meshes(raw: Any) -> tuple[Any, ...]:
    """Validate the document-level ``collision_deform_meshes`` list.

    Accepts a list of either plain node-name strings or ``(name, mesh_index)``
    tuples for nodes with multiple mesh primitives. Returns the canonicalised
    tuple form; entries that don't match either shape raise.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise DeclarativeAnimationError(
            f"collision_deform_meshes must be a list, got {type(raw).__name__}",
        )
    parsed: list[Any] = []
    for entry in raw:
        if isinstance(entry, str):
            parsed.append(entry)
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:    # noqa: PLR2004
            name, mesh_index = entry
            parsed.append((str(name), int(mesh_index)))
        else:
            raise DeclarativeAnimationError(
                f"collision_deform_meshes entry must be a string or "
                f"[name, mesh_index] pair, got {entry!r}",
            )
    return tuple(parsed)


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


def _parse_camera_keys(
    raw: Any, bpm: float,
) -> tuple[CameraKey, ...]:
    """Validate the document-level ``camera`` keyframe array.

    Each entry must specify ``at_sec`` OR ``at_beat`` (the latter
    requires ``bpm > 0``). Position and target are 3-vectors of
    numbers; ``fov`` (degrees) is optional and falls through to the
    Camera's existing fov when missing. Output is sorted by time so
    the per-frame bracket search is a simple bisect.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise DeclarativeAnimationError("'camera' must be an array of keyframes")
    keys: list[CameraKey] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise DeclarativeAnimationError(
                "each camera keyframe must be an object",
            )
        keys.append(_parse_camera_key(entry, bpm))
    keys.sort(key=lambda k: k.at_sec)
    return tuple(keys)


def _parse_camera_key(entry: dict[str, Any], bpm: float) -> CameraKey:
    has_sec = "at_sec" in entry
    has_beat = "at_beat" in entry
    if has_sec == has_beat:
        raise DeclarativeAnimationError(
            "camera keyframe needs exactly one of 'at_sec' or 'at_beat'",
        )
    if has_beat:
        if bpm <= 0.0:
            raise DeclarativeAnimationError(
                "camera keyframe uses 'at_beat' but the document has no "
                "positive 'bpm' to convert it",
            )
        at_sec = float(entry["at_beat"]) * _SECONDS_PER_MINUTE / bpm
    else:
        at_sec = float(entry["at_sec"])
    position = _parse_vec3(entry.get("position"), "camera.position")
    target = _parse_vec3(entry.get("target"), "camera.target")
    fov_raw = entry.get("fov")
    fov = float(fov_raw) if fov_raw is not None else None
    return CameraKey(
        at_sec=at_sec, position=position, target=target, fov_degrees=fov,
    )


def _parse_vec3(raw: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != _VEC3_LEN:
        raise DeclarativeAnimationError(
            f"{field_name} must be a 3-element array of numbers",
        )
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _parse_pose_dict(
    raw: Any, field_name: str,
) -> dict[str, PoseSpec]:
    """Validate a ``{preset_name: {bone: {axis: scalar}}}`` document fragment.

    Shared validation for both ``pose_library`` and ``hand_library`` —
    same shape, same axis whitelist, same error message style. The
    caller chooses which built-in library to merge the result into.
    """
    if not isinstance(raw, dict):
        raise DeclarativeAnimationError(f"'{field_name}' must be an object")
    out: dict[str, PoseSpec] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise DeclarativeAnimationError(
                f"{field_name}[{name!r}] must be an object of bones",
            )
        bones: PoseSpec = {}
        for bone, axes in spec.items():
            if not isinstance(axes, dict):
                raise DeclarativeAnimationError(
                    f"{field_name}[{name!r}][{bone!r}] must be an axis object",
                )
            unknown = set(axes) - set(_BONE_AXES)
            if unknown:
                raise DeclarativeAnimationError(
                    f"{field_name}[{name!r}][{bone!r}] has unknown axes "
                    f"{sorted(unknown)}; expected any of {list(_BONE_AXES)}",
                )
            bones[str(bone)] = {str(k): float(v) for k, v in axes.items()}
        out[str(name)] = bones
    return out


def _parse_pose_library(raw: Any) -> dict[str, PoseSpec]:
    """Validate the document-level ``pose_library`` and merge with built-ins."""
    if raw is None:
        return merge_pose_libraries(None)
    return merge_pose_libraries(_parse_pose_dict(raw, "pose_library"))


def _parse_hand_library(raw: Any) -> dict[str, PoseSpec]:
    """Validate the document-level ``hand_library`` and merge with built-ins."""
    if raw is None:
        return merge_hand_libraries(None)
    return merge_hand_libraries(_parse_pose_dict(raw, "hand_library"))


_DEFAULT_LYRIC_DURATION_SEC = 1.0


def _parse_lyrics(raw: Any, bpm: float) -> tuple[LyricLine, ...]:
    """Validate the document-level ``lyrics`` array.

    Each entry needs exactly one of ``at_sec`` / ``at_beat`` for its
    start and at most one of ``duration_sec`` / ``duration_beats`` for
    its length (defaults to a 1-second flash if neither is set —
    matches the typical "show line for one beat" karaoke convention).
    Output is sorted by start time so the per-frame active-lyric scan
    is a simple bisect-style walk.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise DeclarativeAnimationError("'lyrics' must be an array")
    out: list[LyricLine] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise DeclarativeAnimationError(
                "each lyric entry must be an object",
            )
        out.append(_parse_lyric_line(entry, bpm))
    out.sort(key=lambda line: line.start_sec)
    return tuple(out)


def _parse_lyric_line(entry: dict[str, Any], bpm: float) -> LyricLine:
    text = entry.get("text")
    if not isinstance(text, str):
        raise DeclarativeAnimationError(
            "lyric entry needs a string 'text' field",
        )
    start_sec = _resolve_time_field(
        entry, "at_sec", "at_beat", bpm, "lyric.start",
    )
    duration_sec = _DEFAULT_LYRIC_DURATION_SEC
    has_dur_sec = "duration_sec" in entry
    has_dur_beats = "duration_beats" in entry
    if has_dur_sec and has_dur_beats:
        raise DeclarativeAnimationError(
            "lyric entry has both 'duration_sec' and 'duration_beats'; "
            "specify exactly one",
        )
    if has_dur_beats:
        if bpm <= 0.0:
            raise DeclarativeAnimationError(
                "lyric uses 'duration_beats' but the document has no "
                "positive 'bpm' to convert it",
            )
        duration_sec = float(entry["duration_beats"]) * _SECONDS_PER_MINUTE / bpm
    elif has_dur_sec:
        duration_sec = float(entry["duration_sec"])
    return LyricLine(
        start_sec=start_sec,
        end_sec=start_sec + duration_sec,
        text=text,
    )


def _resolve_time_field(
    entry: dict[str, Any],
    sec_key: str,
    beat_key: str,
    bpm: float,
    label: str,
) -> float:
    """Pick exactly one of ``sec_key`` / ``beat_key`` and return seconds.

    Used by camera keyframes and lyric lines — both share the
    "specify time in either seconds or beats" convention.
    """
    has_sec = sec_key in entry
    has_beat = beat_key in entry
    if has_sec == has_beat:
        raise DeclarativeAnimationError(
            f"{label} needs exactly one of {sec_key!r} or {beat_key!r}",
        )
    if has_beat:
        if bpm <= 0.0:
            raise DeclarativeAnimationError(
                f"{label} uses {beat_key!r} but the document has no "
                f"positive 'bpm' to convert it",
            )
        return float(entry[beat_key]) * _SECONDS_PER_MINUTE / bpm
    return float(entry[sec_key])


_VALID_COLLIDER_KINDS = ("sphere", "capsule")


def _parse_cloth(raw: Any) -> tuple[ClothPieceSpec, ...]:
    """Validate the optional document-level ``cloth`` array."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise DeclarativeAnimationError("'cloth' must be an array of objects")
    out: list[ClothPieceSpec] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise DeclarativeAnimationError(
                "each cloth entry must be an object",
            )
        node = entry.get("mesh_node")
        if not isinstance(node, str) or not node:
            raise DeclarativeAnimationError(
                "cloth entry needs a non-empty string 'mesh_node'",
            )
        track = entry.get("track_bone")
        if track is not None and (not isinstance(track, str) or not track):
            raise DeclarativeAnimationError(
                "cloth.track_bone must be a non-empty string when set",
            )
        out.append(ClothPieceSpec(
            mesh_node=node,
            structural_stiffness=float(entry.get("structural_stiffness", 0.85)),
            bend_stiffness=float(entry.get("bend_stiffness", 0.12)),
            linear_damping=float(entry.get("linear_damping", 0.985)),
            rest_pull=float(entry.get("rest_pull", 4.0)),
            anchor_axis=int(entry.get("anchor_axis", 1)),
            anchor_fraction=float(entry.get("anchor_fraction", 0.15)),
            anchor_mode=str(entry.get("anchor_mode", "top_axis")),
            iterations=int(entry.get("iterations", 8)),
            substeps=int(entry.get("substeps", 2)),
            track_bone=str(track) if track else None,
        ))
    return tuple(out)


def _parse_colliders(raw: Any) -> tuple[ColliderSpec, ...]:
    """Validate the optional document-level ``colliders`` array."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise DeclarativeAnimationError("'colliders' must be an array of objects")
    out: list[ColliderSpec] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise DeclarativeAnimationError(
                "each collider entry must be an object",
            )
        kind = entry.get("kind", "sphere")
        if kind not in _VALID_COLLIDER_KINDS:
            raise DeclarativeAnimationError(
                f"collider kind {kind!r} not supported; expected one of "
                f"{list(_VALID_COLLIDER_KINDS)}",
            )
        follow = entry.get("follow_bone")
        if not isinstance(follow, str) or not follow:
            raise DeclarativeAnimationError(
                "collider entry needs a non-empty string 'follow_bone'",
            )
        end = entry.get("end_bone")
        if kind == "capsule" and (not isinstance(end, str) or not end):
            raise DeclarativeAnimationError(
                "capsule collider needs a non-empty string 'end_bone'",
            )
        radius = entry.get("radius")
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise DeclarativeAnimationError(
                "collider 'radius' must be a positive number",
            )
        out.append(ColliderSpec(
            kind=str(kind),
            follow_bone=follow,
            radius=float(radius),
            end_bone=str(end) if end else None,
            skin_offset=float(entry.get("skin_offset", 0.005)),
        ))
    return tuple(out)


def _parse_hide(raw: Any) -> tuple[str, ...]:
    """Validate the optional document-level ``hide`` array."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise DeclarativeAnimationError("'hide' must be an array of node names")
    out: list[str] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry:
            raise DeclarativeAnimationError(
                "each 'hide' entry must be a non-empty node-name string",
            )
        out.append(entry)
    return tuple(out)


def _parse_audio(raw: Any) -> AudioSpec | None:
    """Validate the optional document-level ``audio`` block."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise DeclarativeAnimationError("'audio' must be an object or null")
    path = raw.get("path")
    if not isinstance(path, str) or not path:
        raise DeclarativeAnimationError(
            "audio.path must be a non-empty string",
        )
    return AudioSpec(
        path=path,
        offset_sec=float(raw.get("offset_sec", 0.0)),
        sync_clock=bool(raw.get("sync_clock", False)),
    )


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
    # Optional reference to the viewport's Camera. When present AND the
    # document declared a camera keyframe array, the runtime lerps
    # between bracketing keyframes each frame and writes position /
    # target / fov_degrees onto this object. ``None`` → camera is
    # untouched (legacy / headless tests / character-only demos).
    camera_api: Any | None = None
    # Optional callable ``set_text(str) -> None`` for the karaoke
    # overlay. Called each frame with the active lyric (or empty
    # string when no lyric is active). Bootstrap wires this to
    # ``viewport.set_overlay_text``; tests pass a list-collecting stub.
    overlay_api: Callable[[str], None] | None = None
    # Resolution root for paths declared in the JSON (currently just
    # ``audio.path``). Defaults to None → paths resolve relative to
    # CWD; the loader sets this to the .json file's parent so audio
    # can sit next to the dance document.
    source_dir: Any | None = None
    # Optional override of the ``AudioPlayer`` factory — tests pass a
    # stub that records play / pause without touching QtMultimedia.
    audio_player_factory: Any | None = None
    # Optional reference to the engine's ``ClothHost``. When present
    # AND the document declared cloth pieces / colliders, the runtime
    # binds them at start and updates collider transforms per frame.
    cloth_host: Any | None = None
    _audio_player: Any | None = field(default=None, init=False)
    _wall_time: Callable[[], float] | None = field(default=None, init=False)
    _last_lyric_text: str = field(default="", init=False)
    # Bone-tracking colliders: list of (collider_obj, kind, head_node, tail_node).
    # ``tail_node`` is None for spheres, a Node for capsules.
    _bone_colliders: list[tuple[Any, str, Any, Any]] = field(default_factory=list)
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
    # Cached rest world rotation for every floor_align bone. Captured at
    # ``_start`` (before any pose-driven rotation overrides), used by
    # :meth:`_apply_floor_align` to restore the bone's INITIAL world
    # orientation — which on a humanoid rig is 'palm-down, fingers-out'
    # for wrists and 'sole-flat, toes-forward' for ankles. Single-axis
    # alignment can't recover this because the bone's local axes are
    # arbitrary 3D directions, not standard XYZ.
    _floor_align_rest_world: dict[int, np.ndarray] = field(default_factory=dict)
    # Per-bone 'bone direction in local frame' captured at start from the
    # first child's rest translation. Used as the secondary axis for the
    # 2-axis floor alignment so the foot lies HEEL-TO-TOE along the
    # chain's extension direction (sole flat AND heel on floor) instead
    # of just being sole-down with the foot's long axis free.
    _floor_align_bone_dir_local: dict[int, np.ndarray] = field(default_factory=dict)
    # Per-bone chain root node — used at align time to compute the
    # 'natural extension direction' for the bone (parent_joint → bone),
    # which is the world target for the bone's local bone-direction axis.
    _floor_align_chain_parent: dict[int, Any] = field(default_factory=dict)
    # Arm chain cache for hand-IK. Same shape as ``_leg_chain_nodes``:
    # ``{"L": (shoulder, elbow, wrist), "R": (...)}``. Populated at
    # ``_start`` from ``rig.arm_chain_l/r`` (resolved through the
    # body_bones alias). Empty when the rig didn't declare arm chains;
    # the runtime then skips hand IK entirely.
    _arm_chain_nodes: dict[str, tuple[Any, Any, Any]] = field(default_factory=dict)
    # Per-frame parent-world rotation cache. Both ``_set_bone`` and
    # ``_world_delta_to_local`` walk the bone's parent chain to compose
    # parent-world rotation. Adjacent bones (left arm, right arm, fingers)
    # share most of that chain — caching per-node within a single
    # ``_update`` collapses ~8 ancestor walks per bone × dozens of
    # written bones into a handful of unique chains. Cleared at the top
    # of each ``_update`` so a parent rotation changed earlier in the
    # frame doesn't leak into a later bone's reading.
    _frame_parent_world: dict[int, np.ndarray] = field(default_factory=dict)

    # ----- Hook surface (matches ScriptHost expectations) -------------------
    def hooks(self) -> dict[str, Callable[..., Any]]:
        return {
            "start": self._start,
            "update": self._update,
            "on_event": self._on_event,
        }

    # ----- start ------------------------------------------------------------
    def _start(self) -> None:
        # Detach any nodes the document asked us to hide BEFORE caching
        # bones so the cache reflects what's actually in the tree. The
        # bundled character.glb ships with a Stairs prop; a clean dance
        # scene declares ``hide: ["Stairs"]`` to drop it.
        self._detach_hidden_nodes()
        # Find the character root + cache its rest pose.
        root_name = self.animation.rig.character_root
        if root_name:
            root_node = self.scene.find(root_name)
            if root_node is not None:
                # Force the node's TRS fields to reflect the actual bind
                # transform — glTF nodes with a ``matrix`` field land in
                # ``matrix_override`` with TRS defaulted to identity. If we
                # cache that default identity as ``rest_rotation``, the
                # runtime's ``basis_quat * rest_rotation`` math composes
                # against the wrong base.
                root_node.transform._hydrate_trs_from_override()  # noqa: SLF001
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
            for bone_name in _phase_target_bones(
                phase,
                self.animation.rig,
                self.animation.pose_library,
                self.animation.hand_library,
            ):
                if bone_name in self._bone_drives:
                    continue
                node = self.scene.find(bone_name)
                if node is None:
                    continue
                # Same hydration as the root cache above — ensures the
                # captured ``rest_rotation`` is the bind rotation rather
                # than the identity placeholder that glTF
                # ``matrix_override`` nodes carry by default.
                node.transform._hydrate_trs_from_override()  # noqa: SLF001
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
        # Audio: load + start playback, optionally swap the time
        # provider so phases progress with the music's clock.
        self._setup_audio()
        # Cloth + bone-following colliders: register cloth pieces
        # (e.g. skirt) with the cloth host and create capsule / sphere
        # colliders that track named bones each frame so cloth keeps
        # clear of arms / hands as the dance moves them.
        self._setup_cloth_and_colliders()
        self._cache_ik_chains()

    def _cache_ik_chains(self) -> None:
        """Resolve leg + arm chains from the rig into scene-node tuples.

        Split out of :meth:`_start` to keep that method below the cyclomatic
        branch limit — the four (leg L, leg R, arm L, arm R) chain lookups
        each add a branch and tipped ``_start`` over the line.
        """
        for side, chain_names in (
            ("L", self.animation.rig.leg_chain_l),
            ("R", self.animation.rig.leg_chain_r),
        ):
            if chain_names is None:
                continue
            nodes = tuple(self.scene.find(n) for n in chain_names)
            if all(n is not None for n in nodes):
                self._leg_chain_nodes[side] = nodes
        aliases = self.animation.rig.body_bones
        for side, chain_names in (
            ("L", self.animation.rig.arm_chain_l),
            ("R", self.animation.rig.arm_chain_r),
        ):
            if chain_names is None:
                continue
            resolved_names = tuple(aliases.get(n, n) for n in chain_names)
            nodes = tuple(self.scene.find(n) for n in resolved_names)
            if all(n is not None for n in nodes):
                self._arm_chain_nodes[side] = nodes
            else:
                _log.warning(
                    "declarative: arm chain %s missing bone — IK disabled "
                    "(expected: %s, found: %s)",
                    side, resolved_names,
                    tuple(n.name if n is not None else None for n in nodes),
                )
        self._capture_floor_align_rest_world(aliases)

    def _capture_floor_align_rest_world(self, aliases: dict[str, str]) -> None:
        """For each floor_align bone, capture the LOCAL DIRECTION pointing world -Y in rest.

        On a humanoid bind pose 'palm faces world -Y' for the wrist
        and 'sole faces world -Y' for the ankle. The LOCAL direction
        carrying that 'world down' meaning is rig-specific AND in
        general not axis-aligned — Herta's wrist needs a 3D mixture
        of local axes (~(-0.49, +0.70, -0.53)) because its rest world
        rotation puts the bone at a ~45° tilt. Naive single-axis
        detection picks 'local +Y' because that column has the most
        negative world-Y component, but local +Y is the FINGER
        direction, not the palm normal — aligning it to world -Y
        sent the fingers stabbing into the floor.

        Math: world -Y direction expressed in the bone's local frame
        equals ``-rot.T[:, 1]`` = ``-rot[1, :]`` (row 1 of the rest
        rotation matrix, negated). Storing this 3-vector lets
        :meth:`_apply_floor_align` rotate the bone so this vector
        points world -Y in EVERY pose — sole/palm stays flat without
        spinning the bone's twist around its own axis.
        """
        bone_keys: set[str] = set()
        for phase in self.animation.phases:
            bone_keys.update(phase.floor_align)
        if not bone_keys:
            return
        # Map bone_key -> the IK chain whose END is that bone, so we can
        # find the chain's 'previous joint' (knee for foot, elbow for
        # hand) to derive the natural foot/hand extension direction.
        end_to_parent: dict[str, Any] = {}
        empty_chain = (None, None, None)
        for side, chain_def, chain_nodes in (
            ("L", self.animation.rig.leg_chain_l, self._leg_chain_nodes),
            ("R", self.animation.rig.leg_chain_r, self._leg_chain_nodes),
            ("L", self.animation.rig.arm_chain_l, self._arm_chain_nodes),
            ("R", self.animation.rig.arm_chain_r, self._arm_chain_nodes),
        ):
            if not chain_def:
                continue
            end_to_parent[chain_def[2]] = chain_nodes.get(side, empty_chain)[1]
        for bone_key in bone_keys:
            bone_name = aliases.get(bone_key, bone_key)
            node = self.scene.find(bone_name)
            if node is None:
                continue
            matrix = node.transform.to_matrix()
            parent = node.parent
            while parent is not None:
                matrix = parent.transform.to_matrix() @ matrix
                parent = parent.parent
            rot = matrix[:3, :3].astype(np.float32, copy=False)
            contact_local = -rot[1, :].astype(np.float32, copy=True)
            norm = float(np.linalg.norm(contact_local))
            if norm > 1.0e-6:  # noqa: PLR2004  # degenerate rest-pose guard
                contact_local = contact_local / norm
            self._floor_align_rest_world[id(node)] = contact_local
            # Bone direction in local frame = first child's REST translation,
            # normalised. Captures the rig's 'down the bone' direction
            # (toward fingertips / toe tip).
            bone_dir_local: np.ndarray | None = None
            if node.children:
                offset = np.asarray(
                    node.children[0].transform.translation, dtype=np.float32,
                ).reshape(3)
                offset_norm = float(np.linalg.norm(offset))
                if offset_norm > 1.0e-6:  # noqa: PLR2004
                    bone_dir_local = offset / offset_norm
            if bone_dir_local is not None:
                self._floor_align_bone_dir_local[id(node)] = bone_dir_local
            parent_chain = end_to_parent.get(bone_key)
            if parent_chain is not None:
                self._floor_align_chain_parent[id(node)] = parent_chain

    def _apply_pose_blends(
        self,
        pose_blends: dict[str, tuple[np.ndarray, float]],
        yaw: float,
    ) -> None:
        """Slerp each bone from its current rotation toward the pose target.

        ``pose_blends[bone_key] = (target_body_quat, weight)``. The
        target is yaw-conjugated to world, then composed with the
        bone's rest rotation to get the desired final local rotation.
        We then slerp from the bone's CURRENT local rotation (which
        already includes the gait's writes from the same frame) toward
        the desired final by ``weight``. ``weight = 0`` leaves gait
        intact; ``weight = 1`` snaps to the pose; in between produces
        the gait→pose interpolation that authors actually want when
        they ramp pose_weight or ride a cross-fade between phases.
        """
        for bone_key, (target_body_quat, weight) in pose_blends.items():
            if weight <= 0.0:
                continue
            bone_name = self.animation.rig.body_bones.get(bone_key, bone_key)
            drive = self._bone_drives.get(bone_name)
            if drive is None:
                continue
            target_world_delta = _yaw_to_world(target_body_quat, yaw)
            target_local = self._world_delta_to_local(
                drive.node, target_world_delta,
            )
            target_full = quat_mul(target_local, drive.rest_rotation)
            current = drive.node.transform.rotation
            blended = quat_slerp(
                np.asarray(current, dtype=np.float32),
                target_full.astype(np.float32, copy=False),
                float(weight),
            )
            drive.node.transform.set_rotation(
                blended.astype(np.float32, copy=False),
            )
            self._invalidate_parent_world_for(drive.node)

    def _world_delta_to_local(
        self, node: Any, delta_world: np.ndarray,
    ) -> np.ndarray:
        """Conjugate a world-frame delta into the bone's parent-local frame.

        Same conjugation as ``_set_bone`` does inline; pulled out so
        ``_apply_pose_blends`` can compose its target the same way
        without duplicating the parent-chain walk.
        """
        parent_world = self._parent_world_rotation(node)
        parent_world_inv = quat_inverse(parent_world)
        return quat_mul(quat_mul(parent_world_inv, delta_world), parent_world)

    def _parent_world_rotation(self, node: Any) -> np.ndarray:
        """Compose ``node``'s parent-chain world rotation, memoised per frame.

        Recursive with a per-frame cache keyed on ``id(parent)``:
        once a chain is walked, every descendant whose path passes
        through the same ancestor reuses the cached partial. A 30-bone
        write pass over the Herta rig (each bone 5–7 levels deep)
        used to do ~200 ``quat_mul`` calls; the cache collapses that
        to ~30 (one per unique parent node visited that frame).
        """
        parent = node.parent
        if parent is None:
            return _IDENTITY_QUAT
        cached = self._frame_parent_world.get(id(parent))
        if cached is not None:
            return cached
        grandparent_world = self._parent_world_rotation(parent)
        result = quat_mul(grandparent_world, parent.transform.rotation)
        self._frame_parent_world[id(parent)] = result
        return result

    def _invalidate_parent_world_for(self, node: Any) -> None:
        """Drop the per-frame cache if ``node`` is currently cached as an ancestor.

        Cache entries are keyed on a node's ``id`` and hold THAT node's
        world rotation. Writing to a bone that has been cached as an
        ancestor by an earlier walk (rare in practice — typical
        animation writes are to leaves like hands / feet / head, which
        are never cached as parents) would leave a stale composition
        for any subsequent walk that passes through it. Clearing the
        whole map is cheaper than tracking descendants, and the common
        leaf-write path skips this branch entirely.
        """
        if id(node) in self._frame_parent_world:
            self._frame_parent_world.clear()

    def _detach_hidden_nodes(self) -> None:
        """Remove every named node in ``animation.hide_nodes`` from its parent.

        Names go through ``scene.find`` (whatever the parser already
        uses elsewhere) so any descendant matching the name is detached.
        Names that don't resolve are logged and skipped — covers the
        case where the same animation runs against two scenes that
        share most prop names but not all.
        """
        for name in self.animation.hide_nodes:
            node = self.scene.find(name)
            if node is None:
                _log.debug(
                    "declarative: hide target %r not in scene; skipping",
                    name,
                )
                continue
            parent = node.parent
            if parent is None:
                _log.warning(
                    "declarative: cannot hide %r — has no parent (root?)",
                    name,
                )
                continue
            parent.remove_child(node)

    def _setup_cloth_and_colliders(self) -> None:
        """Bind cloth pieces + create bone-tracking colliders at start.

        Both halves are gated on ``cloth_host`` being wired in (it
        comes from the script API; tests can stub it). For each cloth
        piece in the animation: locate the named scene node, call
        ``cloth_host.add_cloth_for_node`` with the per-spec PBD
        parameters. For each collider: instantiate the right shape
        (``SphereCollider`` for one-bone targets, ``CapsuleCollider``
        for two-bone targets) and stash the collider + bone-node refs
        in ``_bone_colliders`` so :meth:`_update_bone_colliders` can
        push their world positions into the collider geometry every
        frame. Missing modules / nodes / bones are logged + skipped
        rather than raising — same gating philosophy as audio / camera.
        """
        if self.cloth_host is None:
            return
        auto_meshes = self._setup_ground_and_discover_auto_meshes()
        wants_auto = (
            (self.animation.cloth_pieces or self.animation.collision_deform_meshes)
            and self.animation.auto_body_colliders
        )
        if (
            not self.animation.cloth_pieces
            and not self.animation.colliders
            and not self.animation.collision_deform_meshes
            and not auto_meshes
            and not wants_auto
        ):
            return
        try:
            from posecascade.animation.cloth import (  # noqa: PLC0415
                CapsuleCollider,
                SphereCollider,
            )
        except ImportError as err:
            _log.warning("declarative: cloth module unavailable: %s", err)
            return
        self._register_collision_deform_meshes()
        if auto_meshes:
            self._register_auto_skinned_meshes(auto_meshes)
        for piece in self.animation.cloth_pieces:
            node = self.scene.find(piece.mesh_node)
            if node is None:
                _log.warning(
                    "declarative: cloth mesh node %r not in scene; skipping",
                    piece.mesh_node,
                )
                continue
            cloth_piece = self.cloth_host.add_cloth_for_node(
                node,
                cloth_name=piece.mesh_node,
                anchor_axis=piece.anchor_axis,
                anchor_fraction=piece.anchor_fraction,
                anchor_mode=piece.anchor_mode,
                structural_stiffness=piece.structural_stiffness,
                bend_stiffness=piece.bend_stiffness,
                linear_damping=piece.linear_damping,
                iterations=piece.iterations,
                substeps=piece.substeps,
                rest_pull=piece.rest_pull,
            )
            self._maybe_register_anchor_follower(piece, cloth_piece)
        for spec in self._resolve_collider_specs():
            head_node = self.scene.find(spec.follow_bone)
            if head_node is None:
                _log.warning(
                    "declarative: collider follow_bone %r not in scene; skipping",
                    spec.follow_bone,
                )
                continue
            tail_node = None
            if spec.kind == "capsule":
                tail_node = self.scene.find(spec.end_bone) if spec.end_bone else None
                if tail_node is None:
                    _log.warning(
                        "declarative: capsule collider end_bone %r not in scene; "
                        "skipping",
                        spec.end_bone,
                    )
                    continue
            head_pos = _world_position(head_node).astype(np.float32)
            if spec.kind == "sphere":
                collider = SphereCollider(
                    center=head_pos.copy(),
                    radius=spec.radius,
                    skin_offset=spec.skin_offset,
                )
            else:
                tail_pos = _world_position(tail_node).astype(np.float32)
                collider = CapsuleCollider(
                    a=head_pos.copy(),
                    b=tail_pos.copy(),
                    radius=spec.radius,
                    skin_offset=spec.skin_offset,
                )
            self.cloth_host.add_collider(collider)
            self._bone_colliders.append(
                (collider, spec.kind, head_node, tail_node),
            )
            self._register_collider_bone_filter(collider, head_node, tail_node)

    def _register_collider_bone_filter(
        self, collider: object, head_node: Any, tail_node: Any | None,
    ) -> None:
        """Tell the host which skin joints this collider should NOT push.

        Registers ``head_node`` (and ``tail_node`` if distinct) plus every
        descendant — the passive-skin-deform filter then leaves verts
        weighted to those bones alone so a hand sphere doesn't inflate
        the hand mesh, and a thigh capsule doesn't push the thigh skin
        out of itself. Silent no-op if the host doesn't support the
        filter API (e.g. a stub host in tests).
        """
        if not hasattr(self.cloth_host, "register_collider_bone_filter"):
            return
        self.cloth_host.register_collider_bone_filter(collider, head_node)
        if tail_node is not None and tail_node is not head_node:
            self.cloth_host.register_collider_bone_filter(collider, tail_node)

    def _setup_ground_and_discover_auto_meshes(self) -> list[str]:
        """Wire ``cloth_host.floor_y`` from the animation's ground block
        + return the list of auto-discovered skinned meshes to register.

        Pulled out of :meth:`_setup_cloth_and_colliders` to keep that
        method below the cyclomatic branch limit (the ground + auto
        discovery were two of three branches the linter flagged).
        """
        ground = self.animation.ground
        flat_ground = ground is not None and ground.kind == "flat"
        if flat_ground and hasattr(self.cloth_host, "floor_y"):
            self.cloth_host.floor_y = _resolve_scalar(ground.params.get("y", 0.0))
        if flat_ground and self.animation.auto_clamp_skinned_to_ground:
            return self._discover_skinned_collision_deform_meshes()
        return []

    def _discover_skinned_collision_deform_meshes(self) -> list[str]:
        """Return scene-node names of every skinned mesh not already explicit.

        Walks the scene tree looking for nodes whose components include a
        :class:`SkinRefComponent`. Drops names already in
        ``animation.collision_deform_meshes`` so the registration pass
        doesn't process them twice. Result is the set that the
        auto-clamp path should register as ``passive_skin_deform``.

        Returns ``[]`` when the scripting / cloth dependency tree is
        unavailable (lets the runtime degrade cleanly in tests that stub
        the import surface).
        """
        try:
            from posecascade.scene.component import (  # noqa: PLC0415
                SkinRefComponent,
            )
        except ImportError:
            return []
        explicit: set[str] = set()
        for entry in self.animation.collision_deform_meshes:
            name = entry[0] if isinstance(entry, tuple) else entry
            explicit.add(str(name))
        discovered: list[str] = []
        seen: set[str] = set()
        # Iterative walk — recursion through the bundled Herta rig (354
        # joints + accessory subtrees) easily hits Python's default
        # recursion limit on smaller stacks, so we drain a stack here
        # instead.
        stack = [self.scene.root]
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            if node.name in explicit or node.name in seen:
                continue
            for component in node.components:
                if isinstance(component, SkinRefComponent):
                    discovered.append(node.name)
                    seen.add(node.name)
                    break
        return discovered

    def _register_auto_skinned_meshes(self, node_names: list[str]) -> None:
        """Register each auto-discovered skinned mesh as a passive cloth piece.

        Same shape parameters as :meth:`_register_collision_deform_meshes`
        but takes the names through a separate code path so an explicit
        entry that overrides a name (with a custom mesh_index, for
        example) wins on configuration — the discovery pass de-duped
        already, so we know these are fresh.
        """
        for name in node_names:
            node = self.scene.find(name)
            if node is None:
                continue
            piece = self.cloth_host.add_cloth_for_node(
                node,
                cloth_name=f"auto_clamp_{name}",
                anchor_axis=1,
                anchor_fraction=0.0,
                anchor_mode="top_axis",
                structural_stiffness=0.0,
                bend_stiffness=0.0,
                linear_damping=1.0,
                iterations=1,
                rest_pull=0.0,
            )
            if piece is None:
                continue
            piece.params.passive_skin_deform = True
            self.cloth_host.register_skin_target_follower(piece, node)

    def _register_collision_deform_meshes(self) -> None:
        """Register every ``collision_deform_meshes`` entry with the cloth host.

        Each mesh becomes a ``passive_skin_deform`` cloth piece: the
        cloth host drives it from the scene's bone skinning every tick
        (no dynamics) and projects it against the registered colliders.
        Missing nodes or unskinned meshes are logged + skipped — never
        a hard failure, because a swap to a different rig may simply
        omit the named mesh.
        """
        if not self.animation.collision_deform_meshes:
            return
        for entry in self.animation.collision_deform_meshes:
            if isinstance(entry, tuple):
                node_name, mesh_index = entry
            else:
                node_name, mesh_index = entry, None
            node = self.scene.find(node_name)
            if node is None:
                _log.warning(
                    "declarative: collision_deform mesh node %r not in scene; skipping",
                    node_name,
                )
                continue
            piece = self.cloth_host.add_cloth_for_node(
                node,
                mesh_index=mesh_index,
                cloth_name=f"collision_deform_{node_name}",
                anchor_axis=1,
                anchor_fraction=0.0,
                anchor_mode="top_axis",
                structural_stiffness=0.0,
                bend_stiffness=0.0,
                linear_damping=1.0,
                iterations=1,
                rest_pull=0.0,
            )
            if piece is None:
                continue
            piece.params.passive_skin_deform = True
            self.cloth_host.register_skin_target_follower(piece, node)

    def _resolve_collider_specs(self) -> list[ColliderSpec]:
        """Return the final collider list = explicit JSON + auto-emit body set.

        Explicit specs from ``animation.colliders`` always win on the
        ``(follow_bone, end_bone, kind)`` key, so a document that wants
        a custom hip-sphere radius gets exactly that without the engine
        appending its default one too. Auto-emit only fires when cloth
        pieces are registered AND the document didn't opt out via
        ``auto_body_colliders: false`` — keeps non-humanoid rigs out of
        the humanoid leg-collider machinery.
        """
        explicit = list(self.animation.colliders)
        if not (
            self.animation.cloth_pieces and self.animation.auto_body_colliders
        ):
            return explicit
        from posecascade.animation.auto_colliders import (  # noqa: PLC0415
            emit_humanoid_body_colliders,
        )
        aliases = getattr(self.scene, "bone_aliases", {}) or {}
        existing = {(s.follow_bone, s.end_bone, s.kind) for s in explicit}
        merged = list(explicit)
        for auto in emit_humanoid_body_colliders(
            aliases, spec_factory=ColliderSpec,
        ):
            key = (auto.follow_bone, auto.end_bone, auto.kind)
            if key not in existing:
                merged.append(auto)
                existing.add(key)
        return merged

    def _maybe_register_anchor_follower(
        self, spec: ClothPieceSpec, cloth_piece: Any,
    ) -> None:
        """Register the cloth piece's anchor verts to follow ``spec.track_bone``.

        Forwarded to :meth:`ClothHost.register_anchor_follower` — the host
        snaps each tracked piece's anchor ``positions`` / ``prev_positions``
        / ``rest_positions`` to the bone's current world frame at the
        start of every :meth:`ClothHost.tick`, so the waistband stays
        glued to a moving hip / shoulder bone instead of floating in
        space when the body translates.

        Skips silently when the spec has no track bone, the bone isn't in
        the scene, or the cloth host returned no piece (the latter happens
        for stub cloth hosts in tests).
        """
        if spec.track_bone is None or cloth_piece is None:
            return
        bone_node = self.scene.find(spec.track_bone)
        if bone_node is None:
            _log.warning(
                "declarative: cloth.track_bone %r not in scene; skipping",
                spec.track_bone,
            )
            return
        self.cloth_host.register_anchor_follower(cloth_piece, bone_node)

    def _update_bone_colliders(self) -> None:
        """Refresh each bone-following collider's geometry from its bones.

        Called each frame after gait + bones have written this frame's
        bone rotations, so the collider positions reflect the
        post-pose world transforms — meaning the cloth solver sees the
        arms / hands where they actually are this frame, not where
        they were on the previous tick.

        Stashes the previous frame's centre / endpoints onto the collider
        before overwriting them. The cloth solver's swept-capsule CCD reads
        those to project verts outside the volume the collider passed
        through, not just where it ended up — without this, a hand that
        moves more than its own radius between frames tunnels straight
        through the skirt.
        """
        if not self._bone_colliders:
            return
        for collider, kind, head_node, tail_node in self._bone_colliders:
            head_pos = _world_position(head_node).astype(np.float32)
            if kind == "sphere":
                collider.prev_center = collider.center.copy()
                collider.center = head_pos
            else:
                tail_pos = _world_position(tail_node).astype(np.float32)
                collider.prev_a = collider.a.copy()
                collider.prev_b = collider.b.copy()
                collider.a = head_pos
                collider.b = tail_pos

    def _setup_audio(self) -> None:
        """Optional audio-player attach + clock swap.

        Document with no audio block → no AudioPlayer instantiated, no
        Qt audio import, identical behaviour to pre-phase-8 docs. When
        present, the audio file path is resolved relative to the
        ``source_dir`` (the .json's parent), the player attempts a
        QtMultimedia attach (silently falls back to a wall-clock-only
        mode when the backend isn't available), and ``play()`` runs.
        With ``sync_clock: true`` the runtime's ``self.time`` is
        wrapped to return ``audio.current_time_seconds() - offset_sec``
        so phase scheduling drifts with the music's actual playback
        rate — same idea as the timeline dock's ``attach_audio`` flow.
        """
        spec = self.animation.audio
        if spec is None:
            return
        try:
            from pathlib import Path  # noqa: PLC0415

            from posecascade.audio.clip import load_wav_file  # noqa: PLC0415
            from posecascade.audio.player import AudioPlayer  # noqa: PLC0415
        except ImportError as err:
            _log.warning("declarative: audio module unavailable: %s", err)
            return
        path = Path(spec.path)
        if self.source_dir is not None and not path.is_absolute():
            path = Path(self.source_dir) / path
        try:
            clip = load_wav_file(path)
        except (PoseCascadeError, OSError) as err:
            _log.warning(
                "declarative: failed to load audio %s: %s", path, err,
            )
            return
        factory = self.audio_player_factory or AudioPlayer
        self._audio_player = factory(clip=clip)
        self._audio_player.attach_qt()  # best-effort; falls back to wall clock
        self._audio_player.play()
        if spec.sync_clock:
            # Capture the original time provider so a future on_event
            # ``reset`` could restore it; not in scope for this phase.
            self._wall_time = self.time
            offset = float(spec.offset_sec)
            player = self._audio_player

            def _audio_clock() -> float:
                return float(player.current_time_seconds()) - offset

            self.time = _audio_clock

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
        # Fresh per-frame parent-world cache. ``_parent_world_rotation``
        # populates it lazily; clearing here guarantees the very first
        # lookup of the frame walks the live transforms, and any
        # bone write within the same frame falls through the per-write
        # invalidation in ``_invalidate_parent_world_below``.
        self._frame_parent_world.clear()
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
        # Phase-explicit bones (from ``phase.bones``) override gait
        # directly — same rule as before. Skip identity deltas so a
        # zeroed-out axis curve doesn't snap the bone back to rest.
        for bone_key, body_delta in output.bones.items():
            if _is_identity_quat(body_delta):
                continue
            self._set_bone(bone_key, _yaw_to_world(body_delta, output.yaw))
        # Blender-style local basis bones. Applied AFTER ``bones`` so a
        # phase that mixes both forms gets the local-frame write last —
        # consistent with the "Blender output is the source of truth"
        # workflow these values come from.
        for bone_key, basis_quat in output.bones_local.items():
            if _is_identity_quat(basis_quat):
                continue
            self._set_bone_local(bone_key, basis_quat)
        # Pose blends slerp from gait's CURRENT bone rotation toward
        # the pose target by per-frame weight — produces a real "slow
        # rise from hanging to posed" because at low weight the arm is
        # still mostly at gait's hang. Cross-fade lerps the weight
        # naturally so a sway → reach transition rises smoothly without
        # passing through the rest pose.
        if output.pose_blends:
            self._apply_pose_blends(output.pose_blends, output.yaw)
        if output.morphs and self.morph_api is not None:
            for name, weight in output.morphs.items():
                self.morph_api.set(str(name), float(weight))
        # Hand IK runs AFTER every explicit / pose / gait bone write so
        # the IK solver sees the final upper-body pose and can plant
        # the wrist at the target world position regardless of what the
        # rest of the chain just did. Mirrors the foot-IK ordering.
        self._apply_hand_ik(phase, scope, phase_t)
        if phase.floor_align:
            self._apply_floor_align(phase.floor_align)
        self._post_bone_writes(output, elapsed)

    def _post_bone_writes(self, output: PhaseOutput, elapsed: float) -> None:
        """Side-effect wiring that runs once all per-frame bone writes are
        in: foot planter forward, camera lerp, lyric overlay, collider
        bone-tracking. Pulled out of ``_update`` to keep its branch
        count below the cyclomatic-complexity bound — each of the four
        independent feature gates was a separate branch there.
        """
        if self.floor_api is not None:
            self.floor_api.set_body_forward(
                (math.sin(output.yaw), 0.0, math.cos(output.yaw)),
            )
        if self.animation.camera_keys and self.camera_api is not None:
            self._apply_camera(elapsed)
        if self.animation.lyrics and self.overlay_api is not None:
            self._apply_lyrics(elapsed)
        if self._bone_colliders:
            self._update_bone_colliders()

    def _apply_camera(self, elapsed: float) -> None:
        """Lerp between bracketing camera keyframes and write to ``camera_api``.

        Position / target are 3-vector lerps; fov is a scalar lerp on
        ``Camera.fov_degrees`` only when both bracketing keyframes set
        a non-None fov (otherwise the camera's existing fov is left
        untouched). Before the first keyframe or after the last,
        snaps to the boundary keyframe's values — common pattern for
        "hold this composition before/after the animated section".
        """
        keys = self.animation.camera_keys
        if elapsed <= keys[0].at_sec:
            self._write_camera(keys[0])
            return
        if elapsed >= keys[-1].at_sec:
            self._write_camera(keys[-1])
            return
        # Bracket search — keys are pre-sorted by at_sec at parse time.
        for idx in range(len(keys) - 1):
            a = keys[idx]
            b = keys[idx + 1]
            if a.at_sec <= elapsed < b.at_sec:
                span = b.at_sec - a.at_sec
                t = (elapsed - a.at_sec) / span if span > 0 else 0.0
                self._blend_camera_keys(a, b, t)
                return

    def _apply_lyrics(self, elapsed: float) -> None:
        """Look up the active lyric for ``elapsed`` and push it to overlay.

        Lyrics are pre-sorted by start_sec; lines are non-overlapping
        in the typical karaoke case, but if two lines do overlap the
        FIRST one in the array wins (predictable for the author).
        Nothing-active resolves to the empty string so the overlay
        clears between lines. Last-text is cached so we only call
        ``overlay_api`` when the active text actually changes —
        avoids re-painting the viewport overlay every frame for
        sub-second flickers.
        """
        active = ""
        for line in self.animation.lyrics:
            if line.start_sec <= elapsed < line.end_sec:
                active = line.text
                break
            if line.start_sec > elapsed:
                break  # array is sorted; no point scanning further
        if active != self._last_lyric_text:
            self._last_lyric_text = active
            self.overlay_api(active)

    def _write_camera(self, key: CameraKey) -> None:
        self.camera_api.position = vec3(*key.position)
        self.camera_api.target = vec3(*key.target)
        if key.fov_degrees is not None:
            self.camera_api.fov_degrees = float(key.fov_degrees)

    def _blend_camera_keys(
        self, a: CameraKey, b: CameraKey, t: float,
    ) -> None:
        self.camera_api.position = vec3(
            *_lerp_translation(a.position, b.position, t),
        )
        self.camera_api.target = vec3(
            *_lerp_translation(a.target, b.target, t),
        )
        # Only lerp fov when both endpoints set it; otherwise leaving
        # the camera's existing value alone matches "fov keyframes are
        # optional" intent.
        if a.fov_degrees is not None and b.fov_degrees is not None:
            self.camera_api.fov_degrees = (
                a.fov_degrees + (b.fov_degrees - a.fov_degrees) * t
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

        Pose preset composition splits into two outputs:
        - ``pose_blends``: gait-aware path. The preset's full target
          rotation is stored per bone alongside the per-frame pose
          weight, so the runtime can slerp from gait's current rotation
          (e.g. arm hanging) toward the pose target. Same semantic for
          ``hand_L`` / ``hand_R`` (always full weight 1.0).
        - ``bones``: hard override path. Anything the phase declared
          explicitly under ``bones`` writes the rotation directly,
          overriding both gait and pose blend on a per-axis basis.
        """
        yaw = _resolve_value_curve(phase.body_yaw_rad, phase_t, scope)
        lean = _resolve_value_curve(phase.body_lean_x_rad, phase_t, scope)
        translation = _resolve_translation(phase.body_translation, phase_t, scope)
        output = PhaseOutput(yaw=yaw, lean=lean, translation=translation)
        self._fill_pose_blends(output, phase, scope, phase_t)
        self._fill_explicit_bones(output, phase, scope, phase_t)
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

    def _fill_pose_blends(
        self,
        output: PhaseOutput,
        phase: Phase,
        scope: dict[str, float],
        phase_t: float,
    ) -> None:
        """Compute target body-frame quaternions for all pose / hand
        preset bones plus their effective weight, into ``pose_blends``.

        Bones that the phase ALSO declared in ``phase.bones`` are
        excluded from pose_blends — the explicit-bone path always wins
        per-bone, so authors who want to override a single bone of a
        preset can do so without inheriting any of the preset's
        contribution to that bone.
        """
        explicit_bones = set(phase.bones)
        if phase.pose:
            preset = self.animation.pose_library.get(phase.pose)
            if preset is None:
                _log.warning(
                    "declarative: pose %r not in pose_library; skipping",
                    phase.pose,
                )
            else:
                weight = float(
                    _resolve_value_curve(phase.pose_weight, phase_t, scope),
                )
                _store_pose_targets(
                    output.pose_blends, preset, weight, exclude=explicit_bones,
                )
        for hand_field, side_label in (
            (phase.hand_l, "hand_L"), (phase.hand_r, "hand_R"),
        ):
            if not hand_field:
                continue
            hand_preset = self.animation.hand_library.get(hand_field)
            if hand_preset is None:
                _log.warning(
                    "declarative: %s preset %r not in hand_library; skipping",
                    side_label, hand_field,
                )
                continue
            _store_pose_targets(
                output.pose_blends, hand_preset, 1.0, exclude=explicit_bones,
            )

    def _fill_explicit_bones(
        self,
        output: PhaseOutput,
        phase: Phase,
        scope: dict[str, float],
        phase_t: float,
    ) -> None:
        """Evaluate ``phase.bones`` per-axis curves into ``output.bones``.

        Also evaluates ``phase.bones_local`` into ``output.bones_local``
        using intrinsic XYZ Euler (Blender convention) so the per-axis
        ``rotation_euler`` values exported from Blender drop straight in.
        """
        for bone_key, axes_spec in phase.bones.items():
            try:
                x = _resolve_value_curve(axes_spec.get("x_rad", 0.0), phase_t, scope)
                y = _resolve_value_curve(axes_spec.get("y_rad", 0.0), phase_t, scope)
                z = _resolve_value_curve(axes_spec.get("z_rad", 0.0), phase_t, scope)
            except DeclarativeAnimationError as err:
                _log.warning(
                    "declarative: bones[%r] failed to evaluate: %s", bone_key, err,
                )
                continue
            output.bones[bone_key] = _euler_zyx_quat(x, y, z)
        for bone_key, axes_spec in phase.bones_local.items():
            try:
                x = _resolve_value_curve(axes_spec.get("x_rad", 0.0), phase_t, scope)
                y = _resolve_value_curve(axes_spec.get("y_rad", 0.0), phase_t, scope)
                z = _resolve_value_curve(axes_spec.get("z_rad", 0.0), phase_t, scope)
            except DeclarativeAnimationError as err:
                _log.warning(
                    "declarative: bones_local[%r] failed to evaluate: %s", bone_key, err,
                )
                continue
            output.bones_local[bone_key] = _euler_xyz_intrinsic_quat(x, y, z)


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

    def _apply_hand_ik(
        self, phase: Phase, scope: dict[str, float], phase_t: float,
    ) -> None:
        """Solve analytical 2-bone IK on each arm AND leg chain that has a target.

        Uses :func:`solve_two_bone_analytic` — closed-form law-of-cosines.
        Arm chain runs against ``ik.hand_l_target`` / ``hand_r_target``
        (resolved via ``rig.arm_chain_l/r``), leg chain runs against
        ``ik.foot_l_target`` / ``foot_r_target`` (via ``rig.leg_chain_l/r``).

        Foot IK is what fixes the 'foot mesh visibly squashed against
        the floor' bug: when a pose drives the leg via bones_local
        alone, the ankle bone usually ends up underground (the chain
        can't naturally land at Y=0 from a deep knee fold). The cloth
        floor clamp then pancakes the foot mesh to the floor plane.
        Driving the ankle via IK puts the bone AT the floor target so
        the foot mesh sits naturally on top instead.

        ``body_forward_world`` (taken from the foot planter when
        available) serves as the bend hint so knees fold forward and
        elbows fold backward along the body's facing plane.
        """
        try:
            from posecascade.animation.ik import (  # noqa: PLC0415
                solve_two_bone_analytic,
            )
        except ImportError:
            return
        bend_hint = vec3(0.0, 0.0, -1.0)
        if self.floor_api is not None:
            planter = getattr(self.floor_api, "_planter", None)
            if planter is not None:
                bend_hint = np.asarray(
                    planter.body_forward_world, dtype=np.float32,
                )
        # Arms.
        for side, target_spec in (
            ("L", phase.hand_l_target), ("R", phase.hand_r_target),
        ):
            if target_spec is None:
                continue
            chain = self._arm_chain_nodes.get(side)
            if chain is None:
                continue
            target = self._resolve_ik_target(
                target_spec, f"hand_{side.lower()}_target", phase_t, scope,
            )
            if target is None:
                continue
            root, mid, end = chain
            solve_two_bone_analytic(root, mid, end, target, bend_hint=bend_hint)
        # Legs. Knees fold the OTHER way relative to elbows; flip the
        # bend hint Z so a body facing -Z gets knee bend in +Z (forward).
        leg_bend = (-bend_hint).astype(np.float32, copy=False)
        for side, target_spec in (
            ("L", phase.foot_l_target), ("R", phase.foot_r_target),
        ):
            if target_spec is None:
                continue
            chain = self._leg_chain_nodes.get(side)
            if chain is None:
                continue
            target = self._resolve_ik_target(
                target_spec, f"foot_{side.lower()}_target", phase_t, scope,
            )
            if target is None:
                continue
            root, mid, end = chain
            solve_two_bone_analytic(root, mid, end, target, bend_hint=leg_bend)

    def _resolve_ik_target(
        self,
        target_spec: tuple[Any, Any, Any],
        field_name: str,
        phase_t: float,
        scope: dict[str, float],
    ) -> np.ndarray | None:
        """Resolve a 3-tuple of value curves to a world-space target vec3."""
        try:
            tx = _resolve_value_curve(target_spec[0], phase_t, scope)
            ty = _resolve_value_curve(target_spec[1], phase_t, scope)
            tz = _resolve_value_curve(target_spec[2], phase_t, scope)
        except DeclarativeAnimationError as err:
            _log.warning(
                "declarative: %s failed to evaluate: %s", field_name, err,
            )
            return None
        return np.array((tx, ty, tz), dtype=np.float32)

    def _apply_floor_align(self, bone_keys: tuple[str, ...]) -> None:
        """Full 2-axis orientation lock: contact-normal down + bone-axis horizontal.

        Each bone's rotation is set so that:
        - the rest 'contact normal' local direction (palm normal for
          wrist, sole normal for ankle) points world -Y, AND
        - the local 'bone direction' (toward first child = toe / fingertip)
          points along the chain's extension in the horizontal plane
          (knee→ankle for foot, elbow→wrist for hand).

        Result: foot lies flat with heel-and-toe both on the floor,
        toe pointing along the leg's extension direction; hand lies
        flat with fingers pointing along the forearm's extension
        direction. No more 'heel lifted' artifact from single-axis
        alignment leaving the bone's twist free.

        Falls back to single-axis alignment when chain parent or
        bone direction couldn't be captured (e.g. a non-IK bone the
        user added to floor_align manually).
        """
        try:
            from posecascade.animation.ik import (  # noqa: PLC0415
                align_bone_axis_to_world,
                align_bone_two_axes_to_world,
            )
        except ImportError:
            return
        aliases = self.animation.rig.body_bones
        for bone_key in bone_keys:
            bone_name = aliases.get(bone_key, bone_key)
            node = self.scene.find(bone_name)
            if node is None:
                _log.warning(
                    "declarative: floor_align bone %r not in scene; skipping",
                    bone_key,
                )
                continue
            contact_local = self._floor_align_rest_world.get(id(node))
            if contact_local is None:
                continue
            bone_dir_local = self._floor_align_bone_dir_local.get(id(node))
            parent_joint = self._floor_align_chain_parent.get(id(node))
            if bone_dir_local is None or parent_joint is None:
                align_bone_axis_to_world(node, tuple(contact_local), (0.0, -1.0, 0.0))
                continue
            # World extension direction = (bone pos - parent_joint pos),
            # flattened to the horizontal plane (foot lies along
            # ground, not tilted up toward shin).
            bone_pos = _world_position(node)
            parent_pos = _world_position(parent_joint)
            extension = bone_pos - parent_pos
            extension[1] = 0.0  # horizontal only
            ext_norm = float(np.linalg.norm(extension))
            if ext_norm < 1.0e-6:  # noqa: PLR2004
                align_bone_axis_to_world(node, tuple(contact_local), (0.0, -1.0, 0.0))
                continue
            extension = extension / ext_norm
            from posecascade.animation.ik import (  # noqa: PLC0415
                _rotate_vec_by_quat,
                _world_rotation,
            )
            cur_world_rot = _world_rotation(node)
            cur_bone_dir = _rotate_vec_by_quat(
                cur_world_rot, np.asarray(bone_dir_local, dtype=np.float64),
            )
            cur_norm = float(np.linalg.norm(cur_bone_dir))
            if cur_norm > 1.0e-6:  # noqa: PLR2004
                cur_bone_dir = cur_bone_dir / cur_norm
                if float(np.dot(cur_bone_dir, extension)) < 0.0:
                    align_bone_axis_to_world(
                        node, tuple(contact_local), (0.0, -1.0, 0.0),
                    )
                    continue
            align_bone_two_axes_to_world(
                node,
                tuple(bone_dir_local), tuple(extension),
                tuple(contact_local), (0.0, -1.0, 0.0),
            )

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

        Some rigs (Sketchfab FBX rips, March 7th) ship with shoulder
        rest rotations that ALREADY hang the arms at a natural A-pose.
        On those rigs, set ``arm_hang_rad: 0.0`` in the gait — the rest
        pose is the desired hang, no additional Z rotation needed. The
        side-flipping convention here assumes T-pose; with both rests
        identical, ``side * arm_hang`` introduces visible asymmetry.
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
        parent_world = self._parent_world_rotation(drive.node)
        parent_world_inv = quat_inverse(parent_world)
        delta_local = quat_mul(
            quat_mul(parent_world_inv, delta_world),
            parent_world,
        )
        drive.node.transform.set_rotation(
            quat_mul(delta_local, drive.rest_rotation).astype(np.float32, copy=False),
        )
        self._invalidate_parent_world_for(drive.node)

    def _set_bone_local(self, bone_key: str, basis_quat: np.ndarray) -> None:
        """Apply a Blender-style local basis rotation directly.

        Where :meth:`_set_bone` interprets its input as a WORLD-frame
        delta and conjugates through the parent chain, this method takes
        a rotation already expressed in the bone's local basis frame —
        the same value Blender stores in ``pose_bone.matrix_basis`` /
        ``rotation_quaternion``. Composition is just
        ``rotation = basis_quat * rest_rotation``, no conjugation; the
        parent's posed rotation is handled by the scene graph's normal
        matrix composition at evaluation time.

        Use this for poses authored in Blender (or any DCC where the
        natural per-bone rotation is local-frame). For programmatically
        composed world-frame deltas (gait, IK solvers), use ``_set_bone``.
        """
        bone_name = self.animation.rig.body_bones.get(bone_key, bone_key)
        drive = self._bone_drives.get(bone_name)
        if drive is None:
            return
        drive.node.transform.set_rotation(
            quat_mul(basis_quat, drive.rest_rotation).astype(np.float32, copy=False),
        )
        self._invalidate_parent_world_for(drive.node)

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


def _phase_target_bones(
    phase: Phase,
    rig: RigBindings,
    pose_library: dict[str, PoseSpec] | None = None,
    hand_library: dict[str, PoseSpec] | None = None,
) -> tuple[str, ...]:
    """All bone names a phase needs rest rotations cached for.

    Union of the gait's reset-set, any bone keyed in ``phase.bones``,
    and any bone driven by the phase's body / hand presets (when the
    presets are present in the document's libraries). All go through
    the rig's alias map so authors can rename bones in one place
    without touching every phase.
    """
    names: set[str] = set()
    if phase.gait is not None:
        names.update(_gait_target_bones(phase.gait, rig))
    for bone_key in phase.bones:
        names.add(rig.body_bones.get(bone_key, bone_key))
    for bone_key in phase.bones_local:
        names.add(rig.body_bones.get(bone_key, bone_key))
    if phase.pose and pose_library is not None:
        preset = pose_library.get(phase.pose)
        if preset is not None:
            for bone_key in preset:
                names.add(rig.body_bones.get(bone_key, bone_key))
    for hand_name in (phase.hand_l, phase.hand_r):
        if hand_name and hand_library is not None:
            hand_preset = hand_library.get(hand_name)
            if hand_preset is not None:
                for bone_key in hand_preset:
                    names.add(rig.body_bones.get(bone_key, bone_key))
    return tuple(names)


def _store_pose_targets(
    pose_blends: dict[str, tuple[np.ndarray, float]],
    preset: PoseSpec,
    weight: float,
    *,
    exclude: set[str] | None = None,
) -> None:
    """Stash each preset bone's full-target body-frame quaternion + the
    per-frame weight into ``pose_blends``.

    The runtime applies these by slerping from the bone's current
    rotation (which already has the gait applied) toward the target by
    the weight — so weight 0 leaves gait alone, weight 1 fully poses
    the bone, in between produces a real interpolation between hanging
    and posed without ever passing through the rest pose. Multiple
    presets writing the same bone (body pose + hand pose for fingers
    on the same arm — rare but possible) keep the LAST one written;
    this matches the previous merge order (body pose → hand_L → hand_R).

    ``exclude`` is the set of bone keys the phase already drives via
    its explicit ``phase.bones``; those bones skip the pose path so the
    explicit override is the sole writer for them.
    """
    excl = exclude or set()
    for bone_key, axes in preset.items():
        if str(bone_key) in excl:
            continue
        target = _euler_zyx_quat(
            float(axes.get("x_rad", 0.0)),
            float(axes.get("y_rad", 0.0)),
            float(axes.get("z_rad", 0.0)),
        )
        pose_blends[str(bone_key)] = (target, float(weight))


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


def _euler_xyz_intrinsic_quat(x: float, y: float, z: float) -> np.ndarray:
    """Compose ``Rx · Ry · Rz`` as a single quaternion.

    Intrinsic XYZ Euler (== extrinsic ZYX). Matches Blender's default
    ``rotation_mode='XYZ'`` so values pulled from
    ``pose_bone.rotation_euler`` or ``Matrix.to_euler('XYZ')`` can be
    written straight into the ``bones_local`` block and reproduce the
    same rotation in PoseCascade's runtime. Used only by
    :meth:`DeclarativeRuntime._set_bone_local`.
    """
    qx = quat_axis_angle(vec3(1.0, 0.0, 0.0), x)
    qy = quat_axis_angle(vec3(0.0, 1.0, 0.0), y)
    qz = quat_axis_angle(vec3(0.0, 0.0, 1.0), z)
    return quat_mul(qx, quat_mul(qy, qz))


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


def _world_matrix(node: Any) -> np.ndarray:
    """Compose ``node``'s full parent chain into a 4x4 world matrix.

    Mirrors ``_world_position`` but returns the entire transform —
    needed by cloth anchor tracking which has to rotate / translate
    per-vertex offsets, not just read an origin.
    """
    matrix = node.transform.to_matrix()
    parent = node.parent
    while parent is not None:
        matrix = parent.transform.to_matrix() @ matrix
        parent = parent.parent
    return np.asarray(matrix, dtype=np.float32)


def _world_position(node: Any) -> np.ndarray:
    """Compose ``node``'s parent chain to get its world-space origin."""
    matrix = _world_matrix(node)
    return np.array([matrix[0, 3], matrix[1, 3], matrix[2, 3]], dtype=np.float64)


def _resolve_translation(
    spec: Any, phase_t: float, scope: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    # Shorthand: ``[x, y, z]`` array form. Each element is itself a
    # value curve (number, expression string, [from,to], or dict), so
    # authors who want a static origin can write ``[0, 0, 0]`` and
    # those who want one axis moving can write
    # ``[0, "0.005 * sin(...)", 0]``.
    if isinstance(spec, list):
        if len(spec) != _VEC3_LEN:
            raise DeclarativeAnimationError(
                f"body.translation array must have 3 entries (x, y, z), "
                f"got {len(spec)}",
            )
        return (
            _resolve_value_curve(spec[0], phase_t, scope),
            _resolve_value_curve(spec[1], phase_t, scope),
            _resolve_value_curve(spec[2], phase_t, scope),
        )
    if not isinstance(spec, dict):
        raise DeclarativeAnimationError(
            f"body.translation must be a dict or [x, y, z] array, "
            f"got {type(spec).__name__}",
        )
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
    source_dir = Path(filename).resolve().parent if filename else None
    document = resolve_extends(document, source_dir)
    parsed = parse_animation(document)
    runtime = DeclarativeRuntime(
        animation=parsed,
        scene=api["scene"],
        time=api["time"],
        floor_api=api.get("floor"),
        physics_lite=api.get("physics_lite"),
        morph_api=api.get("morphs"),
        camera_api=api.get("camera"),
        overlay_api=api.get("overlay"),
        cloth_host=api.get("cloth_host"),
        source_dir=source_dir,
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
    "resolve_extends",
]
