"""Built-in finger / hand presets for the declarative animation runtime.

Each preset maps ``bone_key → {x_rad?, y_rad?, z_rad?: float}`` over a
VRoid-style hand bone vocabulary: ``J_Bip_{L,R}_{Thumb,Index,Middle,
Ring,Little}{1,2,3}``. Authors reference a preset on a phase via
``hand_L: "name"`` / ``hand_R: "name"`` and the runtime composes it
into the bones output the same way body poses do (per-axis override
by phase-level ``bones``).

Sided presets are stored separately because the per-finger axis values
are typically mirrored across hands. ``LEFT`` and ``RIGHT`` collapse
into one library at module load via :func:`build_hand_library` so the
runtime sees a single name space (``peace_L``, ``peace_R``, ...).

Curl convention: each finger joint uses ``x_rad`` for "curl toward
palm" — VRoid's finger bone local X is the hinge axis. Magnitude
~1.4 rad ≈ ~80 ° corresponds to a fully closed fist on the bundled
character; smaller values (~0.6–0.8) read as a relaxed grip. The
thumb's natural axis differs slightly so the closed-thumb preset
also rotates around z to bring the thumb across the palm.
"""
from __future__ import annotations

PoseSpec = dict[str, dict[str, float]]

# Per-finger curl magnitudes (positive curls toward the palm on
# VRoid's bone axis). Keep below ~1.5 to stay within the rig's hinge
# limit — finger bones don't usually carry knee_limit-style clamps,
# but going past π/2 starts producing clipping.
_CURL_FULL = 1.4
_CURL_MED = 0.9
_CURL_LIGHT = 0.4
# Thumb-across-palm sweep. Thumb's local axes are orientated
# differently from the four fingers, so the "curl in" rotation is
# split across X (knuckle bend) and Z (sweep across palm).
_THUMB_TUCK_X = 0.6
_THUMB_TUCK_Z = -0.7


def _finger_chain(side: str, finger: str, curl: float) -> dict[str, dict[str, float]]:
    """Three-segment curl for one non-thumb finger."""
    return {
        f"J_Bip_{side}_{finger}1": {"x_rad": curl},
        f"J_Bip_{side}_{finger}2": {"x_rad": curl},
        f"J_Bip_{side}_{finger}3": {"x_rad": curl},
    }


def _thumb(side: str, *, curl: float = 0.0, tuck: bool = False) -> dict[str, dict[str, float]]:
    """Thumb posture — straight when curl=0, partially across-palm when tuck."""
    bones: dict[str, dict[str, float]] = {
        f"J_Bip_{side}_Thumb1": {},
        f"J_Bip_{side}_Thumb2": {"x_rad": curl},
        f"J_Bip_{side}_Thumb3": {"x_rad": curl},
    }
    if tuck:
        # Mirror the Z sweep across hands so both thumbs tuck inward.
        z_sign = -1.0 if side == "L" else 1.0
        bones[f"J_Bip_{side}_Thumb1"] = {
            "x_rad": _THUMB_TUCK_X,
            "z_rad": _THUMB_TUCK_Z * z_sign,
        }
    return bones


def _peace(side: str) -> PoseSpec:
    """Index + Middle extended; Ring + Little + Thumb curled."""
    bones: PoseSpec = {}
    bones.update(_finger_chain(side, "Index", 0.0))
    bones.update(_finger_chain(side, "Middle", 0.0))
    bones.update(_finger_chain(side, "Ring", _CURL_FULL))
    bones.update(_finger_chain(side, "Little", _CURL_FULL))
    bones.update(_thumb(side, curl=_CURL_MED, tuck=True))
    return bones


def _fist(side: str) -> PoseSpec:
    bones: PoseSpec = {}
    for finger in ("Index", "Middle", "Ring", "Little"):
        bones.update(_finger_chain(side, finger, _CURL_FULL))
    bones.update(_thumb(side, curl=_CURL_FULL, tuck=True))
    return bones


def _point(side: str) -> PoseSpec:
    """Index extended; everything else curled. The classic 'point'."""
    bones: PoseSpec = {}
    bones.update(_finger_chain(side, "Index", 0.0))
    for finger in ("Middle", "Ring", "Little"):
        bones.update(_finger_chain(side, finger, _CURL_FULL))
    bones.update(_thumb(side, curl=_CURL_MED, tuck=True))
    return bones


def _open_palm(side: str) -> PoseSpec:
    """All fingers extended with a slight relaxed curl — the rest pose
    most rigs ship with already, but explicit so authors can transition
    'fist → open_palm' for a wave / clap accent."""
    bones: PoseSpec = {}
    for finger in ("Index", "Middle", "Ring", "Little"):
        bones.update(_finger_chain(side, finger, _CURL_LIGHT))
    bones.update(_thumb(side, curl=_CURL_LIGHT))
    return bones


def _thumbs_up(side: str) -> PoseSpec:
    """Thumb extended up, all four fingers curled."""
    bones: PoseSpec = {}
    for finger in ("Index", "Middle", "Ring", "Little"):
        bones.update(_finger_chain(side, finger, _CURL_FULL))
    # Thumb1 explicitly NOT tucked — extended along its rest axis.
    bones[f"J_Bip_{side}_Thumb1"] = {}
    bones[f"J_Bip_{side}_Thumb2"] = {}
    bones[f"J_Bip_{side}_Thumb3"] = {}
    return bones


def build_hand_library() -> dict[str, PoseSpec]:
    """Materialise the L / R variants of every built-in hand preset."""
    builders = {
        "peace": _peace,
        "fist": _fist,
        "point": _point,
        "open_palm": _open_palm,
        "thumbs_up": _thumbs_up,
    }
    out: dict[str, PoseSpec] = {}
    for name, builder in builders.items():
        out[f"{name}_L"] = builder("L")
        out[f"{name}_R"] = builder("R")
    return out


BUILTIN_HANDS: dict[str, PoseSpec] = build_hand_library()


def merge_libraries(user: dict[str, PoseSpec] | None) -> dict[str, PoseSpec]:
    """Overlay a user-supplied hand library on the built-ins.

    Same precedence rule as :mod:`pose_library`: user entries with the
    same name as a built-in win wholesale.
    """
    merged: dict[str, PoseSpec] = {
        name: {bone: dict(axes) for bone, axes in spec.items()}
        for name, spec in BUILTIN_HANDS.items()
    }
    if user:
        for name, spec in user.items():
            merged[str(name)] = {
                str(bone): {str(axis): float(v) for axis, v in axes.items()}
                for bone, axes in spec.items()
            }
    return merged


__all__ = ["BUILTIN_HANDS", "PoseSpec", "build_hand_library", "merge_libraries"]
