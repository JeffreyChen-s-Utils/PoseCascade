"""Format-internal types for VMD (Vocaloid Motion Data) documents.

VMD stores per-bone, per-morph, per-camera, etc. keyframes for an MMD
animation. The reader produces these dataclasses; the importer adapter
in :mod:`vmd.importer` consumes them and emits engine-facing
:class:`~posecascade.animation.vmd_track.VmdMotion` objects.

Naming follows the de-facto MMD community spec — every field maps directly
to a documented byte field in the file format so a parser bug surfaces as
an obvious cross-reference mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

# VMD's frame counter is fixed at 30 fps; helper here so callers don't keep
# sprinkling magic numbers across the codebase.
VMD_FRAMES_PER_SECOND = 30


class VmdSelfShadowMode(IntEnum):
    OFF = 0
    ON = 1
    FULL = 2


@dataclass(frozen=True)
class VmdHeader:
    """Decoded VMD header.

    The file's 30-byte signature is one of two strings depending on the
    writer version; we keep the raw value for diagnostics and the
    associated ``model_name`` (also SJIS) for round-trip fidelity.
    """

    signature: str
    model_name: str


@dataclass(frozen=True)
class VmdBoneKeyframe:
    """One bone keyframe.

    Bezier handles are stored as a ``(4, 4)`` ``uint8`` array; the four
    rows are the X/Y/Z/R channels and the four columns are
    ``(X1, Y1, X2, Y2)`` ∈ ``[0, 127]``. The handles describe the curve
    *entering* this keyframe (i.e., from the previous keyframe to this
    one); keyframe 0 ignores them.
    """

    bone_name: str
    frame: int
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]   # quat xyzw
    bezier_handles: tuple[tuple[int, int, int, int], ...]   # (4, 4) uint8


@dataclass(frozen=True)
class VmdMorphKeyframe:
    morph_name: str
    frame: int
    weight: float


@dataclass(frozen=True)
class VmdCameraKeyframe:
    """VMD camera record. ``rotation`` is Euler XYZ in radians; ``distance``
    is positive for "behind the target" and may be negative for the
    cinema-trick rear-view convention."""

    frame: int
    distance: float
    target: tuple[float, float, float]
    rotation: tuple[float, float, float]
    bezier_handles: tuple[tuple[int, int, int, int], ...]   # (6, 4) for X/Y/Z/R/Dist/FOV
    fov_degrees: int
    perspective_off: bool


@dataclass(frozen=True)
class VmdLightKeyframe:
    frame: int
    color: tuple[float, float, float]
    direction: tuple[float, float, float]


@dataclass(frozen=True)
class VmdSelfShadowKeyframe:
    frame: int
    mode: VmdSelfShadowMode
    distance: float


@dataclass(frozen=True)
class VmdIkSwitch:
    """One bone's IK enable/disable state inside a VMD ``IK`` segment."""

    bone_name: str
    enabled: bool


@dataclass(frozen=True)
class VmdIkKeyframe:
    frame: int
    visible: bool
    switches: tuple[VmdIkSwitch, ...]


@dataclass(frozen=True)
class VmdMotion:
    """The whole VMD document — one keyframe stream per kind, sorted by frame."""

    header: VmdHeader
    bone_keyframes: tuple[VmdBoneKeyframe, ...] = field(default_factory=tuple)
    morph_keyframes: tuple[VmdMorphKeyframe, ...] = field(default_factory=tuple)
    camera_keyframes: tuple[VmdCameraKeyframe, ...] = field(default_factory=tuple)
    light_keyframes: tuple[VmdLightKeyframe, ...] = field(default_factory=tuple)
    self_shadow_keyframes: tuple[VmdSelfShadowKeyframe, ...] = field(default_factory=tuple)
    ik_keyframes: tuple[VmdIkKeyframe, ...] = field(default_factory=tuple)
