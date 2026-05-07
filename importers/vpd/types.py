"""Format-internal types for VPD (Vocaloid Pose Data) documents.

VPD is a single-frame snapshot — a list of bone overrides (translation +
rotation) and (optionally, in newer files) a list of morph weights. The
importer adapter normalises the raw text records into engine-friendly
dictionaries that :mod:`posecascade.animation.vpd_apply` can drop onto
a scene's bone Nodes and morph state.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VpdBoneOverride:
    """One bone's translation + rotation override.

    ``translation`` is an offset relative to the bone's rest pose (mirrors
    VMD's convention so a VPD applied at frame 0 of a VMD timeline doesn't
    fight the rest-pose snapshot).
    """

    name: str
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]   # quaternion xyzw


@dataclass(frozen=True)
class VpdMorphOverride:
    name: str
    weight: float


@dataclass(frozen=True)
class VpdPose:
    """The whole pose: model-name reference + bone / morph overrides."""

    model_name: str
    bones: tuple[VpdBoneOverride, ...] = field(default_factory=tuple)
    morphs: tuple[VpdMorphOverride, ...] = field(default_factory=tuple)
