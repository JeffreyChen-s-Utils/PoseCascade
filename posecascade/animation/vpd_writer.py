"""VPD pose serialiser.

Mirror of :mod:`vpd.reader` — given a :class:`~vpd.types.VpdPose`, emit
the canonical SJIS-encodable text. Number formatting matches MMD's own
writer (six fractional digits) so a hand-edited file round-trips
through ``parse → serialize → parse`` without value drift.
"""
from __future__ import annotations

from vpd.types import VpdBoneOverride, VpdMorphOverride, VpdPose

_HEADER = "Vocaloid Pose Data file"
# MMD writes coordinates with six fractional digits — keeps small values
# like a single-axis rotation legible while staying lossless for any
# float32 that originated in a model file.
_FLOAT_FORMAT = "{:.6f}"


def serialize_vpd(pose: VpdPose) -> str:
    """Serialise ``pose`` to a VPD-formatted string (LF newlines)."""
    lines: list[str] = [_HEADER, ""]
    lines.append(f"{pose.model_name};")
    lines.append(f"{len(pose.bones)};")
    lines.append("")
    for index, bone in enumerate(pose.bones):
        lines.extend(_format_bone_block(index, bone))
        lines.append("")
    for index, morph in enumerate(pose.morphs):
        lines.extend(_format_morph_block(index, morph))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def serialize_vpd_bytes(pose: VpdPose) -> bytes:
    """Serialise to SJIS bytes — VPD's canonical on-disk encoding."""
    return serialize_vpd(pose).encode("cp932", errors="replace")


def _format_bone_block(index: int, bone: VpdBoneOverride) -> list[str]:
    return [
        f"Bone{index}{{{bone.name}",
        "  " + _format_vec3(bone.translation) + ";",
        "  " + _format_vec4(bone.rotation) + ";",
        "}",
    ]


def _format_morph_block(index: int, morph: VpdMorphOverride) -> list[str]:
    return [
        f"Morph{index}{{{morph.name}",
        "  " + _FLOAT_FORMAT.format(morph.weight) + ";",
        "}",
    ]


def _format_vec3(values: tuple[float, float, float]) -> str:
    return ",".join(_FLOAT_FORMAT.format(v) for v in values)


def _format_vec4(values: tuple[float, float, float, float]) -> str:
    return ",".join(_FLOAT_FORMAT.format(v) for v in values)
