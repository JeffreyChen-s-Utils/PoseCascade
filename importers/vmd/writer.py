"""VMD binary writer — the inverse of :mod:`vmd.reader`.

Round-trip with :func:`vmd.reader.parse_vmd` is value-identical for the
fields the format defines: every record reproduces the same field
values it parsed from. The 64-byte bone bezier block's trailing 48
"scratchpad" bytes are zero-filled here (MMD itself repeats the
leading 16 bytes in three quirky permutations there; nothing we ship
ever reads them, so leaving them zero keeps the writer simple while
staying compatible with everyone else's parser).
"""
from __future__ import annotations

import struct

from vmd.types import (
    VmdBoneKeyframe,
    VmdCameraKeyframe,
    VmdHeader,
    VmdIkKeyframe,
    VmdLightKeyframe,
    VmdMorphKeyframe,
    VmdMotion,
    VmdSelfShadowKeyframe,
)

_HEADER_BYTES = 30
_MODEL_NAME_BYTES = 20
_BONE_NAME_BYTES = 15
_MORPH_NAME_BYTES = 15
_IK_BONE_NAME_BYTES = 20
_BONE_BEZIER_BYTES = 64
_BONE_BEZIER_CHANNELS = 4    # X / Y / Z position + rotation
_CAMERA_BEZIER_BYTES = 24
_CAMERA_BEZIER_CHANNELS = 6  # X / Y / Z position + rotation + distance + FOV


def serialize_vmd(motion: VmdMotion) -> bytes:
    """Serialise a :class:`VmdMotion` to its on-disk byte form."""
    parts: list[bytes] = [
        _encode_header(motion.header),
        _encode_bone_keyframes(motion.bone_keyframes),
        _encode_morph_keyframes(motion.morph_keyframes),
        _encode_camera_keyframes(motion.camera_keyframes),
        _encode_light_keyframes(motion.light_keyframes),
        _encode_self_shadow_keyframes(motion.self_shadow_keyframes),
        _encode_ik_keyframes(motion.ik_keyframes),
    ]
    return b"".join(parts)


# ----- header -----------------------------------------------------------
def _encode_header(header: VmdHeader) -> bytes:
    return _encode_padded_sjis(header.signature, _HEADER_BYTES) + _encode_padded_sjis(
        header.model_name, _MODEL_NAME_BYTES,
    )


def _encode_padded_sjis(text: str, byte_count: int) -> bytes:
    raw = text.encode("cp932", errors="replace")
    if len(raw) >= byte_count:
        return raw[:byte_count]
    return raw + b"\x00" * (byte_count - len(raw))


# ----- bone keyframes --------------------------------------------------
def _encode_bone_keyframes(keyframes: tuple[VmdBoneKeyframe, ...]) -> bytes:
    out = [struct.pack("<I", len(keyframes))]
    for kf in keyframes:
        out.append(_encode_padded_sjis(kf.bone_name, _BONE_NAME_BYTES))
        out.append(struct.pack("<I", int(kf.frame)))
        out.append(struct.pack("<fff", *kf.position))
        out.append(struct.pack("<ffff", *kf.rotation))
        out.append(_encode_bone_bezier(kf.bezier_handles))
    return b"".join(out)


def _encode_bone_bezier(
    handles: tuple[tuple[int, int, int, int], ...],
) -> bytes:
    """Pack ``(4, 4)`` channel handles into the 64-byte VMD block.

    Layout matches :func:`vmd.reader._decode_bone_bezier`: channel ``c``'s
    ``(X1, Y1, X2, Y2)`` lands at bytes ``c, 4+c, 8+c, 12+c``. The trailing
    48 bytes of the block are MMD-internal scratch space; we zero-fill.
    """
    block = bytearray(_BONE_BEZIER_BYTES)
    for channel, row in enumerate(handles):
        if channel >= _BONE_BEZIER_CHANNELS:
            break
        block[channel + 0] = int(row[0]) & 0xFF
        block[channel + 4] = int(row[1]) & 0xFF
        block[channel + 8] = int(row[2]) & 0xFF
        block[channel + 12] = int(row[3]) & 0xFF
    return bytes(block)


# ----- morph keyframes -------------------------------------------------
def _encode_morph_keyframes(keyframes: tuple[VmdMorphKeyframe, ...]) -> bytes:
    out = [struct.pack("<I", len(keyframes))]
    for kf in keyframes:
        out.append(_encode_padded_sjis(kf.morph_name, _MORPH_NAME_BYTES))
        out.append(struct.pack("<I", int(kf.frame)))
        out.append(struct.pack("<f", float(kf.weight)))
    return b"".join(out)


# ----- camera keyframes ------------------------------------------------
def _encode_camera_keyframes(keyframes: tuple[VmdCameraKeyframe, ...]) -> bytes:
    out = [struct.pack("<I", len(keyframes))]
    for kf in keyframes:
        out.append(struct.pack("<I", int(kf.frame)))
        out.append(struct.pack("<f", float(kf.distance)))
        out.append(struct.pack("<fff", *kf.target))
        out.append(struct.pack("<fff", *kf.rotation))
        out.append(_encode_camera_bezier(kf.bezier_handles))
        out.append(struct.pack("<I", int(kf.fov_degrees)))
        out.append(bytes([1 if kf.perspective_off else 0]))
    return b"".join(out)


def _encode_camera_bezier(
    handles: tuple[tuple[int, int, int, int], ...],
) -> bytes:
    block = bytearray(_CAMERA_BEZIER_BYTES)
    for channel, row in enumerate(handles):
        if channel >= _CAMERA_BEZIER_CHANNELS:
            break
        block[channel * 4 + 0] = int(row[0]) & 0xFF
        block[channel * 4 + 1] = int(row[1]) & 0xFF
        block[channel * 4 + 2] = int(row[2]) & 0xFF
        block[channel * 4 + 3] = int(row[3]) & 0xFF
    return bytes(block)


# ----- light keyframes -------------------------------------------------
def _encode_light_keyframes(keyframes: tuple[VmdLightKeyframe, ...]) -> bytes:
    out = [struct.pack("<I", len(keyframes))]
    for kf in keyframes:
        out.append(struct.pack("<I", int(kf.frame)))
        out.append(struct.pack("<fff", *kf.color))
        out.append(struct.pack("<fff", *kf.direction))
    return b"".join(out)


# ----- self-shadow keyframes ------------------------------------------
def _encode_self_shadow_keyframes(
    keyframes: tuple[VmdSelfShadowKeyframe, ...],
) -> bytes:
    out = [struct.pack("<I", len(keyframes))]
    for kf in keyframes:
        out.append(struct.pack("<I", int(kf.frame)))
        out.append(bytes([int(kf.mode) & 0xFF]))
        out.append(struct.pack("<f", float(kf.distance)))
    return b"".join(out)


# ----- IK keyframes ----------------------------------------------------
def _encode_ik_keyframes(keyframes: tuple[VmdIkKeyframe, ...]) -> bytes:
    out = [struct.pack("<I", len(keyframes))]
    for kf in keyframes:
        out.append(struct.pack("<I", int(kf.frame)))
        out.append(bytes([1 if kf.visible else 0]))
        out.append(struct.pack("<I", len(kf.switches)))
        for switch in kf.switches:
            out.append(_encode_padded_sjis(switch.bone_name, _IK_BONE_NAME_BYTES))
            out.append(bytes([1 if switch.enabled else 0]))
    return b"".join(out)
