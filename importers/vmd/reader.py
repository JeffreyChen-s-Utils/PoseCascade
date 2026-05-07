"""VMD (Vocaloid Motion Data) binary parser.

Walks a ``.vmd`` byte buffer into :class:`vmd.types.VmdMotion`. Every
section after the header is independently length-prefixed, so an older
file that stops after the bone keyframes (no morph/camera/etc.) parses
cleanly — we only consume sections that actually exist in the buffer.

Text fields throughout VMD are Shift-JIS (cp932) padded with ``0x00`` (and
sometimes ``0xFD`` from older toolchains). The importer adapter is the
only consumer of decoded names; it re-encodes them to truncated SJIS to
match the lookup convention MMD uses for cross-referencing PMX bones.
"""
from __future__ import annotations

from pmx.encoding import Cursor, read_pmd_text

from posecascade.errors import MalformedAssetError
from vmd.types import (
    VmdBoneKeyframe,
    VmdCameraKeyframe,
    VmdHeader,
    VmdIkKeyframe,
    VmdIkSwitch,
    VmdLightKeyframe,
    VmdMorphKeyframe,
    VmdMotion,
    VmdSelfShadowKeyframe,
    VmdSelfShadowMode,
)

# Two known signatures across MMD versions. The 30-byte field is NUL-padded
# in both cases and the value is informational — we do not gate parsing on
# either string but reject obviously malformed (e.g. wrong length) headers.
_HEADER_BYTES = 30
_MODEL_NAME_BYTES = 20
_BONE_NAME_BYTES = 15
_MORPH_NAME_BYTES = 15
_IK_BONE_NAME_BYTES = 20

_BONE_BEZIER_BYTES = 64
_CAMERA_BEZIER_BYTES = 24
_BONE_BEZIER_HEADER_BYTES = 16   # leading 16 bytes of the 64-byte block carry every value
_LENGTH_PREFIX_BYTES = 4         # uint32 record-count prefix at every section's head


def parse_vmd(data: bytes) -> VmdMotion:
    """Parse a VMD byte buffer end-to-end.

    Sections are read in the spec's documented order; reading stops cleanly
    when the buffer runs out (older VMD writers omit the optional sections
    after the bone/morph stream).
    """
    cursor = Cursor(data=data)
    header = _read_header(cursor)
    bones = _read_bone_keyframes(cursor)
    morphs = _read_morph_keyframes(cursor)
    cameras = _read_camera_keyframes(cursor)
    lights = _read_light_keyframes(cursor)
    self_shadows = _read_self_shadow_keyframes(cursor)
    iks = _read_ik_keyframes(cursor)
    return VmdMotion(
        header=header,
        bone_keyframes=bones,
        morph_keyframes=morphs,
        camera_keyframes=cameras,
        light_keyframes=lights,
        self_shadow_keyframes=self_shadows,
        ik_keyframes=iks,
    )


# ----- header -------------------------------------------------------------
def _read_header(cursor: Cursor) -> VmdHeader:
    raw_signature = cursor.read_bytes(_HEADER_BYTES)
    signature = _decode_padded_sjis(raw_signature)
    raw_model_name = cursor.read_bytes(_MODEL_NAME_BYTES)
    model_name = _decode_padded_sjis(raw_model_name)
    return VmdHeader(signature=signature, model_name=model_name)


def _decode_padded_sjis(raw: bytes) -> str:
    """Strip ``0x00`` / ``0xFD`` tails and decode as cp932 (Shift-JIS)."""
    nul = raw.find(b"\x00")
    fd = raw.find(b"\xFD")
    candidates = [end for end in (nul, fd) if end >= 0]
    end = min(candidates) if candidates else len(raw)
    return raw[:end].decode("cp932", errors="replace")


# ----- bone keyframes ----------------------------------------------------
def _read_bone_keyframes(cursor: Cursor) -> tuple[VmdBoneKeyframe, ...]:
    if cursor.remaining() < _LENGTH_PREFIX_BYTES:
        return ()
    count = cursor.read_uint32()
    out: list[VmdBoneKeyframe] = []
    for _ in range(count):
        out.append(_read_one_bone_keyframe(cursor))
    return tuple(out)


def _read_one_bone_keyframe(cursor: Cursor) -> VmdBoneKeyframe:
    bone_name = read_pmd_text(cursor, _BONE_NAME_BYTES)
    frame = cursor.read_uint32()
    position = cursor.read_vec3()
    rotation = cursor.read_vec4()
    bezier_block = cursor.read_bytes(_BONE_BEZIER_BYTES)
    return VmdBoneKeyframe(
        bone_name=bone_name, frame=int(frame),
        position=position, rotation=rotation,
        bezier_handles=_decode_bone_bezier(bezier_block),
    )


def _decode_bone_bezier(block: bytes) -> tuple[tuple[int, int, int, int], ...]:
    """Decode the 64-byte bezier block into ``(4, 4)`` of ``uint8`` 0–127.

    The first 16 bytes contain every interpolation value — the remaining 48
    are repeats / scratchpad written by MMD's own editor and ignored by
    most parsers. Within the leading 16 bytes the layout is interleaved by
    channel:

    - ``X1`` of (X, Y, Z, R) at positions 0, 1, 2, 3
    - ``Y1`` of (X, Y, Z, R) at positions 4, 5, 6, 7
    - ``X2`` of (X, Y, Z, R) at positions 8, 9, 10, 11
    - ``Y2`` of (X, Y, Z, R) at positions 12, 13, 14, 15

    so channel ``c``'s ``(X1, Y1, X2, Y2)`` lives at bytes
    ``c, 4+c, 8+c, 12+c``.
    """
    if len(block) < _BONE_BEZIER_HEADER_BYTES:
        raise MalformedAssetError(
            f"VMD bone bezier block too small: {len(block)} bytes"
        )
    return tuple(
        (
            block[channel + 0],
            block[channel + 4],
            block[channel + 8],
            block[channel + 12],
        )
        for channel in range(4)
    )


# ----- morph keyframes ---------------------------------------------------
def _read_morph_keyframes(cursor: Cursor) -> tuple[VmdMorphKeyframe, ...]:
    if cursor.remaining() < _LENGTH_PREFIX_BYTES:
        return ()
    count = cursor.read_uint32()
    out: list[VmdMorphKeyframe] = []
    for _ in range(count):
        morph_name = read_pmd_text(cursor, _MORPH_NAME_BYTES)
        frame = cursor.read_uint32()
        weight = cursor.read_float()
        out.append(VmdMorphKeyframe(morph_name=morph_name, frame=int(frame), weight=weight))
    return tuple(out)


# ----- camera keyframes --------------------------------------------------
def _read_camera_keyframes(cursor: Cursor) -> tuple[VmdCameraKeyframe, ...]:
    if cursor.remaining() < _LENGTH_PREFIX_BYTES:
        return ()
    count = cursor.read_uint32()
    out: list[VmdCameraKeyframe] = []
    for _ in range(count):
        out.append(_read_one_camera_keyframe(cursor))
    return tuple(out)


def _read_one_camera_keyframe(cursor: Cursor) -> VmdCameraKeyframe:
    frame = cursor.read_uint32()
    distance = cursor.read_float()
    target = cursor.read_vec3()
    rotation = cursor.read_vec3()
    bezier_block = cursor.read_bytes(_CAMERA_BEZIER_BYTES)
    fov = cursor.read_uint32()
    perspective_byte = cursor.read_uint8()
    return VmdCameraKeyframe(
        frame=int(frame), distance=distance, target=target, rotation=rotation,
        bezier_handles=_decode_camera_bezier(bezier_block),
        fov_degrees=int(fov),
        perspective_off=bool(perspective_byte),
    )


def _decode_camera_bezier(block: bytes) -> tuple[tuple[int, int, int, int], ...]:
    """Decode the camera 24-byte bezier block (6 channels × 4 control points)."""
    return tuple(
        tuple(block[channel * 4: channel * 4 + 4])  # type: ignore[misc]
        for channel in range(6)
    )


# ----- light keyframes ---------------------------------------------------
def _read_light_keyframes(cursor: Cursor) -> tuple[VmdLightKeyframe, ...]:
    if cursor.remaining() < _LENGTH_PREFIX_BYTES:
        return ()
    count = cursor.read_uint32()
    out: list[VmdLightKeyframe] = []
    for _ in range(count):
        frame = cursor.read_uint32()
        color = cursor.read_vec3()
        direction = cursor.read_vec3()
        out.append(VmdLightKeyframe(frame=int(frame), color=color, direction=direction))
    return tuple(out)


# ----- self-shadow keyframes --------------------------------------------
def _read_self_shadow_keyframes(cursor: Cursor) -> tuple[VmdSelfShadowKeyframe, ...]:
    if cursor.remaining() < _LENGTH_PREFIX_BYTES:
        return ()
    count = cursor.read_uint32()
    out: list[VmdSelfShadowKeyframe] = []
    for _ in range(count):
        frame = cursor.read_uint32()
        mode_byte = cursor.read_uint8()
        if mode_byte not in VmdSelfShadowMode._value2member_map_:
            raise MalformedAssetError(f"unknown self-shadow mode {mode_byte}")
        distance = cursor.read_float()
        out.append(
            VmdSelfShadowKeyframe(
                frame=int(frame), mode=VmdSelfShadowMode(mode_byte), distance=distance,
            )
        )
    return tuple(out)


# ----- IK keyframes ------------------------------------------------------
def _read_ik_keyframes(cursor: Cursor) -> tuple[VmdIkKeyframe, ...]:
    if cursor.remaining() < _LENGTH_PREFIX_BYTES:
        return ()
    count = cursor.read_uint32()
    out: list[VmdIkKeyframe] = []
    for _ in range(count):
        out.append(_read_one_ik_keyframe(cursor))
    return tuple(out)


def _read_one_ik_keyframe(cursor: Cursor) -> VmdIkKeyframe:
    frame = cursor.read_uint32()
    visible_byte = cursor.read_uint8()
    switch_count = cursor.read_uint32()
    switches: list[VmdIkSwitch] = []
    for _ in range(switch_count):
        bone_name = read_pmd_text(cursor, _IK_BONE_NAME_BYTES)
        enabled = bool(cursor.read_uint8())
        switches.append(VmdIkSwitch(bone_name=bone_name, enabled=enabled))
    return VmdIkKeyframe(
        frame=int(frame), visible=bool(visible_byte), switches=tuple(switches),
    )


