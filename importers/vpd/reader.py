"""VPD text-format parser.

The format is informal but consistent across MMD's own VPD writer + the
common community editors:

::

    Vocaloid Pose Data file

    miku.osm;
    27;

    Bone0{腰
       0.0,0.0,0.0          ;trans
       0.0,0.0,0.0,1.0      ;rot quat (x,y,z,w)
    }

Comments use ``;`` to end of line, blank lines are ignored, and the
record name (``Bone0`` / ``Morph1``) is purely positional — the bone
itself is identified by the text inside the braces. Some files also
carry morph blocks:

::

    Morph0{あ
      1.000000;
    }

Encoding is Shift-JIS / cp932 throughout.
"""
from __future__ import annotations

import re

from posecascade.errors import MalformedAssetError
from vpd.types import VpdBoneOverride, VpdMorphOverride, VpdPose

_HEADER_LINE = "Vocaloid Pose Data file"
_BONE_BLOCK_RE = re.compile(r"^\s*Bone\d+\s*\{\s*(?P<name>[^\r\n]+?)\s*$")
_MORPH_BLOCK_RE = re.compile(r"^\s*Morph\d+\s*\{\s*(?P<name>[^\r\n]+?)\s*$")
_VEC3_RE = re.compile(
    r"^\s*(?P<x>[-+0-9.eE]+)\s*,\s*(?P<y>[-+0-9.eE]+)\s*,\s*(?P<z>[-+0-9.eE]+)\s*"
)
_VEC4_RE = re.compile(
    r"^\s*(?P<x>[-+0-9.eE]+)\s*,\s*(?P<y>[-+0-9.eE]+)\s*,\s*(?P<z>[-+0-9.eE]+)\s*,"
    r"\s*(?P<w>[-+0-9.eE]+)\s*"
)
_FLOAT_RE = re.compile(r"^\s*(?P<v>[-+0-9.eE]+)\s*")


def parse_vpd(text: str) -> VpdPose:
    """Parse a VPD source string into a :class:`VpdPose`.

    Raises :class:`MalformedAssetError` on a missing header or block
    structure that the spec requires; missing optional fields (like the
    morph section) just produce empty tuples.
    """
    lines = _strip_comments(text.splitlines())
    if not lines:
        raise MalformedAssetError("VPD file is empty")
    if not lines[0].startswith(_HEADER_LINE):
        raise MalformedAssetError(
            f"unexpected VPD header: {lines[0]!r} (expected {_HEADER_LINE!r})",
        )
    cursor = _Cursor(lines, position=1)
    model_name = _read_terminated_token(cursor)
    _read_int(cursor)   # bone count — ignored, we read until EOF
    bones, morphs = _read_blocks(cursor)
    return VpdPose(model_name=model_name, bones=bones, morphs=morphs)


def parse_vpd_bytes(data: bytes) -> VpdPose:
    """Decode SJIS bytes then defer to :func:`parse_vpd`."""
    return parse_vpd(data.decode("cp932", errors="replace"))


def _strip_comments(lines: list[str]) -> list[str]:
    """Drop ``;``-suffixed comments and empty lines."""
    out: list[str] = []
    for raw in lines:
        comment = raw.find(";")
        line = raw if comment < 0 else raw[:comment]
        line = line.strip()
        if line:
            out.append(line)
    return out


class _Cursor:
    """Tiny line cursor used by the recursive-descent VPD parser."""

    def __init__(self, lines: list[str], position: int = 0) -> None:
        self.lines = lines
        self.position = position

    def remaining(self) -> int:
        return len(self.lines) - self.position

    def peek(self) -> str | None:
        if self.position >= len(self.lines):
            return None
        return self.lines[self.position]

    def advance(self) -> str:
        if self.position >= len(self.lines):
            raise MalformedAssetError("unexpected end of VPD file")
        line = self.lines[self.position]
        self.position += 1
        return line


def _read_terminated_token(cursor: _Cursor) -> str:
    """Read one ``foo.osm;`` style line and strip the trailing semicolon."""
    line = cursor.advance().rstrip()
    return line.removesuffix(";").strip()


def _read_int(cursor: _Cursor) -> int:
    line = cursor.advance().rstrip().removesuffix(";").strip()
    try:
        return int(line)
    except ValueError as err:
        raise MalformedAssetError(f"expected integer, got {line!r}") from err


def _read_blocks(cursor: _Cursor) -> tuple[
    tuple[VpdBoneOverride, ...], tuple[VpdMorphOverride, ...],
]:
    """Parse alternating ``BoneN{...}`` and ``MorphN{...}`` blocks until EOF."""
    bones: list[VpdBoneOverride] = []
    morphs: list[VpdMorphOverride] = []
    while cursor.remaining() > 0:
        head = cursor.peek() or ""
        bone_match = _BONE_BLOCK_RE.match(head)
        if bone_match:
            cursor.advance()
            bones.append(_read_bone_body(cursor, bone_match.group("name").strip()))
            continue
        morph_match = _MORPH_BLOCK_RE.match(head)
        if morph_match:
            cursor.advance()
            morphs.append(_read_morph_body(cursor, morph_match.group("name").strip()))
            continue
        # Skip stray lines outside any block — VPD files in the wild
        # sometimes carry extra metadata the spec doesn't formalise.
        cursor.advance()
    return tuple(bones), tuple(morphs)


def _read_bone_body(cursor: _Cursor, name: str) -> VpdBoneOverride:
    translation = _expect_vec3(cursor.advance())
    rotation = _expect_vec4(cursor.advance())
    _expect_close_brace(cursor)
    return VpdBoneOverride(name=name, translation=translation, rotation=rotation)


def _read_morph_body(cursor: _Cursor, name: str) -> VpdMorphOverride:
    weight = _expect_float(cursor.advance())
    _expect_close_brace(cursor)
    return VpdMorphOverride(name=name, weight=weight)


def _expect_close_brace(cursor: _Cursor) -> None:
    line = cursor.advance().strip()
    if not line.startswith("}"):
        raise MalformedAssetError(f"expected '}}', got {line!r}")


def _expect_vec3(line: str) -> tuple[float, float, float]:
    match = _VEC3_RE.match(line)
    if match is None:
        raise MalformedAssetError(f"expected vec3, got {line!r}")
    return (
        float(match.group("x")),
        float(match.group("y")),
        float(match.group("z")),
    )


def _expect_vec4(line: str) -> tuple[float, float, float, float]:
    match = _VEC4_RE.match(line)
    if match is None:
        raise MalformedAssetError(f"expected vec4, got {line!r}")
    return (
        float(match.group("x")),
        float(match.group("y")),
        float(match.group("z")),
        float(match.group("w")),
    )


def _expect_float(line: str) -> float:
    match = _FLOAT_RE.match(line)
    if match is None:
        raise MalformedAssetError(f"expected float, got {line!r}")
    return float(match.group("v"))
