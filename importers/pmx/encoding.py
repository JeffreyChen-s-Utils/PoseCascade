"""Binary cursor + length-prefixed string decoder for PMX / PMD.

PMX uses length-prefixed strings whose encoding is selected by the header
flag (UTF-16-LE = 0, UTF-8 = 1). PMD uses fixed-byte SJIS strings padded
with zeros. Both quirks live here so the section readers stay clean.

The cursor wraps a ``bytes`` buffer rather than a file-like object so we
get O(1) unpacks via :func:`struct.unpack_from` without seek overhead and
without holding a file descriptor open across asset processing.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from pmx.types import PmxTextEncoding
from posecascade.errors import MalformedAssetError
from posecascade.render.constants import MAX_PMX_TEXT_BYTES

_INT8 = "<b"
_UINT8 = "<B"
_INT16 = "<h"
_UINT16 = "<H"
_INT32 = "<i"
_UINT32 = "<I"
_FLOAT = "<f"
_VEC2 = "<ff"
_VEC3 = "<fff"
_VEC4 = "<ffff"

_SIGNED_FORMAT = {1: _INT8, 2: _INT16, 4: _INT32}
_UNSIGNED_FORMAT = {1: _UINT8, 2: _UINT16, 4: _UINT32}


@dataclass
class Cursor:
    """Random-access cursor over a ``bytes`` buffer.

    All format-internal parser code goes through this. It is the only place
    in the importer that calls :func:`struct.unpack_from` directly.
    """

    data: bytes
    offset: int = 0

    def remaining(self) -> int:
        return len(self.data) - self.offset

    def require(self, n: int) -> None:
        """Raise :class:`MalformedAssetError` if fewer than ``n`` bytes remain."""
        if self.offset + n > len(self.data):
            raise MalformedAssetError(
                f"unexpected end of file: need {n} bytes at offset {self.offset}, "
                f"have {self.remaining()}"
            )

    def read_bytes(self, n: int) -> bytes:
        self.require(n)
        chunk = self.data[self.offset:self.offset + n]
        self.offset += n
        return chunk

    def unpack(self, fmt: str) -> tuple:
        """Read ``struct.calcsize(fmt)`` bytes and unpack with ``fmt``."""
        size = struct.calcsize(fmt)
        self.require(size)
        result = struct.unpack_from(fmt, self.data, self.offset)
        self.offset += size
        return result

    # ----- scalars -----
    def read_int8(self) -> int:
        return int(self.unpack(_INT8)[0])

    def read_uint8(self) -> int:
        return int(self.unpack(_UINT8)[0])

    def read_int16(self) -> int:
        return int(self.unpack(_INT16)[0])

    def read_uint16(self) -> int:
        return int(self.unpack(_UINT16)[0])

    def read_int32(self) -> int:
        return int(self.unpack(_INT32)[0])

    def read_uint32(self) -> int:
        return int(self.unpack(_UINT32)[0])

    def read_float(self) -> float:
        return float(self.unpack(_FLOAT)[0])

    # ----- vectors (kept as tuples to stay frozen-dataclass friendly) -----
    def read_vec2(self) -> tuple[float, float]:
        return tuple(float(v) for v in self.unpack(_VEC2))   # type: ignore[return-value]

    def read_vec3(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.unpack(_VEC3))   # type: ignore[return-value]

    def read_vec4(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.unpack(_VEC4))   # type: ignore[return-value]


def read_signed_index(cursor: Cursor, size: int) -> int:
    """Read a signed PMX index (bone / material / morph / texture / rigid).

    PMX's "no reference" sentinel is the all-ones bit pattern at every width
    (0xFF / 0xFFFF / 0xFFFFFFFF). With ``struct``'s signed unpack this comes
    out as ``-1`` directly, so we don't need to re-map.
    """
    fmt = _SIGNED_FORMAT.get(size)
    if fmt is None:
        raise MalformedAssetError(f"unsupported PMX signed index size {size}")
    return int(cursor.unpack(fmt)[0])


def read_unsigned_index(cursor: Cursor, size: int) -> int:
    """Read an unsigned PMX index (used for vertex / face indices)."""
    fmt = _UNSIGNED_FORMAT.get(size)
    if fmt is None:
        raise MalformedAssetError(f"unsupported PMX unsigned index size {size}")
    return int(cursor.unpack(fmt)[0])


def read_pmx_text(cursor: Cursor, encoding: PmxTextEncoding) -> str:
    """Read a length-prefixed PMX string.

    Layout:

    - ``int32`` byte length ``L`` (``0 ≤ L ≤ MAX_PMX_TEXT_BYTES``)
    - ``L`` bytes decoded as UTF-16-LE or UTF-8 per ``encoding``
    """
    byte_length = cursor.read_int32()
    if byte_length < 0:
        raise MalformedAssetError(f"negative PMX text length: {byte_length}")
    if byte_length > MAX_PMX_TEXT_BYTES:
        raise MalformedAssetError(
            f"PMX text length {byte_length} exceeds MAX_PMX_TEXT_BYTES "
            f"({MAX_PMX_TEXT_BYTES})"
        )
    raw = cursor.read_bytes(byte_length)
    if encoding == PmxTextEncoding.UTF16_LE:
        return raw.decode("utf-16-le", errors="replace")
    return raw.decode("utf-8", errors="replace")


def read_pmd_text(cursor: Cursor, byte_count: int) -> str:
    """Read a fixed-length PMD string (Shift-JIS / cp932) padded with NULs.

    PMD's text fields are zero-padded to a known width. Some toolchains use
    ``0xFD`` as a "garbage past here" sentinel; we stop at the first ``0x00``
    or first ``0xFD``, whichever appears first.
    """
    raw = cursor.read_bytes(byte_count)
    nul = raw.find(b"\x00")
    fd = raw.find(b"\xFD")
    candidates = [end for end in (nul, fd) if end >= 0]
    end = min(candidates) if candidates else len(raw)
    return raw[:end].decode("cp932", errors="replace")
