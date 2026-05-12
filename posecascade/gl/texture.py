"""GL texture upload helpers.

Sets up 2D textures for the forward pass: linear filtering with mipmaps,
GL_REPEAT wrap, RGBA8 internal format. Optionally treats the source pixels
as sRGB (for PBR base-colour maps).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_NEAREST,
    GL_REPEAT,
    GL_RGBA,
    GL_RGBA8,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    glBindTexture,
    glDeleteTextures,
    glGenerateMipmap,
    glGenTextures,
    glTexImage2D,
    glTexParameteri,
)

_RGBA_NDIM = 3
_RGBA_CHANNELS = 4


@dataclass
class GLTexture:
    """GPU-resident 2D texture handle."""

    texture_id: int
    width: int
    height: int

    def delete(self) -> None:
        glDeleteTextures(1, [self.texture_id])


def upload_texture(pixels: np.ndarray, *, srgb: bool = False) -> GLTexture:
    """Upload an HxWx4 uint8 RGBA buffer to a fresh GL texture and return it.

    ``srgb`` is accepted for API symmetry with future sRGB-aware framebuffers
    but is currently ignored — we always upload as ``GL_RGBA8`` because the
    forward framebuffer is linear and the GPU's sRGB→linear sample conversion
    would otherwise leave colours visibly under-bright on display.
    """
    _ = srgb  # reserved for when framebuffer-sRGB is wired up
    if pixels.ndim != _RGBA_NDIM or pixels.shape[2] != _RGBA_CHANNELS:
        raise ValueError(f"expected HxWx4 RGBA, got shape {pixels.shape}")
    if pixels.dtype != np.uint8:
        raise ValueError(f"expected uint8 pixels, got {pixels.dtype}")
    height, width, _ = pixels.shape
    contig = np.ascontiguousarray(pixels)
    texture_id = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, contig)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)
    return GLTexture(texture_id=texture_id, width=width, height=height)


def make_white_fallback() -> GLTexture:
    """Create a 1x1 fully-opaque-white texture used when a mesh has no base-colour map."""
    pixels = np.full((1, 1, 4), 255, dtype=np.uint8)
    return upload_texture(pixels, srgb=False)


def set_toon_sampler_params(texture_id: int) -> None:
    """Re-tune ``texture_id`` for MMD-style cel shading.

    Toon ramps encode a stepped lighting LUT along their V axis. The default
    ``LINEAR_MIPMAP_LINEAR`` + ``REPEAT`` sampler smears those bands into a
    smooth gradient and wraps the bottom/top rows around the seam — both fight
    the desired anime look. Switch to ``NEAREST`` + ``CLAMP_TO_EDGE`` so each
    Lambert bucket lands on exactly one ramp texel.

    Safe to call after :func:`upload_texture`; assumes the texture is bound
    to ``GL_TEXTURE_2D`` by this helper (it binds/unbinds itself).
    """
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)
