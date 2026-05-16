"""Synthesise an MMD-style material for non-MMD imports.

glTF / OBJ / FBX meshes don't carry the ``MMDMaterial`` the toon
pipeline keys off, so they normally render through the basic forward
path. When the renderer's ``force_toon_shading`` toggle is on, every
such mesh gets bound to the synthesised material from this module
plus the procedural 2-band toon ramp ``default_toon_ramp_pixels``
returns. The visible effect is cel-banded shading + thin black
outline — MMD look applied to non-MMD assets.

The defaults are tuned to read well on a typical anime-style VRoid
glTF (light ambient, almost-white diffuse, thin black edge at
``edge_size = 0.02``). The toon ramp is a 1×4 image: the bottom two
pixels are the shadow tone, the top two are the lit tone, with the
band falling at the 50% Lambert mark. Tweak via the constants below
if you need a different cel-shade style.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from posecascade.render.material import (
    MAT_FLAG_HAS_EDGE,
    MMDMaterial,
    SphereMode,
)

_TOON_RAMP_HEIGHT = 4
_TOON_RAMP_WIDTH = 1
# Lit / shadow tones sit slightly under full white / black so the
# mesh's own base-colour texture isn't blown out / crushed when the
# toon shader multiplies them in. Picked by eye against the bundled
# VRoid character — the shadow tone reads as a cool blue-grey that
# blends with the gradient sky's ground band.
_SHADOW_TONE = (118, 122, 138, 255)
_LIT_TONE = (220, 220, 222, 255)

# Edge size tuned to read on a 1.5m-tall VRoid model at editor zoom.
# Too thick and the inverted-hull approach's normal-seam artefacts
# (eye sockets, mouth corners) become visible; too thin and the
# outline disappears at typical camera distance.
DEFAULT_TOON_EDGE_SIZE = 0.006
DEFAULT_TOON_EDGE_COLOR = (0.06, 0.06, 0.07, 1.0)


def default_toon_material() -> MMDMaterial:
    """Return a generic MMD material suitable for any non-MMD mesh.

    Keeps the diffuse near white so the mesh's own base-colour map
    drives the visible tint; the toon ramp this module ships gives
    the cel banding. Edge flag is on so the inverted-hull outline
    pass runs.
    """
    return MMDMaterial(
        diffuse=(0.95, 0.95, 0.95, 1.0),
        specular=(0.0, 0.0, 0.0),
        specular_power=0.0,
        # Low ambient — the toon ramp already provides the floor on the
        # shadow side. A larger ambient stacks on top of the lit side
        # too and washes the textures out (the bundled VRoid was nearly
        # white in face / clothes / hair before this was dropped).
        ambient=(0.15, 0.15, 0.18),
        edge_color=DEFAULT_TOON_EDGE_COLOR,
        edge_size=DEFAULT_TOON_EDGE_SIZE,
        sphere_texture_index=None,
        sphere_mode=SphereMode.DISABLED,
        toon_texture_index=None,
        flags=MAT_FLAG_HAS_EDGE,
    )


def default_toon_ramp_pixels() -> NDArray[np.uint8]:
    """Build the 1×4 RGBA ramp used as the default cel-shading LUT.

    Sampling convention (matches the toon fragment shader): V=0 is the
    bottom of the texture and reads as fully lit; V=1 is the top and
    reads as shadow. The shader's ``texture(u_toonTex, vec2(0.5,
    1 - lambert))`` makes Lambert ≈ 1 land at the lit half and
    Lambert ≈ 0 at the shadow half, with the cel boundary at the
    50% Lambert mark.
    """
    pixels = np.zeros((_TOON_RAMP_HEIGHT, _TOON_RAMP_WIDTH, 4), dtype=np.uint8)
    pixels[0, 0] = _LIT_TONE         # V = 0 → fully lit
    pixels[1, 0] = _LIT_TONE         # V = 0.25 → still lit (sharp band)
    pixels[2, 0] = _SHADOW_TONE      # V = 0.50 → step into shadow
    pixels[3, 0] = _SHADOW_TONE      # V = 1.00 → full shadow
    return pixels
