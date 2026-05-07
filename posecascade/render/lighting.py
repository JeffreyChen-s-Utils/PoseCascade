"""Engine-side lighting state.

The forward / toon shaders currently use a hard-coded directional light
at compile time. The :class:`DirectionalLight` dataclass collects what
VMD's light keyframes supply (direction + RGB color) so a future render
pass can bind it as uniforms instead. :class:`SelfShadowState` exposes
VMD's shadow on / off / full toggles plus the distance parameter; the
renderer reads it without yet implementing the shadow map itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from posecascade.utils.math3d import Vec3, vec3

_DIRECTION_EPSILON = 1.0e-6


@dataclass
class DirectionalLight:
    """A single directional light. ``direction`` points from the surface
    toward the light source — same convention the toon frag uses for its
    ``LIGHT_DIR`` constant."""

    color: Vec3 = None  # type: ignore[assignment]
    direction: Vec3 = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.color is None:
            self.color = vec3(1.0, 1.0, 1.0)
        if self.direction is None:
            self.direction = vec3(0.3, 0.7, 0.6)
        # Always normalise the direction; VMD ships the raw vector and
        # the renderer expects unit length.
        norm = float(np.linalg.norm(self.direction))
        if norm > _DIRECTION_EPSILON:
            self.direction = (self.direction / norm).astype(np.float32, copy=False)


@dataclass(frozen=True)
class SelfShadowState:
    """VMD self-shadow toggle. ``mode``: ``0`` off, ``1`` on, ``2`` full.

    ``distance`` is MMD's shadow-map fall-off control — kept as a plain
    float because the rendering layer hasn't grown a shadow pass yet.
    """

    mode: int = 1
    distance: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.mode != 0
