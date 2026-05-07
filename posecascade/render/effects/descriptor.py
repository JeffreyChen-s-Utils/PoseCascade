"""Effect descriptor types — what a single post-pass declares.

A descriptor is the *static* part of an effect: which fragment shader,
which textures it samples, what uniforms (with default values + UI
hints) it exposes. The :class:`EffectChain` then composes one or more
descriptors into a per-frame post-effect pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

UniformValue = float | tuple[float, ...] | bool | int


class EffectUniformKind(IntEnum):
    """How the UI should expose a uniform."""

    SCALAR = 0          # one float; rendered as a spin box
    VEC3_COLOR = 1      # three floats, 0..1, surfaced as a colour swatch
    VEC4_COLOR = 2      # RGBA colour
    BOOL = 3            # checkbox
    INT_ENUM = 4        # combo box (use ``enum_labels`` to label values)


class EffectBlendMode(IntEnum):
    """How an effect's output composites with the underlying scene."""

    REPLACE = 0     # output texture replaces the scene
    ADD = 1         # output is added to the scene (typical for bloom / glow)
    MULTIPLY = 2    # output multiplies the scene (typical for tints / vignettes)


@dataclass(frozen=True)
class EffectUniform:
    """One tunable parameter on an effect."""

    name: str
    kind: EffectUniformKind
    default: UniformValue
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    enum_labels: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class EffectInput:
    """One sampler binding the effect requires.

    ``source`` is the name of either the main-scene buffer
    (``main_color``, ``main_depth``) or another effect's output —
    whatever the chain's executor can resolve at draw time.
    """

    sampler_name: str
    source: str = "main_color"


@dataclass(frozen=True)
class EffectDescriptor:
    """The static description of one post-effect pass."""

    name: str
    fragment_shader: str
    inputs: tuple[EffectInput, ...] = field(default_factory=tuple)
    uniforms: tuple[EffectUniform, ...] = field(default_factory=tuple)
    blend_mode: EffectBlendMode = EffectBlendMode.REPLACE
    output_name: str = "result"
    description: str = ""
