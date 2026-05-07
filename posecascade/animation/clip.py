"""Animation clip — keyframe channels driving translation, rotation, or scale."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class ChannelTarget(StrEnum):
    """Which TRS component a channel drives."""

    TRANSLATION = "translation"
    ROTATION = "rotation"
    SCALE = "scale"


class Interpolation(StrEnum):
    """Channel interpolation mode."""

    STEP = "step"
    LINEAR = "linear"
    CUBIC = "cubic"


@dataclass(frozen=True)
class Sampler:
    """Keyframe samples: ``times`` (s) and ``values`` aligned by index."""

    times: NDArray[np.float32]      # (K,)
    values: NDArray[np.float32]     # (K, D) where D is 3 or 4
    interpolation: Interpolation = Interpolation.LINEAR


@dataclass(frozen=True)
class Channel:
    """A sampler bound to a target node + TRS component."""

    node_index: int
    target: ChannelTarget
    sampler: Sampler


@dataclass(frozen=True)
class AnimationClip:
    """A named bundle of channels with a known duration."""

    name: str
    channels: tuple[Channel, ...] = field(default_factory=tuple)
    duration_seconds: float = 0.0
