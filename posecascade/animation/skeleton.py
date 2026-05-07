"""Skeleton — a tree of bones with bind-pose inverse matrices."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Bone:
    """A single bone: index, parent index (-1 for root), and inverse bind matrix."""

    name: str
    index: int
    parent_index: int
    inverse_bind: NDArray[np.float32]  # 4x4


@dataclass(frozen=True)
class Skeleton:
    """A flat array of bones in topologically sorted order (parents before children)."""

    bones: tuple[Bone, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.bones)

    def root_indices(self) -> tuple[int, ...]:
        return tuple(b.index for b in self.bones if b.parent_index < 0)
