"""Engine-side PMX display-frame groups.

PMX models ship with a list of ``display_frame`` panels — the same
"face / body / arm" tabs MMD's Bone Manipulation Frame uses to group
controls. The timeline editor reuses these panels for tree-style track
grouping; a model whose author painted clean groups gets the same
ergonomics in PoseCascade as in MMD.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class DisplayFrameElementKind(IntEnum):
    BONE = 0
    MORPH = 1


@dataclass(frozen=True)
class DisplayFrameElement:
    """One member of a group — either a bone or a morph index."""

    kind: DisplayFrameElementKind
    index: int


@dataclass(frozen=True)
class DisplayFrameGroup:
    """One PMX panel: a name + a list of members.

    ``is_special`` mirrors PMX's ``Root`` / ``表情`` (Expressions) flag —
    those two panels can't be deleted / renamed in MMD, and the timeline
    UI may treat them differently (always-visible, pinned to the top).
    """

    name: str
    is_special: bool = False
    elements: tuple[DisplayFrameElement, ...] = field(default_factory=tuple)
