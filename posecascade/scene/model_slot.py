"""Multi-model scene container.

A *slot* owns:

- one :class:`~posecascade.assets.types.ImportedScene` (model + skin +
  morphs + physics + …),
- an optional VMD motion to drive that model,
- a world-space ``Transform`` offset (where the slot sits relative to
  the global origin),
- a per-slot visibility flag,
- zero or more :class:`ExternalParentBinding`s — bone-follow rules
  pointing into *another* slot.

:class:`SceneSlots` is the registry the integrator walks each frame —
it preserves insertion order (matters for export) and offers a
name-based lookup the cross-slot binding resolver consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from posecascade.animation.vmd_track import VmdMotionAsset
from posecascade.assets.types import ImportedScene
from posecascade.scene.external_parent import ExternalParentBinding
from posecascade.scene.transform import Transform


@dataclass
class ModelSlot:
    """One model + its motion + its placement.

    ``imported`` is the data side (geometry + skin + …); the slot's
    ``transform`` is layered *above* the imported scene root, so a slot
    placed at ``(5, 0, 0)`` shifts every bone of its model by that
    offset without touching the bone Nodes themselves.
    """

    name: str
    imported: ImportedScene
    motion: VmdMotionAsset | None = None
    transform: Transform = field(default_factory=Transform)
    visible: bool = True
    external_parents: tuple[ExternalParentBinding, ...] = field(default_factory=tuple)
    # Stage slots are passive props (dance floor, walls, environment
    # PMX models): the renderer draws them like any other slot but the
    # animation player skips bone / morph / IK / physics passes on
    # them. Defaults to False so existing model loads stay untouched.
    is_stage: bool = False


@dataclass
class SceneSlots:
    """Ordered registry of :class:`ModelSlot` objects."""

    slots: list[ModelSlot] = field(default_factory=list)

    def add(self, slot: ModelSlot) -> ModelSlot:
        """Append ``slot`` to the registry. Names must be unique."""
        if any(existing.name == slot.name for existing in self.slots):
            raise ValueError(f"slot name already registered: {slot.name!r}")
        self.slots.append(slot)
        return slot

    def remove(self, name: str) -> ModelSlot | None:
        """Drop the slot named ``name``; return it for cleanup or ``None``."""
        for index, slot in enumerate(self.slots):
            if slot.name == name:
                return self.slots.pop(index)
        return None

    def find(self, name: str) -> ModelSlot | None:
        for slot in self.slots:
            if slot.name == name:
                return slot
        return None

    def __iter__(self):                              # noqa: ANN204 — yields ModelSlot
        return iter(self.slots)

    def __len__(self) -> int:
        return len(self.slots)

    def __bool__(self) -> bool:
        return bool(self.slots)
