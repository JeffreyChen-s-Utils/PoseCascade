"""Multi-slot animation driver.

Each slot in a :class:`~posecascade.scene.model_slot.SceneSlots` may
have its own VMD motion. :class:`SlotsPlayer` builds a
:class:`VmdAnimationPlayer` per slot at construction time, then on
each :meth:`apply` walks all of them in registration order — the
slot's per-bone / per-morph / IK / physics chain runs first, then the
cross-slot ``external parent`` resolver snaps any "follow that bone in
the other model" relationships into place.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from posecascade.animation.player import VmdAnimationPlayer
from posecascade.scene.external_parent import apply_external_parents
from posecascade.scene.model_slot import ModelSlot, SceneSlots
from posecascade.utils.profiling import frame_section


@dataclass
class SlotsPlayer:
    """Drives every slot's animation + applies cross-slot bindings.

    The player owns one :class:`VmdAnimationPlayer` per slot whose
    motion is non-``None``. Slots without motion are still tracked (so
    their bones remain reachable via the slot lookup for external
    parenting) but stay at their rest pose.
    """

    slots: SceneSlots
    _players: dict[str, VmdAnimationPlayer] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for slot in self.slots:
            if slot.is_stage:
                # Stage slots are passive props; they get rendered like
                # any other slot but never receive a per-frame animation
                # pass. Skipping the player build also saves the bone /
                # morph / IK / physics setup the player would do at
                # construction time.
                continue
            if slot.motion is not None:
                self._players[slot.name] = VmdAnimationPlayer.for_imported_scene(
                    motion=slot.motion,
                    imported=slot.imported,
                )

    def apply(self, time_seconds: float) -> None:
        """Advance every slot's animation, then resolve external parents."""
        with frame_section("slots_player.apply"):
            for slot in self.slots:
                player = self._players.get(slot.name)
                if player is None:
                    continue
                player.apply(time_seconds)
            apply_external_parents(self.slots, self.slots.find)

    def player_for(self, slot_name: str) -> VmdAnimationPlayer | None:
        """Return the underlying per-slot player (debug + UI integration)."""
        return self._players.get(slot_name)


def make_slot(
    name: str,
    *,
    imported,                               # noqa: ANN001 — runtime ImportedScene
    motion=None,                            # noqa: ANN001 — runtime VmdMotionAsset
    visible: bool = True,
    is_stage: bool = False,
) -> ModelSlot:
    """Convenience builder for callers that don't already import :class:`ModelSlot`."""
    return ModelSlot(
        name=name, imported=imported, motion=motion,
        visible=visible, is_stage=is_stage,
    )


def make_stage_slot(name: str, imported) -> ModelSlot:    # noqa: ANN001
    """Sugar for ``make_slot(... is_stage=True)`` — reads better at call sites."""
    return make_slot(name=name, imported=imported, is_stage=True)
