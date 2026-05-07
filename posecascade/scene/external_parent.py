"""Cross-slot bone-follow ("external parent") binding.

PMX bones can opt into an "external parent" relationship — their world
transform is dictated by a bone in *another* model rather than their own
parent chain. The classic use is rigging a microphone (its own slot) to
follow a vocalist's hand bone (another slot).

The binding stays format-agnostic: we identify the target by
``(slot_name, bone_name)`` rather than an integer index, so a slot
reordering doesn't silently re-target the wrong bone.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from posecascade.scene.node import Node


@dataclass(frozen=True)
class ExternalParentBinding:
    """``self_bone_name`` follows ``target_slot_name``'s ``target_bone_name``.

    The binding is resolved at the very end of the per-frame chain (after
    bone tracks, morphs, IK, and bone resolver), so the eventual value
    always overwrites whatever the slot's own pipeline computed.
    """

    self_bone_name: str
    target_slot_name: str
    target_bone_name: str


def apply_external_parents(
    slots: Iterable,                                       # noqa: ANN001 — ModelSlot, type below
    slot_lookup: Callable[[str], object | None],
) -> None:
    """Apply every slot's bindings — snap self-side bones to the target's world pose.

    ``slot_lookup(name)`` returns the :class:`ModelSlot` registered under
    ``name`` (or ``None``). The caller threads it explicitly so the
    cross-slot resolution stays decoupled from the container shape — a
    test can pass a plain ``dict.get`` and skip the full
    :class:`SceneSlots` plumbing.

    Bindings whose target slot or bone don't resolve are silently
    skipped (mid-edit "dropped accessory" workflow). Self-targeting
    bindings (``target_slot_name`` matches the binding's home slot)
    are also skipped — the resolver runs in input order and a one-pass
    walk avoids the infinite-loop trap a misconfigured circular
    dependency would otherwise create.
    """
    for slot in slots:
        for binding in slot.external_parents:
            target_slot = slot_lookup(binding.target_slot_name)
            if target_slot is None or target_slot is slot:
                continue
            target_node = _find_bone(target_slot, binding.target_bone_name)
            self_node = _find_bone(slot, binding.self_bone_name)
            if target_node is None or self_node is None:
                continue
            _copy_world_pose(source=target_node, sink=self_node)


def _find_bone(slot, bone_name: str) -> Node | None:    # noqa: ANN001 — slot type below
    """Look up ``bone_name`` in ``slot``'s skin; return the live Node."""
    imported = getattr(slot, "imported", None)
    if imported is None or not imported.skins:
        return None
    for joint in imported.skins[0].joints:
        if isinstance(joint, Node) and joint.name == bone_name:
            return joint
    return None


def _copy_world_pose(*, source: Node, sink: Node) -> None:
    """Snapshot ``source``'s TRS into ``sink``'s ``Transform``.

    Both ``source`` and ``sink`` are bone Nodes; their local transforms
    are typically expressed in the slot's root frame, so a direct copy
    of ``translation`` + ``rotation`` reproduces the target's pose in
    the sink slot without any matrix math.
    """
    sink.transform.set_translation(source.transform.translation.copy())
    sink.transform.set_rotation(source.transform.rotation.copy())
