"""Build a tree of editable timeline tracks from a model + document.

This module is intentionally Qt-free — the multi-track widget reads
:func:`build_track_list` and renders it, but every other consumer (a
plain console dump, a future Web preview) can use the same model.

Group layout follows MMD's panel convention:

1. PMX display-frame groups, in their authored order.
2. A synthetic ``Camera / Light / Self-shadow`` panel listing the
   single-track scene-wide channels (only emitted when the document
   has at least one keyframe in any of them).
3. An ``Other`` panel listing bones / morphs that are present in the
   document but absent from every display frame — common when a VMD
   from another model targets bones the current PMX never declared in
   its panel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from posecascade.animation.display_frames import (
    DisplayFrameElementKind,
    DisplayFrameGroup,
)
from posecascade.animation.document import AnimationDocument
from posecascade.assets.types import ImportedScene
from posecascade.scene.node import Node


class TrackKind(IntEnum):
    BONE = 0
    MORPH = 1
    CAMERA = 2
    LIGHT = 3
    SELF_SHADOW = 4
    IK = 5


@dataclass(frozen=True)
class TrackEntry:
    """One row in the multi-track widget."""

    kind: TrackKind
    display_name: str
    keyframe_count: int


@dataclass(frozen=True)
class TrackGroup:
    """A panel of related tracks, e.g. PMX's ``表情`` (face) group."""

    name: str
    is_special: bool = False
    entries: tuple[TrackEntry, ...] = field(default_factory=tuple)


_SYNTHETIC_SCENE_GROUP_NAME = "Camera / Light"
_OTHER_GROUP_NAME = "Other"


def build_track_list(
    document: AnimationDocument,
    scene: ImportedScene | None = None,
) -> tuple[TrackGroup, ...]:
    """Return the ordered ``(group, entries)`` tree the timeline UI displays.

    Passing ``None`` for ``scene`` is supported — the caller may not have a
    model yet (e.g. editing a fresh document). The result then contains
    only the synthetic camera / light / self-shadow group plus the
    "Other" group for any document-only tracks.
    """
    bone_counts = _bone_keyframe_counts(document)
    morph_counts = _morph_keyframe_counts(document)
    used_bone_names: set[str] = set()
    used_morph_names: set[str] = set()
    groups: list[TrackGroup] = []

    if scene is not None:
        for pmx_group in scene.display_frame_groups:
            entry_set = _entries_for_pmx_group(
                pmx_group, scene, bone_counts, morph_counts,
            )
            entries, bones_in_group, morphs_in_group = entry_set
            used_bone_names.update(bones_in_group)
            used_morph_names.update(morphs_in_group)
            groups.append(
                TrackGroup(
                    name=pmx_group.name,
                    is_special=pmx_group.is_special,
                    entries=entries,
                ),
            )

    scene_group = _synthetic_scene_group(document)
    if scene_group is not None:
        groups.append(scene_group)

    other_group = _other_group(
        document, bone_counts, morph_counts, used_bone_names, used_morph_names,
    )
    if other_group is not None:
        groups.append(other_group)

    return tuple(groups)


# ----- counting helpers ------------------------------------------------
def _bone_keyframe_counts(document: AnimationDocument) -> dict[str, int]:
    counts: dict[str, int] = {}
    for kf in document.bone_keyframes:
        counts[kf.bone_name] = counts.get(kf.bone_name, 0) + 1
    return counts


def _morph_keyframe_counts(document: AnimationDocument) -> dict[str, int]:
    counts: dict[str, int] = {}
    for kf in document.morph_keyframes:
        counts[kf.morph_name] = counts.get(kf.morph_name, 0) + 1
    return counts


# ----- group builders --------------------------------------------------
def _entries_for_pmx_group(
    pmx_group: DisplayFrameGroup,
    scene: ImportedScene,
    bone_counts: dict[str, int],
    morph_counts: dict[str, int],
) -> tuple[tuple[TrackEntry, ...], set[str], set[str]]:
    """Resolve a PMX panel's element indices into named entries."""
    entries: list[TrackEntry] = []
    bones_seen: set[str] = set()
    morphs_seen: set[str] = set()
    for element in pmx_group.elements:
        if element.kind == DisplayFrameElementKind.BONE:
            name = _bone_name_at(scene, element.index)
            if name is None:
                continue
            entries.append(
                TrackEntry(
                    kind=TrackKind.BONE,
                    display_name=name,
                    keyframe_count=bone_counts.get(name, 0),
                ),
            )
            bones_seen.add(name)
        else:
            name = _morph_name_at(scene, element.index)
            if name is None:
                continue
            entries.append(
                TrackEntry(
                    kind=TrackKind.MORPH,
                    display_name=name,
                    keyframe_count=morph_counts.get(name, 0),
                ),
            )
            morphs_seen.add(name)
    return tuple(entries), bones_seen, morphs_seen


def _synthetic_scene_group(document: AnimationDocument) -> TrackGroup | None:
    """Camera / light / self-shadow tracks pulled into one virtual panel."""
    entries: list[TrackEntry] = []
    if document.camera_keyframes:
        entries.append(
            TrackEntry(
                kind=TrackKind.CAMERA,
                display_name="Camera",
                keyframe_count=len(document.camera_keyframes),
            ),
        )
    if document.light_keyframes:
        entries.append(
            TrackEntry(
                kind=TrackKind.LIGHT,
                display_name="Light",
                keyframe_count=len(document.light_keyframes),
            ),
        )
    if document.self_shadow_keyframes:
        entries.append(
            TrackEntry(
                kind=TrackKind.SELF_SHADOW,
                display_name="Self-shadow",
                keyframe_count=len(document.self_shadow_keyframes),
            ),
        )
    if document.ik_keyframes:
        entries.append(
            TrackEntry(
                kind=TrackKind.IK,
                display_name="IK toggles",
                keyframe_count=len(document.ik_keyframes),
            ),
        )
    if not entries:
        return None
    return TrackGroup(
        name=_SYNTHETIC_SCENE_GROUP_NAME,
        is_special=True,
        entries=tuple(entries),
    )


def _other_group(
    document: AnimationDocument,
    bone_counts: dict[str, int],
    morph_counts: dict[str, int],
    used_bone_names: set[str],
    used_morph_names: set[str],
) -> TrackGroup | None:
    """A catch-all for bones / morphs in the document not in any PMX panel."""
    entries: list[TrackEntry] = []
    for name, count in bone_counts.items():
        if name in used_bone_names:
            continue
        entries.append(
            TrackEntry(kind=TrackKind.BONE, display_name=name, keyframe_count=count),
        )
    for name, count in morph_counts.items():
        if name in used_morph_names:
            continue
        entries.append(
            TrackEntry(kind=TrackKind.MORPH, display_name=name, keyframe_count=count),
        )
    if not entries:
        return None
    return TrackGroup(name=_OTHER_GROUP_NAME, is_special=False, entries=tuple(entries))


def _bone_name_at(scene: ImportedScene, index: int) -> str | None:
    if not scene.skins:
        return None
    joints = scene.skins[0].joints
    if not 0 <= index < len(joints):
        return None
    joint = joints[index]
    if not isinstance(joint, Node):
        return None
    return joint.name


def _morph_name_at(scene: ImportedScene, index: int) -> str | None:
    morphs = scene.morphs.by_index
    if not 0 <= index < len(morphs):
        return None
    return morphs[index].name
