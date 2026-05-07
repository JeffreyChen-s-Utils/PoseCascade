"""Mutable, edit-friendly animation document.

The importer hands back a frozen :class:`~posecascade.animation.vmd_track.VmdMotionAsset`
for fast playback; the timeline editor needs something it can mutate.
:class:`AnimationDocument` mirrors the VMD record set as plain Python
lists and exposes the per-keyframe edit operations the
:mod:`~posecascade.animation.commands` undo / redo layer wraps.

Saving back to disk goes :class:`AnimationDocument` →
:meth:`to_motion` (build a :class:`VmdMotion`) →
:func:`vmd.writer.serialize_vmd` (produce the byte buffer).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from vmd.types import (
    VmdBoneKeyframe,
    VmdCameraKeyframe,
    VmdHeader,
    VmdIkKeyframe,
    VmdLightKeyframe,
    VmdMorphKeyframe,
    VmdMotion,
    VmdSelfShadowKeyframe,
)


@dataclass
class AnimationDocument:
    """Mutable VMD-shaped document for the timeline editor."""

    header: VmdHeader = field(
        default_factory=lambda: VmdHeader(
            signature="Vocaloid Pose Data 0002",
            model_name="",
        ),
    )
    bone_keyframes: list[VmdBoneKeyframe] = field(default_factory=list)
    morph_keyframes: list[VmdMorphKeyframe] = field(default_factory=list)
    camera_keyframes: list[VmdCameraKeyframe] = field(default_factory=list)
    light_keyframes: list[VmdLightKeyframe] = field(default_factory=list)
    self_shadow_keyframes: list[VmdSelfShadowKeyframe] = field(default_factory=list)
    ik_keyframes: list[VmdIkKeyframe] = field(default_factory=list)

    # ----- ctors / converters -----------------------------------------
    @classmethod
    def from_motion(cls, motion: VmdMotion) -> AnimationDocument:
        """Snapshot a parsed :class:`VmdMotion` into a mutable document."""
        return cls(
            header=motion.header,
            bone_keyframes=list(motion.bone_keyframes),
            morph_keyframes=list(motion.morph_keyframes),
            camera_keyframes=list(motion.camera_keyframes),
            light_keyframes=list(motion.light_keyframes),
            self_shadow_keyframes=list(motion.self_shadow_keyframes),
            ik_keyframes=list(motion.ik_keyframes),
        )

    def to_motion(self) -> VmdMotion:
        """Snapshot the current state back into an immutable :class:`VmdMotion`."""
        return VmdMotion(
            header=self.header,
            bone_keyframes=tuple(self.bone_keyframes),
            morph_keyframes=tuple(self.morph_keyframes),
            camera_keyframes=tuple(self.camera_keyframes),
            light_keyframes=tuple(self.light_keyframes),
            self_shadow_keyframes=tuple(self.self_shadow_keyframes),
            ik_keyframes=tuple(self.ik_keyframes),
        )

    # ----- bone keyframes ---------------------------------------------
    def insert_bone_keyframe(self, keyframe: VmdBoneKeyframe) -> None:
        """Add a bone keyframe; replaces any existing one with the same
        ``(bone_name, frame)``."""
        self._upsert(
            self.bone_keyframes, keyframe,
            key_attrs=("bone_name", "frame"),
        )

    def delete_bone_keyframe(
        self, bone_name: str, frame: int,
    ) -> VmdBoneKeyframe | None:
        return self._delete_by(
            self.bone_keyframes, bone_name=bone_name, frame=frame,
        )

    def find_bone_keyframe(
        self, bone_name: str, frame: int,
    ) -> VmdBoneKeyframe | None:
        for kf in self.bone_keyframes:
            if kf.bone_name == bone_name and kf.frame == frame:
                return kf
        return None

    # ----- morph keyframes --------------------------------------------
    def insert_morph_keyframe(self, keyframe: VmdMorphKeyframe) -> None:
        self._upsert(
            self.morph_keyframes, keyframe,
            key_attrs=("morph_name", "frame"),
        )

    def delete_morph_keyframe(
        self, morph_name: str, frame: int,
    ) -> VmdMorphKeyframe | None:
        return self._delete_by(
            self.morph_keyframes, morph_name=morph_name, frame=frame,
        )

    # ----- camera keyframes -------------------------------------------
    def insert_camera_keyframe(self, keyframe: VmdCameraKeyframe) -> None:
        self._upsert(self.camera_keyframes, keyframe, key_attrs=("frame",))

    def delete_camera_keyframe(self, frame: int) -> VmdCameraKeyframe | None:
        return self._delete_by(self.camera_keyframes, frame=frame)

    # ----- light keyframes --------------------------------------------
    def insert_light_keyframe(self, keyframe: VmdLightKeyframe) -> None:
        self._upsert(self.light_keyframes, keyframe, key_attrs=("frame",))

    def delete_light_keyframe(self, frame: int) -> VmdLightKeyframe | None:
        return self._delete_by(self.light_keyframes, frame=frame)

    # ----- self-shadow keyframes -------------------------------------
    def insert_self_shadow_keyframe(self, keyframe: VmdSelfShadowKeyframe) -> None:
        self._upsert(self.self_shadow_keyframes, keyframe, key_attrs=("frame",))

    def delete_self_shadow_keyframe(self, frame: int) -> VmdSelfShadowKeyframe | None:
        return self._delete_by(self.self_shadow_keyframes, frame=frame)

    # ----- IK keyframes ----------------------------------------------
    def insert_ik_keyframe(self, keyframe: VmdIkKeyframe) -> None:
        self._upsert(self.ik_keyframes, keyframe, key_attrs=("frame",))

    def delete_ik_keyframe(self, frame: int) -> VmdIkKeyframe | None:
        return self._delete_by(self.ik_keyframes, frame=frame)

    # ----- generic helpers --------------------------------------------
    def _upsert(
        self,
        records: list,        # noqa: ANN001 — generic over typed VMD record lists
        new_record,           # noqa: ANN001
        *,
        key_attrs: tuple[str, ...],
    ) -> None:
        """Insert or replace, keyed on the tuple of ``key_attrs``."""
        key = tuple(getattr(new_record, attr) for attr in key_attrs)
        for index, existing in enumerate(records):
            existing_key = tuple(getattr(existing, attr) for attr in key_attrs)
            if existing_key == key:
                records[index] = new_record
                return
        records.append(new_record)

    def _delete_by(self, records: list, **filters):    # noqa: ANN001, ANN201 — generic
        """Remove the first record matching every ``attr=value`` filter; return it."""
        for index, existing in enumerate(records):
            if all(
                getattr(existing, attr) == value
                for attr, value in filters.items()
            ):
                return records.pop(index)
        return None
