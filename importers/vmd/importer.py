"""VMD importer adapter.

VMD describes an *animation*, not a model — the engine-facing return type
is therefore :class:`~posecascade.animation.vmd_track.VmdMotionAsset`
rather than :class:`~posecascade.assets.types.ImportedScene`.

This module also owns the :func:`build_motion_asset` bridge that
translates :class:`~vmd.types.VmdMotion` → :class:`VmdMotionAsset`. It
lives here (in the importer plugin) rather than under
``posecascade.animation`` so the engine layer carries no import-time
dependency on the format-internal types.
"""
from __future__ import annotations

from pathlib import Path

from posecascade.animation.vmd_track import (
    VmdCameraTrack,
    VmdLightTrack,
    VmdMotionAsset,
    VmdSelfShadowTrack,
    build_bone_track,
    build_camera_track,
    build_ik_track,
    build_light_track,
    build_morph_track,
    build_self_shadow_track,
    vmd_bone_key,
)
from posecascade.errors import MalformedAssetError
from vmd.reader import parse_vmd
from vmd.types import (
    VmdBoneKeyframe,
    VmdCameraKeyframe,
    VmdIkKeyframe,
    VmdLightKeyframe,
    VmdMorphKeyframe,
    VmdMotion,
    VmdSelfShadowKeyframe,
)


def build_motion_asset(motion: VmdMotion) -> VmdMotionAsset:
    """Group raw bone / morph / IK keyframes by name + hand the rest off opaquely."""
    bone_tracks = tuple(
        _bone_track_from_keyframes(key, kfs)
        for key, kfs in _group_by_truncated_name(motion.bone_keyframes).items()
    )
    morph_tracks = tuple(
        _morph_track_from_keyframes(key, kfs)
        for key, kfs in _group_morph_by_truncated_name(motion.morph_keyframes).items()
    )
    ik_tracks = tuple(
        build_ik_track(name_key=key, frames=frames, enabled=enabled)
        for key, (frames, enabled) in _explode_ik_keyframes(motion.ik_keyframes).items()
    )
    return VmdMotionAsset(
        target_model_name=motion.header.model_name,
        bone_tracks=bone_tracks,
        morph_tracks=morph_tracks,
        ik_tracks=ik_tracks,
        camera_track=_camera_track_from_keyframes(motion.camera_keyframes),
        light_track=_light_track_from_keyframes(motion.light_keyframes),
        self_shadow_track=_self_shadow_track_from_keyframes(motion.self_shadow_keyframes),
    )


def _group_by_truncated_name(
    keyframes: tuple[VmdBoneKeyframe, ...],
) -> dict[str, list[VmdBoneKeyframe]]:
    out: dict[str, list[VmdBoneKeyframe]] = {}
    for keyframe in keyframes:
        out.setdefault(vmd_bone_key(keyframe.bone_name), []).append(keyframe)
    return out


def _group_morph_by_truncated_name(
    keyframes: tuple[VmdMorphKeyframe, ...],
) -> dict[str, list[VmdMorphKeyframe]]:
    out: dict[str, list[VmdMorphKeyframe]] = {}
    for keyframe in keyframes:
        out.setdefault(vmd_bone_key(keyframe.morph_name), []).append(keyframe)
    return out


def _bone_track_from_keyframes(name_key: str, keyframes: list[VmdBoneKeyframe]):
    return build_bone_track(
        name_key=name_key,
        frames=[kf.frame for kf in keyframes],
        positions=[kf.position for kf in keyframes],
        rotations=[kf.rotation for kf in keyframes],
        bezier_handles=[kf.bezier_handles for kf in keyframes],
    )


def _morph_track_from_keyframes(name_key: str, keyframes: list[VmdMorphKeyframe]):
    return build_morph_track(
        name_key=name_key,
        frames=[kf.frame for kf in keyframes],
        weights=[kf.weight for kf in keyframes],
    )


def _camera_track_from_keyframes(
    keyframes: tuple[VmdCameraKeyframe, ...],
) -> VmdCameraTrack | None:
    if not keyframes:
        return None
    return build_camera_track(
        frames=[kf.frame for kf in keyframes],
        targets=[kf.target for kf in keyframes],
        rotations=[kf.rotation for kf in keyframes],
        distances=[kf.distance for kf in keyframes],
        fovs=[float(kf.fov_degrees) for kf in keyframes],
        perspective_offs=[bool(kf.perspective_off) for kf in keyframes],
        bezier_handles=[kf.bezier_handles for kf in keyframes],
    )


def _light_track_from_keyframes(
    keyframes: tuple[VmdLightKeyframe, ...],
) -> VmdLightTrack | None:
    if not keyframes:
        return None
    return build_light_track(
        frames=[kf.frame for kf in keyframes],
        colors=[kf.color for kf in keyframes],
        directions=[kf.direction for kf in keyframes],
    )


def _self_shadow_track_from_keyframes(
    keyframes: tuple[VmdSelfShadowKeyframe, ...],
) -> VmdSelfShadowTrack | None:
    if not keyframes:
        return None
    return build_self_shadow_track(
        frames=[kf.frame for kf in keyframes],
        modes=[int(kf.mode) for kf in keyframes],
        distances=[kf.distance for kf in keyframes],
    )


def _explode_ik_keyframes(
    keyframes: tuple[VmdIkKeyframe, ...],
) -> dict[str, tuple[list[int], list[bool]]]:
    """Fan VMD's bundled IK keyframes out into per-bone (frames, enabled) lists.

    A single VMD IK keyframe at frame N can flip any number of IK bones at
    once. The animation player wants per-bone tracks; this routine just
    turns the bundled list-of-switches into one entry per bone name.
    """
    out: dict[str, tuple[list[int], list[bool]]] = {}
    for keyframe in keyframes:
        for switch in keyframe.switches:
            key = vmd_bone_key(switch.bone_name)
            frames, enabled = out.setdefault(key, ([], []))
            frames.append(int(keyframe.frame))
            enabled.append(bool(switch.enabled))
    return out


class VmdImporter:
    """Loads ``.vmd`` files into :class:`VmdMotionAsset` or
    :class:`AnimationDocument` depending on the entry point."""

    supported_extensions: tuple[str, ...] = (".vmd",)

    def load(self, path: Path) -> VmdMotionAsset:
        """Read + adapt for *playback*: returns a frozen motion asset."""
        path = path.resolve()
        if not path.is_file():
            raise MalformedAssetError(f"VMD file not found: {path}")
        return build_motion_asset(parse_vmd(path.read_bytes()))

    def load_document(self, path: Path):     # noqa: ANN201 — late-bound to dodge import cycles
        """Read + adapt for *editing*: returns a mutable :class:`AnimationDocument`.

        The document keeps the parser's record types verbatim so a load
        → save round-trip is byte-identical, while exposing the per-
        keyframe edit operations the timeline editor + undo stack
        rely on.
        """
        from posecascade.animation.document import AnimationDocument  # noqa: PLC0415
        path = path.resolve()
        if not path.is_file():
            raise MalformedAssetError(f"VMD file not found: {path}")
        return AnimationDocument.from_motion(parse_vmd(path.read_bytes()))
