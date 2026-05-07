"""Engine-facing VMD motion asset and per-bone animation tracks.

This module is intentionally free of imports from the ``importers/vmd/``
plugin package — those types live in :mod:`vmd.types` and the
:func:`build_motion_asset` bridge lives in :mod:`vmd.importer` to avoid
the engine depending on a specific importer plugin (and to dodge the
circular import the other direction would introduce).

Bone-name matching: VMD truncates names to 15 SJIS bytes. PMX bones
arrive full-Unicode; we round-trip both through the same truncation so
the lookup keys collide regardless of source.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from posecascade.animation.vmd_curves import evaluate_bezier
from posecascade.utils.math3d import (
    Quat,
    Vec3,
    quat_from_euler_xyz,
    quat_identity,
    quat_slerp,
    quat_to_euler_xyz,
    vec3,
)

_VMD_NAME_BYTE_LIMIT = 15
# Frames-per-second is fixed in the VMD spec. The engine duplicates the
# constant here (rather than importing it from the format-internal module)
# so the engine layer has no dependency on the importer plugin.
VMD_FRAMES_PER_SECOND = 30


def vmd_bone_key(name: str) -> str:
    """Return the canonical lookup key for a PMX or VMD bone name.

    MMD truncates VMD bone names to 15 SJIS bytes; PMX bones are stored
    full-width Unicode. We round-trip both through ``cp932`` truncation
    so the lookup table keys match regardless of where the name came
    from.
    """
    encoded = name.encode("cp932", errors="replace")[:_VMD_NAME_BYTE_LIMIT]
    nul = encoded.find(b"\x00")
    if nul >= 0:
        encoded = encoded[:nul]
    return encoded.decode("cp932", errors="replace")


@dataclass(frozen=True)
class VmdBoneTrack:
    """Per-bone keyframe track with bezier interpolation handles per segment.

    ``bezier_handles[i]`` describes the curve *entering* keyframe ``i``
    (the segment from keyframe ``i-1`` to ``i``). Keyframe ``0``'s entry
    is unused but kept in the array for index alignment.
    """

    name_key: str
    frames: NDArray[np.uint32]              # (K,)
    positions: NDArray[np.float32]          # (K, 3)
    rotations: NDArray[np.float32]          # (K, 4) xyzw
    bezier_handles: NDArray[np.uint8]       # (K, 4, 4)

    def sample(self, frame: float) -> tuple[Vec3, Quat]:
        """Evaluate the track at a possibly-fractional ``frame`` value.

        Clamps to the first / last keyframe's value when ``frame`` is
        outside the recorded range.
        """
        if self.frames.size == 0:
            return vec3(0.0, 0.0, 0.0), quat_identity()
        if frame <= float(self.frames[0]):
            return self.positions[0].copy(), self.rotations[0].copy()
        if frame >= float(self.frames[-1]):
            return self.positions[-1].copy(), self.rotations[-1].copy()
        index_high = int(np.searchsorted(self.frames, frame, side="right"))
        index_low = index_high - 1
        f_lo = float(self.frames[index_low])
        f_hi = float(self.frames[index_high])
        t = (frame - f_lo) / (f_hi - f_lo)
        handles = self.bezier_handles[index_high]
        position = self._sample_position(t, index_low, index_high, handles)
        rotation = quat_slerp(
            self.rotations[index_low].astype(np.float32, copy=False),
            self.rotations[index_high].astype(np.float32, copy=False),
            evaluate_bezier(_handle_tuple(handles[3]), t),
        )
        return position, rotation

    def _sample_position(
        self,
        t: float,
        index_low: int,
        index_high: int,
        handles: NDArray[np.uint8],
    ) -> Vec3:
        pos_lo = self.positions[index_low]
        pos_hi = self.positions[index_high]
        eased = (
            evaluate_bezier(_handle_tuple(handles[0]), t),
            evaluate_bezier(_handle_tuple(handles[1]), t),
            evaluate_bezier(_handle_tuple(handles[2]), t),
        )
        return np.array(
            [
                pos_lo[0] + (pos_hi[0] - pos_lo[0]) * eased[0],
                pos_lo[1] + (pos_hi[1] - pos_lo[1]) * eased[1],
                pos_lo[2] + (pos_hi[2] - pos_lo[2]) * eased[2],
            ],
            dtype=np.float32,
        )


def _handle_tuple(row: NDArray[np.uint8]) -> tuple[int, int, int, int]:
    return int(row[0]), int(row[1]), int(row[2]), int(row[3])


@dataclass(frozen=True)
class VmdMorphTrack:
    """Per-morph linear-weight track from a VMD file.

    Unlike bone tracks, VMD morph keyframes don't carry bezier handles —
    interpolation between keyframes is plain linear. ``name_key`` is the
    truncated SJIS form so it can be matched against PMX morphs through
    the same :func:`vmd_bone_key` round-trip the bone tracks use.
    """

    name_key: str
    frames: NDArray[np.uint32]
    weights: NDArray[np.float32]

    def sample(self, frame: float) -> float:
        """Evaluate the track's weight at ``frame`` (clamps outside the range)."""
        if self.frames.size == 0:
            return 0.0
        if frame <= float(self.frames[0]):
            return float(self.weights[0])
        if frame >= float(self.frames[-1]):
            return float(self.weights[-1])
        index_high = int(np.searchsorted(self.frames, frame, side="right"))
        index_low = index_high - 1
        f_lo = float(self.frames[index_low])
        f_hi = float(self.frames[index_high])
        t = (frame - f_lo) / (f_hi - f_lo)
        weight_low = float(self.weights[index_low])
        weight_high = float(self.weights[index_high])
        return weight_low + (weight_high - weight_low) * t


@dataclass(frozen=True)
class VmdIkSwitchTrack:
    """Per-IK-bone on/off track from a VMD ``IK`` segment.

    Step interpolation: the current state at any frame is the ``enabled``
    value of the latest keyframe at or before that frame. The convention
    before the first keyframe is "enabled" (matches MMD's default — IK
    starts on unless explicitly disabled by frame 0).
    """

    name_key: str
    frames: NDArray[np.uint32]
    enabled: NDArray[np.bool_]

    def sample(self, frame: float) -> bool:
        """Return the IK on/off state for ``frame``.

        Uses ``side='right'`` semantics so a keyframe at frame N applies
        starting at frame N (not from the next sample after N).
        """
        if self.frames.size == 0:
            return True
        if frame < float(self.frames[0]):
            return True
        index_high = int(np.searchsorted(self.frames, frame, side="right"))
        return bool(self.enabled[index_high - 1])


@dataclass(frozen=True)
class VmdCameraTrack:
    """VMD camera track — every camera keyframe in one frame-sorted record.

    Each keyframe carries six bezier handles (position X/Y/Z, rotation,
    distance, FOV). Position components ease independently per channel;
    rotation uses a single bezier curve fed through quaternion slerp so
    XYZ Euler interpolations don't gimbal-lock between similar
    orientations. ``perspective_off`` is a binary flag and uses step
    interpolation (the value at the latest ``frame ≤ t`` wins).
    """

    frames: NDArray[np.uint32]                  # (K,)
    targets: NDArray[np.float32]                # (K, 3)
    rotations: NDArray[np.float32]              # (K, 3) Euler XYZ radians
    distances: NDArray[np.float32]              # (K,)
    fovs: NDArray[np.float32]                   # (K,) degrees
    perspective_offs: NDArray[np.bool_]         # (K,)
    bezier_handles: NDArray[np.uint8]           # (K, 6, 4)

    def sample(self, frame: float) -> CameraSample:
        """Evaluate the camera state at a possibly-fractional frame."""
        if self.frames.size == 0:
            return _default_camera_sample()
        if frame <= float(self.frames[0]):
            return _camera_sample_at(self, 0)
        if frame >= float(self.frames[-1]):
            return _camera_sample_at(self, self.frames.size - 1)
        index_high = int(np.searchsorted(self.frames, frame, side="right"))
        index_low = index_high - 1
        f_lo = float(self.frames[index_low])
        f_hi = float(self.frames[index_high])
        t = (frame - f_lo) / (f_hi - f_lo)
        handles = self.bezier_handles[index_high]
        return _interpolate_camera(self, index_low, index_high, t, handles)


@dataclass(frozen=True)
class CameraSample:
    """Resolved camera state at one frame."""

    target: Vec3
    rotation_xyz: tuple[float, float, float]
    distance: float
    fov_degrees: float
    perspective_off: bool


@dataclass(frozen=True)
class VmdLightTrack:
    """VMD directional-light track. Color and direction interpolate linearly."""

    frames: NDArray[np.uint32]                  # (K,)
    colors: NDArray[np.float32]                 # (K, 3)
    directions: NDArray[np.float32]             # (K, 3)

    def sample(self, frame: float) -> LightSample:
        if self.frames.size == 0:
            return LightSample(
                color=vec3(1.0, 1.0, 1.0),
                direction=vec3(0.0, -1.0, 0.0),
            )
        if frame <= float(self.frames[0]):
            return LightSample(
                color=self.colors[0].copy(),
                direction=self.directions[0].copy(),
            )
        if frame >= float(self.frames[-1]):
            return LightSample(
                color=self.colors[-1].copy(),
                direction=self.directions[-1].copy(),
            )
        index_high = int(np.searchsorted(self.frames, frame, side="right"))
        index_low = index_high - 1
        f_lo = float(self.frames[index_low])
        f_hi = float(self.frames[index_high])
        t = (frame - f_lo) / (f_hi - f_lo)
        color = self.colors[index_low] + (self.colors[index_high] - self.colors[index_low]) * t
        direction = (
            self.directions[index_low]
            + (self.directions[index_high] - self.directions[index_low]) * t
        )
        return LightSample(
            color=color.astype(np.float32, copy=False),
            direction=direction.astype(np.float32, copy=False),
        )


@dataclass(frozen=True)
class LightSample:
    color: Vec3
    direction: Vec3


@dataclass(frozen=True)
class VmdSelfShadowTrack:
    """VMD self-shadow on/off / distance track. Step interpolation."""

    frames: NDArray[np.uint32]                  # (K,)
    modes: NDArray[np.uint8]                    # (K,)  — 0/1/2
    distances: NDArray[np.float32]              # (K,)

    def sample(self, frame: float) -> SelfShadowSample:
        if self.frames.size == 0:
            return SelfShadowSample(mode=1, distance=0.0)
        if frame < float(self.frames[0]):
            return SelfShadowSample(mode=int(self.modes[0]), distance=float(self.distances[0]))
        index_high = int(np.searchsorted(self.frames, frame, side="right"))
        index = index_high - 1
        return SelfShadowSample(
            mode=int(self.modes[index]),
            distance=float(self.distances[index]),
        )


@dataclass(frozen=True)
class SelfShadowSample:
    mode: int
    distance: float


def _default_camera_sample() -> CameraSample:
    return CameraSample(
        target=vec3(0.0, 0.0, 0.0),
        rotation_xyz=(0.0, 0.0, 0.0),
        distance=-30.0,
        fov_degrees=30.0,
        perspective_off=False,
    )


def _camera_sample_at(track: VmdCameraTrack, index: int) -> CameraSample:
    return CameraSample(
        target=track.targets[index].copy(),
        rotation_xyz=tuple(float(v) for v in track.rotations[index]),  # type: ignore[arg-type]
        distance=float(track.distances[index]),
        fov_degrees=float(track.fovs[index]),
        perspective_off=bool(track.perspective_offs[index]),
    )


def _interpolate_camera(
    track: VmdCameraTrack,
    index_low: int,
    index_high: int,
    t: float,
    handles: NDArray[np.uint8],
) -> CameraSample:
    target = _interpolate_xyz_with_bezier(
        track.targets[index_low], track.targets[index_high], t, handles[:3],
    )
    rotation_xyz = _interpolate_rotation_quat_slerp(
        track.rotations[index_low], track.rotations[index_high], t, handles[3],
    )
    distance = _lerp_with_bezier(
        float(track.distances[index_low]),
        float(track.distances[index_high]),
        t, handles[4],
    )
    fov = _lerp_with_bezier(
        float(track.fovs[index_low]),
        float(track.fovs[index_high]),
        t, handles[5],
    )
    return CameraSample(
        target=target,
        rotation_xyz=rotation_xyz,
        distance=distance,
        fov_degrees=fov,
        # ``perspective_off`` is binary; steps from the lower keyframe.
        perspective_off=bool(track.perspective_offs[index_low]),
    )


def _interpolate_xyz_with_bezier(
    lo: NDArray[np.float32],
    hi: NDArray[np.float32],
    t: float,
    channel_handles: NDArray[np.uint8],   # (3, 4)
) -> Vec3:
    out = np.empty(3, dtype=np.float32)
    for axis in range(3):
        eased = evaluate_bezier(_handles_row(channel_handles[axis]), t)
        out[axis] = float(lo[axis] + (hi[axis] - lo[axis]) * eased)
    return out


def _lerp_with_bezier(
    lo: float, hi: float, t: float, handles_row: NDArray[np.uint8],
) -> float:
    eased = evaluate_bezier(_handles_row(handles_row), t)
    return lo + (hi - lo) * eased


def _interpolate_rotation_quat_slerp(
    lo_euler: NDArray[np.float32],
    hi_euler: NDArray[np.float32],
    t: float,
    handles_row: NDArray[np.uint8],
) -> tuple[float, float, float]:
    eased = evaluate_bezier(_handles_row(handles_row), t)
    lo_quat = quat_from_euler_xyz(*lo_euler)
    hi_quat = quat_from_euler_xyz(*hi_euler)
    blended = quat_slerp(lo_quat, hi_quat, eased)
    rx, ry, rz = quat_to_euler_xyz(blended)
    return float(rx), float(ry), float(rz)


def _handles_row(row: NDArray[np.uint8]) -> tuple[int, int, int, int]:
    return int(row[0]), int(row[1]), int(row[2]), int(row[3])


@dataclass(frozen=True)
class VmdMotionAsset:
    """The complete VMD motion as engine-friendly tracks + opaque extras.

    Phase 4 consumes ``bone_tracks`` and ``morph_tracks``; later phases pick
    up the camera / light / IK streams. Those non-driven streams are stored
    as opaque tuples (the importer hands over its own internal types) so
    the engine doesn't need to import :mod:`vmd.types`.
    """

    target_model_name: str
    bone_tracks: tuple[VmdBoneTrack, ...] = field(default_factory=tuple)
    morph_tracks: tuple[VmdMorphTrack, ...] = field(default_factory=tuple)
    ik_tracks: tuple[VmdIkSwitchTrack, ...] = field(default_factory=tuple)
    camera_track: VmdCameraTrack | None = None
    light_track: VmdLightTrack | None = None
    self_shadow_track: VmdSelfShadowTrack | None = None

    @property
    def duration_frames(self) -> int:
        """Highest frame number across every track — the clip's duration."""
        max_frame = 0
        for track in self.bone_tracks:
            if track.frames.size:
                max_frame = max(max_frame, int(track.frames[-1]))
        for track in self.morph_tracks:
            if track.frames.size:
                max_frame = max(max_frame, int(track.frames[-1]))
        return max_frame


def build_morph_track(
    name_key: str,
    frames: list[int],
    weights: list[float],
) -> VmdMorphTrack:
    """Pack per-keyframe lists into a frame-sorted :class:`VmdMorphTrack`."""
    if len(frames) != len(weights):
        raise ValueError("morph-track frame and weight arrays have mismatched lengths")
    order = np.argsort(np.asarray(frames, dtype=np.int64))
    return VmdMorphTrack(
        name_key=name_key,
        frames=np.asarray(frames, dtype=np.uint32)[order],
        weights=np.asarray(weights, dtype=np.float32)[order],
    )


def build_camera_track(
    frames: list[int],
    targets: list[tuple[float, float, float]],
    rotations: list[tuple[float, float, float]],
    distances: list[float],
    fovs: list[float],
    perspective_offs: list[bool],
    bezier_handles: list[tuple[tuple[int, int, int, int], ...]],
) -> VmdCameraTrack:
    """Pack VMD camera keyframes into a frame-sorted track.

    All input lists must be the same length; ``bezier_handles[i]`` is the
    six-channel block (X / Y / Z / rotation / distance / FOV) attached to
    the ``i``-th keyframe.
    """
    counts = (
        len(frames), len(targets), len(rotations),
        len(distances), len(fovs), len(perspective_offs), len(bezier_handles),
    )
    if len(set(counts)) != 1:
        raise ValueError(f"camera track input arrays have mismatched lengths: {counts}")
    order = np.argsort(np.asarray(frames, dtype=np.int64))
    return VmdCameraTrack(
        frames=np.asarray(frames, dtype=np.uint32)[order],
        targets=np.asarray(targets, dtype=np.float32)[order],
        rotations=np.asarray(rotations, dtype=np.float32)[order],
        distances=np.asarray(distances, dtype=np.float32)[order],
        fovs=np.asarray(fovs, dtype=np.float32)[order],
        perspective_offs=np.asarray(perspective_offs, dtype=np.bool_)[order],
        bezier_handles=np.asarray(bezier_handles, dtype=np.uint8)[order],
    )


def build_light_track(
    frames: list[int],
    colors: list[tuple[float, float, float]],
    directions: list[tuple[float, float, float]],
) -> VmdLightTrack:
    counts = (len(frames), len(colors), len(directions))
    if len(set(counts)) != 1:
        raise ValueError(f"light track input arrays have mismatched lengths: {counts}")
    order = np.argsort(np.asarray(frames, dtype=np.int64))
    return VmdLightTrack(
        frames=np.asarray(frames, dtype=np.uint32)[order],
        colors=np.asarray(colors, dtype=np.float32)[order],
        directions=np.asarray(directions, dtype=np.float32)[order],
    )


def build_self_shadow_track(
    frames: list[int],
    modes: list[int],
    distances: list[float],
) -> VmdSelfShadowTrack:
    counts = (len(frames), len(modes), len(distances))
    if len(set(counts)) != 1:
        raise ValueError(
            f"self-shadow track input arrays have mismatched lengths: {counts}"
        )
    order = np.argsort(np.asarray(frames, dtype=np.int64))
    return VmdSelfShadowTrack(
        frames=np.asarray(frames, dtype=np.uint32)[order],
        modes=np.asarray(modes, dtype=np.uint8)[order],
        distances=np.asarray(distances, dtype=np.float32)[order],
    )


def build_ik_track(
    name_key: str,
    frames: list[int],
    enabled: list[bool],
) -> VmdIkSwitchTrack:
    """Pack per-keyframe (frame, enabled) lists into a sorted :class:`VmdIkSwitchTrack`."""
    if len(frames) != len(enabled):
        raise ValueError("ik-track frame and enabled arrays have mismatched lengths")
    order = np.argsort(np.asarray(frames, dtype=np.int64))
    return VmdIkSwitchTrack(
        name_key=name_key,
        frames=np.asarray(frames, dtype=np.uint32)[order],
        enabled=np.asarray(enabled, dtype=np.bool_)[order],
    )


def build_bone_track(
    name_key: str,
    frames: list[int],
    positions: list[tuple[float, float, float]],
    rotations: list[tuple[float, float, float, float]],
    bezier_handles: list[tuple[tuple[int, int, int, int], ...]],
) -> VmdBoneTrack:
    """Pack per-keyframe lists into a frame-sorted :class:`VmdBoneTrack`.

    Used by :func:`vmd.importer.build_motion_asset`; isolated here so unit
    tests can build tracks without touching the file format.
    """
    if not (len(frames) == len(positions) == len(rotations) == len(bezier_handles)):
        raise ValueError("bone-track input arrays have mismatched lengths")
    order = np.argsort(np.asarray(frames, dtype=np.int64))
    sorted_frames = np.asarray(frames, dtype=np.uint32)[order]
    sorted_positions = np.asarray(positions, dtype=np.float32)[order]
    sorted_rotations = np.asarray(rotations, dtype=np.float32)[order]
    sorted_bezier = np.asarray(bezier_handles, dtype=np.uint8)[order]
    return VmdBoneTrack(
        name_key=name_key,
        frames=sorted_frames,
        positions=sorted_positions,
        rotations=sorted_rotations,
        bezier_handles=sorted_bezier,
    )
