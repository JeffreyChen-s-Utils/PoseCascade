"""VMD scene-state driver: camera + light + self-shadow per frame.

The bone player handles per-bone, per-morph, and physics state. Camera /
light / self-shadow are scene-wide and consumed by the render layer
rather than the scene graph, so they live behind a separate driver
instead of bolting more channels onto :class:`VmdAnimationPlayer`.

The integrator (UI / viewport) calls
:meth:`VmdSceneDriver.camera_at(time_seconds)` (etc.) right before
:meth:`Renderer.draw` to refresh the live :class:`Camera` /
:class:`DirectionalLight` / :class:`SelfShadowState`. Construction is
cheap: the driver just holds a reference to the motion asset and the
:class:`VMD_FRAMES_PER_SECOND` conversion is done inline.
"""
from __future__ import annotations

from dataclasses import dataclass

from posecascade.animation.vmd_track import (
    VMD_FRAMES_PER_SECOND,
    VmdMotionAsset,
)
from posecascade.render.camera import Camera
from posecascade.render.constants import (
    DEFAULT_FAR_PLANE,
    DEFAULT_NEAR_PLANE,
)
from posecascade.render.lighting import DirectionalLight, SelfShadowState


@dataclass
class VmdSceneDriver:
    """Resolves VMD camera / light / self-shadow state at a given time."""

    motion: VmdMotionAsset
    near: float = DEFAULT_NEAR_PLANE
    far: float = DEFAULT_FAR_PLANE

    @property
    def has_camera_track(self) -> bool:
        return self.motion.camera_track is not None

    @property
    def has_light_track(self) -> bool:
        return self.motion.light_track is not None

    @property
    def has_self_shadow_track(self) -> bool:
        return self.motion.self_shadow_track is not None

    def camera_at(self, time_seconds: float) -> Camera | None:
        """Return a fully-configured :class:`Camera` for ``time_seconds``.

        Falls through to ``None`` when the motion has no camera track —
        the caller can fall back to its scene-graph default camera.
        """
        track = self.motion.camera_track
        if track is None:
            return None
        sample = track.sample(self._frame_for(time_seconds))
        return Camera.from_vmd_state(
            target=sample.target,
            rotation_xyz=sample.rotation_xyz,
            distance=sample.distance,
            fov_degrees=sample.fov_degrees,
            perspective_off=sample.perspective_off,
            near=self.near,
            far=self.far,
        )

    def light_at(self, time_seconds: float) -> DirectionalLight | None:
        track = self.motion.light_track
        if track is None:
            return None
        sample = track.sample(self._frame_for(time_seconds))
        return DirectionalLight(color=sample.color, direction=sample.direction)

    def self_shadow_at(self, time_seconds: float) -> SelfShadowState | None:
        track = self.motion.self_shadow_track
        if track is None:
            return None
        sample = track.sample(self._frame_for(time_seconds))
        return SelfShadowState(mode=sample.mode, distance=sample.distance)

    def _frame_for(self, time_seconds: float) -> float:
        return max(0.0, time_seconds * VMD_FRAMES_PER_SECOND)
