"""Perspective / orthographic camera with VMD-style construction support."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from posecascade.render.constants import (
    DEFAULT_FAR_PLANE,
    DEFAULT_FOV_DEGREES,
    DEFAULT_NEAR_PLANE,
)
from posecascade.utils.math3d import Mat4, Vec3, quat_from_euler_xyz, quat_rotate_vec, vec3


@dataclass
class Camera:
    """Perspective camera with position / target / up.

    Setting ``orthographic_half_height`` to a positive value flips the
    projection to orthographic — used for VMD's ``perspective_off``
    keyframe state. The half-height matches what the perspective view
    would cover at the camera's distance, so toggling between the two
    modes mid-shot stays approximately the same scale.
    """

    position: Vec3 = None  # type: ignore[assignment]
    target: Vec3 = None  # type: ignore[assignment]
    up: Vec3 = None  # type: ignore[assignment]
    fov_degrees: float = DEFAULT_FOV_DEGREES
    near: float = DEFAULT_NEAR_PLANE
    far: float = DEFAULT_FAR_PLANE
    orthographic_half_height: float | None = None

    @classmethod
    def from_vmd_state(
        cls,
        *,
        target: Vec3,
        rotation_xyz: tuple[float, float, float],
        distance: float,
        fov_degrees: float,
        perspective_off: bool,
        near: float = DEFAULT_NEAR_PLANE,
        far: float = DEFAULT_FAR_PLANE,
    ) -> Camera:
        """Build a camera from the canonical VMD camera-state tuple.

        VMD's convention places the camera at ``target + rotate(rotation,
        (0, 0, distance))`` looking back at ``target``. ``distance`` is
        usually negative — the camera sits behind the target on the local
        ``-Z`` axis. ``perspective_off`` flips to orthographic; the
        ortho half-height is derived from ``|distance| × tan(fov/2)`` so
        the on-screen scale matches the perspective view.
        """
        target_arr = np.asarray(target, dtype=np.float32)
        rotation_quat = quat_from_euler_xyz(*rotation_xyz)
        offset = quat_rotate_vec(
            rotation_quat,
            np.array([0.0, 0.0, float(distance)], dtype=np.float32),
        )
        position = (target_arr + offset).astype(np.float32, copy=False)
        camera_up = quat_rotate_vec(
            rotation_quat,
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ).astype(np.float32, copy=False)
        ortho_half_height: float | None = None
        if perspective_off:
            half_radian = float(np.deg2rad(fov_degrees)) * 0.5
            ortho_half_height = abs(float(distance)) * float(np.tan(half_radian))
        return cls(
            position=position,
            target=target_arr.copy(),
            up=camera_up,
            fov_degrees=float(fov_degrees),
            near=near,
            far=far,
            orthographic_half_height=ortho_half_height,
        )

    def __post_init__(self) -> None:
        if self.position is None:
            self.position = vec3(0.0, 1.0, 3.0)
        if self.target is None:
            self.target = vec3(0.0, 0.0, 0.0)
        if self.up is None:
            self.up = vec3(0.0, 1.0, 0.0)

    def view_matrix(self) -> Mat4:
        """Build the view matrix (right-handed, column-major)."""
        forward = self.target - self.position
        forward = forward / max(float(np.linalg.norm(forward)), 1.0e-6)
        right = np.cross(forward, self.up)
        right = right / max(float(np.linalg.norm(right)), 1.0e-6)
        new_up = np.cross(right, forward)
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = new_up
        view[2, :3] = -forward
        view[0, 3] = -float(np.dot(right, self.position))
        view[1, 3] = -float(np.dot(new_up, self.position))
        view[2, 3] = float(np.dot(forward, self.position))
        return view

    def projection_matrix(self, aspect: float) -> Mat4:
        """Build the projection matrix for the given viewport aspect.

        Returns an orthographic frustum when
        :attr:`orthographic_half_height` is set, otherwise the standard
        perspective matrix derived from :attr:`fov_degrees`.
        """
        if self.orthographic_half_height is not None:
            return self._orthographic_matrix(aspect)
        fov_radians = float(np.deg2rad(self.fov_degrees))
        f = 1.0 / float(np.tan(fov_radians / 2.0))
        depth = self.near - self.far
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / depth
        proj[2, 3] = (2.0 * self.far * self.near) / depth
        proj[3, 2] = -1.0
        return proj

    def _orthographic_matrix(self, aspect: float) -> Mat4:
        """Return the orthographic projection matrix sized to match the
        perspective view at the camera's current distance."""
        half_height = float(self.orthographic_half_height or 1.0)
        half_width = half_height * aspect
        depth = self.far - self.near
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 1.0 / half_width
        proj[1, 1] = 1.0 / half_height
        proj[2, 2] = -2.0 / depth
        proj[2, 3] = -(self.far + self.near) / depth
        proj[3, 3] = 1.0
        return proj
