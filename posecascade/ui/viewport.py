"""Qt OpenGL viewport widget — Blender-style camera + model interaction.

Mouse controls
--------------
* **Middle-/Right-button drag** — orbit camera around the look-at point.
* **Shift + Middle-/Right-button drag** — pan camera and look-at together
  (slides parallel to the screen plane).
* **Mouse wheel** — zoom (changes camera ↔ look-at distance).
* **Left-button click on a model** — picks that top-level holder; subsequent
  drag while the button is held translates the holder in screen space at
  its current depth, so it follows the cursor.

Pitch is clamped to ±89° and orbit distance to ``[_MIN_DISTANCE, _MAX_DISTANCE]``
to avoid gimbal flip and runaway zoom. The orbit state is re-synced from the
camera's current pose on every mouse press, so external code (CLI flags,
scripts) that moves the camera between drags doesn't desync the state.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from posecascade.gl.context import GLContext, claim_current_thread
from posecascade.render.camera import Camera
from posecascade.render.renderer import Renderer, _world_matrix
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.utils.logging import get_logger

_log = get_logger(__name__)

_ORBIT_SENSITIVITY = 0.005      # radians per pixel of mouse drag
_PITCH_LIMIT_DEGREES = 89.0     # clamp to avoid gimbal flip
_MIN_DISTANCE = 0.1             # closest the camera can zoom to its target
_MAX_DISTANCE = 10_000.0        # furthest the camera can zoom out
_ZOOM_PER_NOTCH = 0.9           # one wheel notch ≈ 10% closer
_ZOOM_NOTCH_LIMIT = 200.0       # cap notches per call to avoid pow() overflow
_MIN_DRAG_DEPTH = 0.1           # minimum holder distance for translate-drag
_DRAG_NONE = "none"
_DRAG_ORBIT = "orbit"
_DRAG_PAN = "pan"
_DRAG_TRANSLATE = "translate"
# Karaoke overlay typography. Pulled to module-level constants per
# the magic-number lint rule and to make tweaking the overlay's look
# a one-line change.
_OVERLAY_FONT_PT = 22
_OVERLAY_BOTTOM_MARGIN_PX = 32
_VEC_NORMALIZE_EPS = 1.0e-6     # treat shorter vectors as degenerate
_DEGENERATE_DISTANCE = 1.0e-4   # camera offset shorter than this is treated as zero


class Viewport(QOpenGLWidget):
    """``QOpenGLWidget`` that hosts the engine's render passes + interactive controls."""

    initialized = Signal(object)  # GLContext

    def __init__(self, parent: object = None) -> None:
        super().__init__(parent)
        self._gl_context: GLContext | None = None
        self._renderer: Renderer | None = None
        self._scene: Scene | None = None
        self._camera: Camera = Camera()
        self._cloth_host: object | None = None
        self._shaders_root: Path = _detect_shaders_root()
        self._drag_mode: str = _DRAG_NONE
        self._last_mouse_xy: tuple[int, int] | None = None
        self._selected_holder: Node | None = None
        self._orbit_yaw: float = 0.0
        self._orbit_pitch: float = 0.0
        self._orbit_distance: float = 1.0
        # Karaoke overlay text drawn on top of the GL pass each frame.
        # Empty string → overlay pass skipped.
        self._overlay_text: str = ""

    def set_scene(self, scene: Scene) -> None:
        self._scene = scene
        self._selected_holder = None
        self.update()

    def set_cloth_host(self, host: object | None) -> None:
        """Bind a cloth simulator so per-frame paintGL streams its positions to GPU."""
        self._cloth_host = host

    @property
    def scene(self) -> Scene | None:
        return self._scene

    @property
    def camera(self) -> Camera:
        return self._camera

    @property
    def renderer(self) -> Renderer | None:
        return self._renderer

    @property
    def selected_holder(self) -> Node | None:
        return self._selected_holder

    def initializeGL(self) -> None:  # noqa: N802 — Qt override
        self._gl_context = claim_current_thread()
        self._renderer = Renderer(shaders_root=self._shaders_root)
        self._renderer.initialize()
        _log.info("renderer initialized on viewport thread")
        self.initialized.emit(self._gl_context)

    def resizeGL(self, width: int, height: int) -> None:  # noqa: N802 — Qt override
        _log.debug("viewport resized to %dx%d", width, height)

    def paintGL(self) -> None:  # noqa: N802 — Qt override
        if self._gl_context is None or self._renderer is None:
            return
        self._gl_context.assert_owned()
        if self._scene is None:
            return
        if self._cloth_host is not None:
            self._renderer.apply_cloth_state(self._cloth_host)
        size = self.size()
        self._renderer.draw(
            self._scene,
            self._camera,
            (int(size.width()), int(size.height())),
        )

    def set_overlay_text(self, text: str) -> None:
        """Set the karaoke / debug overlay text drawn on top of the GL pass.

        Empty string clears the overlay. Triggers a repaint so the new
        text shows on the next frame without waiting for the timer's
        scheduled tick.
        """
        if text == self._overlay_text:
            return
        self._overlay_text = text
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt override
        """Run the GL pass, then draw the karaoke overlay on top.

        ``QOpenGLWidget.paintEvent`` dispatches ``paintGL`` for the 3D
        content; we open a ``QPainter`` on the widget afterward for a
        2D text pass. Skips entirely when no overlay text is set so
        the empty / no-lyric case stays a single-pass paint.
        """
        super().paintEvent(event)
        if not self._overlay_text:
            return
        from PySide6.QtCore import Qt as _Qt  # noqa: PLC0415
        from PySide6.QtGui import QColor, QFont, QPainter  # noqa: PLC0415
        painter = QPainter(self)
        try:
            painter.setFont(QFont("Arial", _OVERLAY_FONT_PT, QFont.Weight.Bold))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                self.rect().adjusted(0, 0, 0, -_OVERLAY_BOTTOM_MARGIN_PX),
                _Qt.AlignmentFlag.AlignBottom | _Qt.AlignmentFlag.AlignHCenter,
                self._overlay_text,
            )
        finally:
            painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt override
        button = event.button()
        modifiers = event.modifiers()
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        position = event.position()
        sx, sy = int(position.x()), int(position.y())
        if button == Qt.MouseButton.LeftButton and not shift:
            holder = self._pick_holder(sx, sy)
            if holder is not None:
                self._selected_holder = holder
                self._drag_mode = _DRAG_TRANSLATE
                self._last_mouse_xy = (sx, sy)
                event.accept()
                return
        elif button in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._sync_orbit_from_camera()
            self._drag_mode = _DRAG_PAN if shift else _DRAG_ORBIT
            self._last_mouse_xy = (sx, sy)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 — Qt override
        if self._drag_mode == _DRAG_NONE or self._last_mouse_xy is None:
            super().mouseMoveEvent(event)
            return
        position = event.position()
        current = (int(position.x()), int(position.y()))
        dx = current[0] - self._last_mouse_xy[0]
        dy = current[1] - self._last_mouse_xy[1]
        self._last_mouse_xy = current
        if self._drag_mode == _DRAG_ORBIT:
            self._apply_orbit_drag(dx, dy)
        elif self._drag_mode == _DRAG_PAN:
            self._apply_pan_drag(dx, dy)
        elif self._drag_mode == _DRAG_TRANSLATE and self._selected_holder is not None:
            self._apply_translate_drag(self._selected_holder, dx, dy)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt override
        if self._drag_mode != _DRAG_NONE:
            self._drag_mode = _DRAG_NONE
            self._last_mouse_xy = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 — Qt override
        delta_y = event.angleDelta().y()
        if delta_y == 0:
            super().wheelEvent(event)
            return
        notches = delta_y / 120.0
        self._sync_orbit_from_camera()
        self._apply_zoom(notches)
        self.update()
        event.accept()

    def _apply_orbit_drag(self, dx: int, dy: int) -> None:
        self._orbit_yaw -= dx * _ORBIT_SENSITIVITY
        self._orbit_pitch -= dy * _ORBIT_SENSITIVITY
        limit = math.radians(_PITCH_LIMIT_DEGREES)
        if self._orbit_pitch > limit:
            self._orbit_pitch = limit
        elif self._orbit_pitch < -limit:
            self._orbit_pitch = -limit
        self._apply_orbit_to_camera()

    def _apply_pan_drag(self, dx: int, dy: int) -> None:
        cam = self._camera
        forward, right, up = _camera_basis(cam)
        height = max(self.height(), 1)
        wpp = _world_per_pixel(cam.fov_degrees, self._orbit_distance, height)
        delta = (-right * dx + up * dy) * wpp
        cam.position = (np.asarray(cam.position, dtype=np.float32) + delta).astype(np.float32)
        cam.target = (np.asarray(cam.target, dtype=np.float32) + delta).astype(np.float32)

    def _apply_zoom(self, notches: float) -> None:
        # Bound notches before raising to a power: pow(0.9, ±N) for N in the
        # tens of thousands overflows Python's float, but the result would
        # land on the clamp range anyway.
        bounded = max(-_ZOOM_NOTCH_LIMIT, min(_ZOOM_NOTCH_LIMIT, notches))
        factor = pow(_ZOOM_PER_NOTCH, bounded)
        new_distance = self._orbit_distance * factor
        self._orbit_distance = max(_MIN_DISTANCE, min(_MAX_DISTANCE, new_distance))
        self._apply_orbit_to_camera()

    def _apply_translate_drag(self, holder: Node, dx: int, dy: int) -> None:
        cam = self._camera
        forward, right, up = _camera_basis(cam)
        holder_world = np.asarray(_world_matrix(holder)[:3, 3], dtype=np.float32)
        cam_to_holder = holder_world - np.asarray(cam.position, dtype=np.float32)
        depth = max(_MIN_DRAG_DEPTH, float(np.dot(cam_to_holder, forward)))
        height = max(self.height(), 1)
        wpp = _world_per_pixel(cam.fov_degrees, depth, height)
        delta = (right * dx - up * dy) * wpp
        # Top-level holders sit directly under composite.root (identity), so
        # adding the world-space delta to the local translation is correct.
        new_translation = np.asarray(holder.transform.translation, dtype=np.float32) + delta
        holder.transform.set_translation(new_translation.astype(np.float32))

    def _pick_holder(self, sx: int, sy: int) -> Node | None:
        if self._scene is None or self._renderer is None:
            return None
        ray = self._screen_to_world_ray(sx, sy)
        if ray is None:
            return None
        origin, direction = ray
        closest_t = float("inf")
        closest_holder: Node | None = None
        for holder in self._scene.root.children:
            bounds = self._renderer.world_aabb_of_subtree(holder)
            if bounds is None:
                continue
            mn, mx = bounds
            t = _ray_aabb_distance(origin, direction, mn, mx)
            if t is not None and 0.0 < t < closest_t:
                closest_t = t
                closest_holder = holder
        return closest_holder

    def _screen_to_world_ray(
        self, sx: int, sy: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        cam = self._camera
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        view = cam.view_matrix()
        proj = cam.projection_matrix(width / height)
        try:
            inv_vp = np.linalg.inv(proj @ view)
        except np.linalg.LinAlgError:
            return None
        ndc_x = (2.0 * sx / width) - 1.0
        ndc_y = 1.0 - (2.0 * sy / height)
        near = inv_vp @ np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
        far = inv_vp @ np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float32)
        near_world = (near[:3] / near[3]).astype(np.float32)
        far_world = (far[:3] / far[3]).astype(np.float32)
        direction = far_world - near_world
        norm = float(np.linalg.norm(direction))
        if norm < _VEC_NORMALIZE_EPS:
            return None
        return near_world, (direction / norm).astype(np.float32)

    def _sync_orbit_from_camera(self) -> None:
        offset = np.asarray(self._camera.position, dtype=np.float32) - np.asarray(
            self._camera.target, dtype=np.float32,
        )
        distance = float(np.linalg.norm(offset))
        if distance < _DEGENERATE_DISTANCE:
            self._orbit_distance = 1.0
            self._orbit_yaw = 0.0
            self._orbit_pitch = 0.0
            return
        self._orbit_distance = distance
        sin_pitch = max(-1.0, min(1.0, float(offset[1]) / distance))
        self._orbit_pitch = math.asin(sin_pitch)
        self._orbit_yaw = math.atan2(float(offset[0]), float(offset[2]))

    def _apply_orbit_to_camera(self) -> None:
        cp = math.cos(self._orbit_pitch)
        sp = math.sin(self._orbit_pitch)
        cy = math.cos(self._orbit_yaw)
        sy = math.sin(self._orbit_yaw)
        offset = np.array(
            [
                self._orbit_distance * cp * sy,
                self._orbit_distance * sp,
                self._orbit_distance * cp * cy,
            ],
            dtype=np.float32,
        )
        target = np.asarray(self._camera.target, dtype=np.float32)
        self._camera.position = (target + offset).astype(np.float32)


def _camera_basis(cam: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the camera's (forward, right, up) basis as unit vectors."""
    forward = np.asarray(cam.target, dtype=np.float32) - np.asarray(cam.position, dtype=np.float32)
    forward_norm = max(float(np.linalg.norm(forward)), 1.0e-6)
    forward = (forward / forward_norm).astype(np.float32)
    right = np.cross(forward, np.asarray(cam.up, dtype=np.float32))
    right_norm = max(float(np.linalg.norm(right)), 1.0e-6)
    right = (right / right_norm).astype(np.float32)
    up = np.cross(right, forward).astype(np.float32)
    return forward, right, up


def _world_per_pixel(fov_degrees: float, depth: float, screen_height: int) -> float:
    """World-space units covered by one screen pixel at ``depth`` from the camera."""
    fov_radians = math.radians(fov_degrees)
    return (2.0 * math.tan(fov_radians / 2.0) * depth) / max(screen_height, 1)


def _ray_aabb_distance(
    origin: np.ndarray, direction: np.ndarray,
    mn: np.ndarray, mx: np.ndarray,
) -> float | None:
    """Return the parametric distance along ``direction`` to enter the AABB.

    Uses the slab method; returns ``None`` for misses or hits behind the
    origin. Treats zero-direction components by replacing them with a tiny
    epsilon — the resulting ±∞ slabs collapse cleanly through the min/max.
    """
    eps = 1.0e-8
    safe_direction = np.where(np.abs(direction) < eps, eps, direction)
    t1 = (mn - origin) / safe_direction
    t2 = (mx - origin) / safe_direction
    t_near = float(np.max(np.minimum(t1, t2)))
    t_far = float(np.min(np.maximum(t1, t2)))
    if t_near > t_far or t_far < 0.0:
        return None
    return t_near if t_near >= 0.0 else t_far


def _detect_shaders_root() -> Path:
    """Locate the bundled ``shaders/`` dir by walking up from this module."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        candidate = parent / "shaders"
        if candidate.is_dir():
            return candidate
    return here.parent / "shaders"
