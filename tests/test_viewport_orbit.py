"""Tests for the right-click-drag orbit camera in :mod:`posecascade.ui.viewport`.

Covers the pure orbit math (`_sync_orbit_from_camera` ↔ `_apply_orbit_to_camera`)
and the drag dispatch (`_apply_orbit_drag` → camera position updates). No real Qt
event loop needed — the viewport is constructed under the ``qapp`` fixture.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from posecascade.utils.math3d import vec3


@pytest.fixture
def viewport(qapp: object) -> object:
    pytest.importorskip("PySide6.QtOpenGLWidgets")
    from posecascade.ui.viewport import Viewport  # noqa: PLC0415

    vp = Viewport()
    vp.camera.position = vec3(3.0, 1.6, 5.0)
    vp.camera.target = vec3(0.5, 1.4, 0.5)
    return vp


def test_round_trip_position(viewport: object) -> None:
    initial_position = viewport.camera.position.copy()
    viewport._sync_orbit_from_camera()  # noqa: SLF001
    viewport._apply_orbit_to_camera()  # noqa: SLF001
    np.testing.assert_allclose(viewport.camera.position, initial_position, atol=1.0e-5)


def test_yaw_drag_rotates_camera_around_target(viewport: object) -> None:
    target = viewport.camera.target.copy()
    initial_dist = float(np.linalg.norm(viewport.camera.position - target))
    viewport._sync_orbit_from_camera()  # noqa: SLF001
    viewport._apply_orbit_drag(dx=100, dy=0)  # noqa: SLF001 — pure horizontal drag
    after_dist = float(np.linalg.norm(viewport.camera.position - target))
    np.testing.assert_allclose(after_dist, initial_dist, atol=1.0e-4)


def test_pitch_clamped(viewport: object) -> None:
    viewport._sync_orbit_from_camera()  # noqa: SLF001
    viewport._apply_orbit_drag(dx=0, dy=10_000)  # noqa: SLF001 — way past the limit
    assert viewport._orbit_pitch <= -math.radians(89.0) + 1.0e-3  # noqa: SLF001
    assert viewport._orbit_pitch >= -math.radians(89.0) - 1.0e-3  # noqa: SLF001


def test_pitch_clamped_upward(viewport: object) -> None:
    viewport._sync_orbit_from_camera()  # noqa: SLF001
    viewport._apply_orbit_drag(dx=0, dy=-10_000)  # noqa: SLF001
    assert abs(viewport._orbit_pitch - math.radians(89.0)) < 1.0e-3  # noqa: SLF001


def test_zero_distance_does_not_blow_up(qapp: object) -> None:
    pytest.importorskip("PySide6.QtOpenGLWidgets")
    from posecascade.ui.viewport import Viewport  # noqa: PLC0415

    vp = Viewport()
    vp.camera.position = vec3(0.0, 0.0, 0.0)
    vp.camera.target = vec3(0.0, 0.0, 0.0)
    vp._sync_orbit_from_camera()  # noqa: SLF001 — must not raise / NaN
    assert vp._orbit_distance == 1.0  # noqa: SLF001 — fallback default
    assert vp._orbit_pitch == 0.0  # noqa: SLF001
    assert vp._orbit_yaw == 0.0  # noqa: SLF001


def test_drag_recovers_after_external_camera_move(viewport: object) -> None:
    """A scripted camera move between drags should not break orbit tracking."""
    viewport._sync_orbit_from_camera()  # noqa: SLF001
    viewport._apply_orbit_drag(dx=20, dy=10)  # noqa: SLF001
    # External code teleports the camera (e.g. scene_compose changes target).
    viewport.camera.position = vec3(6.0, 2.0, 6.0)
    viewport.camera.target = vec3(2.0, 1.0, 2.0)
    # Next press syncs orbit from the new pose; round-trip stays consistent.
    viewport._sync_orbit_from_camera()  # noqa: SLF001
    viewport._apply_orbit_to_camera()  # noqa: SLF001
    np.testing.assert_allclose(viewport.camera.position, [6.0, 2.0, 6.0], atol=1.0e-5)


def test_pan_translates_camera_and_target_together(viewport: object) -> None:
    """Pan must move position and target by the same vector — relative pose stays the same."""
    cam_offset_before = viewport.camera.position - viewport.camera.target
    viewport._sync_orbit_from_camera()  # noqa: SLF001
    viewport._apply_pan_drag(dx=50, dy=30)  # noqa: SLF001
    cam_offset_after = viewport.camera.position - viewport.camera.target
    np.testing.assert_allclose(cam_offset_after, cam_offset_before, atol=1.0e-5)


def test_zoom_in_decreases_distance(viewport: object) -> None:
    viewport._sync_orbit_from_camera()  # noqa: SLF001
    distance_before = viewport._orbit_distance  # noqa: SLF001
    viewport._apply_zoom(notches=3.0)  # noqa: SLF001 — three wheel notches in
    assert viewport._orbit_distance < distance_before  # noqa: SLF001


def test_zoom_clamped_to_minimum(viewport: object) -> None:
    viewport._sync_orbit_from_camera()  # noqa: SLF001
    viewport._apply_zoom(notches=10_000.0)  # noqa: SLF001 — way past min
    assert viewport._orbit_distance >= 0.099  # noqa: SLF001 — at the floor


def test_zoom_clamped_to_maximum(viewport: object) -> None:
    viewport._sync_orbit_from_camera()  # noqa: SLF001
    viewport._apply_zoom(notches=-10_000.0)  # noqa: SLF001 — way past max
    assert viewport._orbit_distance <= 10_001  # noqa: SLF001 — at the ceiling


def test_translate_drag_moves_holder(qapp: object) -> None:
    """Left-button drag on a holder must offset its translation along screen axes."""
    pytest.importorskip("PySide6.QtOpenGLWidgets")
    from posecascade.scene.node import Node  # noqa: PLC0415
    from posecascade.ui.viewport import Viewport  # noqa: PLC0415

    vp = Viewport()
    vp.camera.position = vec3(0.0, 0.0, 5.0)
    vp.camera.target = vec3(0.0, 0.0, 0.0)
    vp.resize(800, 600)
    holder = Node(name="cube")
    holder.transform.set_translation(vec3(0.0, 0.0, 0.0))
    before = holder.transform.translation.copy()
    vp._apply_translate_drag(holder, dx=100, dy=0)  # noqa: SLF001
    after = holder.transform.translation
    # Camera looks down -Z, so a +X mouse drag moves holder along world +X.
    assert after[0] > before[0]
    np.testing.assert_allclose(after[1], before[1], atol=1.0e-5)
    np.testing.assert_allclose(after[2], before[2], atol=1.0e-5)


def test_ray_aabb_distance_hit() -> None:
    from posecascade.ui.viewport import _ray_aabb_distance  # noqa: PLC0415

    origin = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    mn = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    mx = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    t = _ray_aabb_distance(origin, direction, mn, mx)
    assert t is not None
    np.testing.assert_allclose(t, 4.0, atol=1.0e-5)  # near face at z=1, distance 5-1=4


def test_ray_aabb_distance_miss() -> None:
    from posecascade.ui.viewport import _ray_aabb_distance  # noqa: PLC0415

    origin = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    mn = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    mx = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    assert _ray_aabb_distance(origin, direction, mn, mx) is None


def test_ray_aabb_distance_behind() -> None:
    from posecascade.ui.viewport import _ray_aabb_distance  # noqa: PLC0415

    origin = np.array([0.0, 0.0, -5.0], dtype=np.float32)
    direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # away from box
    mn = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    mx = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    assert _ray_aabb_distance(origin, direction, mn, mx) is None
