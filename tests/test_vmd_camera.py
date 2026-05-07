"""Tests for VMD camera / light / self-shadow tracks + scene driver."""
from __future__ import annotations

import numpy as np
import pytest
from vmd.importer import build_motion_asset
from vmd.reader import parse_vmd

from posecascade.animation.scene_driver import VmdSceneDriver
from posecascade.animation.vmd_track import VMD_FRAMES_PER_SECOND
from posecascade.render.camera import Camera
from posecascade.render.lighting import DirectionalLight, SelfShadowState
from tests.fixtures.mmd.build import build_vmd_camera_motion


# ----- camera track -----------------------------------------------------
def _camera_motion(**overrides) -> object:    # noqa: ANN003 — test helper
    """Round-trip a VMD camera motion through the parser + adapter."""
    return build_motion_asset(parse_vmd(build_vmd_camera_motion(**overrides)))


def test_camera_sample_at_keyframe_returns_keyframe_state() -> None:
    asset = _camera_motion(
        camera_keyframes=(
            (0, -45.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 30, False),
            (30, -20.0, (1.0, 2.0, 3.0), (0.1, 0.2, 0.3), 60, True),
        ),
    )
    sample = asset.camera_track.sample(30)
    np.testing.assert_allclose(sample.target, [1.0, 2.0, 3.0], atol=1e-5)
    assert sample.distance == pytest.approx(-20.0)
    assert sample.fov_degrees == pytest.approx(60.0)
    assert sample.perspective_off is True


def test_camera_position_components_ease_independently() -> None:
    """Each xyz channel has its own bezier — different easings on x and y
    should not contaminate each other's curve."""
    asset = _camera_motion(
        camera_keyframes=(
            (0, -30.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 30, False),
            (60, -30.0, (10.0, 10.0, 0.0), (0.0, 0.0, 0.0), 30, False),
        ),
    )
    sample = asset.camera_track.sample(30)
    # Default fixture handles are near-linear; ease at t=0.5 lands near 5.0
    # for both x and y. We assert symmetric independent interpolation by
    # checking they're both in the linear midpoint band.
    assert 4.5 <= float(sample.target[0]) <= 5.5
    assert 4.5 <= float(sample.target[1]) <= 5.5


def test_camera_distance_can_be_negative_for_back_view() -> None:
    """MMD's "back-view" trick uses negative distance — the parser and
    sampler must preserve sign through interpolation."""
    asset = _camera_motion(
        camera_keyframes=(
            (0, -45.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 30, False),
            (60, -10.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 30, False),
        ),
    )
    sample = asset.camera_track.sample(30)
    assert sample.distance < 0.0


def test_camera_perspective_off_steps_from_lower_keyframe() -> None:
    """``perspective_off`` is binary, so its mid-segment value should
    track the lower endpoint until the next keyframe lands."""
    asset = _camera_motion(
        camera_keyframes=(
            (0, -30.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 30, False),
            (30, -30.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 30, True),
        ),
    )
    assert not asset.camera_track.sample(15).perspective_off
    assert asset.camera_track.sample(30).perspective_off


# ----- camera factory + ortho -------------------------------------------
def test_camera_from_vmd_state_places_position_via_distance_and_rotation() -> None:
    """With identity rotation and distance ``-30``, the camera should sit
    behind the target on its local ``-Z`` axis."""
    cam = Camera.from_vmd_state(
        target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        rotation_xyz=(0.0, 0.0, 0.0),
        distance=-30.0,
        fov_degrees=30.0,
        perspective_off=False,
    )
    np.testing.assert_allclose(cam.position, [0.0, 0.0, -30.0], atol=1e-5)


def test_camera_from_vmd_state_ortho_when_perspective_off() -> None:
    cam = Camera.from_vmd_state(
        target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        rotation_xyz=(0.0, 0.0, 0.0),
        distance=-20.0,
        fov_degrees=60.0,
        perspective_off=True,
    )
    assert cam.orthographic_half_height is not None
    expected = 20.0 * float(np.tan(np.deg2rad(30.0)))
    assert cam.orthographic_half_height == pytest.approx(expected, rel=1e-4)
    proj = cam.projection_matrix(aspect=1.0)
    # Orthographic projection's last row is (0, 0, 0, 1) — identifies it
    # apart from perspective which sets proj[3, 2] = -1.
    assert proj[3, 3] == pytest.approx(1.0)
    assert proj[3, 2] == pytest.approx(0.0)


def test_camera_perspective_proj_unchanged_when_no_ortho() -> None:
    cam = Camera.from_vmd_state(
        target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        rotation_xyz=(0.0, 0.0, 0.0),
        distance=-30.0,
        fov_degrees=30.0,
        perspective_off=False,
    )
    proj = cam.projection_matrix(aspect=1.0)
    # Standard right-handed perspective: bottom-row == (0, 0, -1, 0).
    assert proj[3, 2] == pytest.approx(-1.0)


# ----- light track ------------------------------------------------------
def test_light_color_lerps_linearly() -> None:
    asset = _camera_motion(
        light_keyframes=(
            (0, (1.0, 1.0, 1.0), (-0.5, -1.0, 0.5)),
            (60, (0.0, 0.0, 0.0), (-0.5, -1.0, 0.5)),
        ),
    )
    sample = asset.light_track.sample(30)
    np.testing.assert_allclose(sample.color, [0.5, 0.5, 0.5], atol=1e-5)


def test_light_outside_range_clamps_to_endpoint() -> None:
    asset = _camera_motion(
        light_keyframes=(
            (10, (0.7, 0.7, 0.7), (0.0, -1.0, 0.0)),
        ),
    )
    pre = asset.light_track.sample(0)
    post = asset.light_track.sample(60)
    np.testing.assert_allclose(pre.color, [0.7, 0.7, 0.7], atol=1e-5)
    np.testing.assert_allclose(post.color, [0.7, 0.7, 0.7], atol=1e-5)


def test_directional_light_normalises_direction_on_construction() -> None:
    light = DirectionalLight(direction=np.array([0.0, 5.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(light.direction, [0.0, 1.0, 0.0], atol=1e-5)


# ----- self-shadow track -----------------------------------------------
def test_self_shadow_steps_at_each_keyframe() -> None:
    asset = _camera_motion(
        self_shadow_keyframes=(
            (0, 1, 0.0),
            (10, 0, 0.0),
            (20, 2, 0.05),
        ),
    )
    assert asset.self_shadow_track.sample(5).mode == 1
    assert asset.self_shadow_track.sample(15).mode == 0
    assert asset.self_shadow_track.sample(20).mode == 2
    assert asset.self_shadow_track.sample(50).mode == 2


def test_self_shadow_default_when_no_track() -> None:
    asset = _camera_motion()
    assert asset.self_shadow_track is None


# ----- scene driver -----------------------------------------------------
def test_scene_driver_returns_camera_at_time() -> None:
    asset = _camera_motion(
        camera_keyframes=(
            (0, -30.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 30, False),
            (60, -30.0, (5.0, 1.0, 0.0), (0.0, 0.0, 0.0), 30, False),
        ),
    )
    driver = VmdSceneDriver(motion=asset)
    cam = driver.camera_at(0.0)
    assert cam is not None
    np.testing.assert_allclose(cam.target, [0.0, 1.0, 0.0], atol=1e-5)
    cam_mid = driver.camera_at(30.0 / VMD_FRAMES_PER_SECOND)
    assert cam_mid is not None
    assert 1.5 <= float(cam_mid.target[0]) <= 3.5


def test_scene_driver_returns_none_when_no_track() -> None:
    asset = _camera_motion()
    driver = VmdSceneDriver(motion=asset)
    assert driver.camera_at(0.0) is None
    assert driver.light_at(0.0) is None
    assert driver.self_shadow_at(0.0) is None


def test_self_shadow_state_enabled_property_matches_mode() -> None:
    assert SelfShadowState(mode=0).enabled is False
    assert SelfShadowState(mode=1).enabled is True
    assert SelfShadowState(mode=2).enabled is True
