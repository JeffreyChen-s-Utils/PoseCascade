"""Tests for :mod:`posecascade.scripting.api` helpers exposed to user scripts."""
from __future__ import annotations

import math

import numpy as np

from posecascade.scripting.api import (
    SCRIPT_API_HELPERS,
    clamp,
    lerp,
    quat_axis_angle,
    quat_from_euler,
)
from posecascade.scripting.sandbox import build_api
from posecascade.utils.math3d import vec3


def test_quat_axis_angle_identity_at_zero() -> None:
    q = quat_axis_angle(vec3(0.0, 1.0, 0.0), 0.0)
    np.testing.assert_allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1.0e-6)


def test_quat_axis_angle_unit_norm() -> None:
    for angle in (0.1, math.pi * 0.5, math.pi, math.tau, -1.7):
        q = quat_axis_angle(vec3(1.0, 2.0, 3.0), angle)  # axis is normalized internally
        norm = float(np.linalg.norm(q))
        assert abs(norm - 1.0) < 1.0e-5


def test_quat_axis_angle_y_pi_flips_xz() -> None:
    q = quat_axis_angle(vec3(0.0, 1.0, 0.0), math.pi)
    # y-axis half-rotation by π: q ≈ (0, 1, 0, 0)
    np.testing.assert_allclose(q, [0.0, 1.0, 0.0, 0.0], atol=1.0e-6)


def test_quat_from_euler_identity() -> None:
    q = quat_from_euler(0.0, 0.0, 0.0)
    np.testing.assert_allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1.0e-6)


def test_lerp_basic() -> None:
    assert lerp(0.0, 10.0, 0.0) == 0.0
    assert lerp(0.0, 10.0, 1.0) == 10.0
    assert lerp(0.0, 10.0, 0.5) == 5.0


def test_lerp_unclamped() -> None:
    assert lerp(0.0, 10.0, -1.0) == -10.0
    assert lerp(0.0, 10.0, 2.0) == 20.0


def test_clamp_inside_range() -> None:
    assert clamp(0.5, 0.0, 1.0) == 0.5


def test_clamp_below() -> None:
    assert clamp(-3.0, 0.0, 1.0) == 0.0


def test_clamp_above() -> None:
    assert clamp(5.0, 0.0, 1.0) == 1.0


def test_helpers_merged_into_build_api() -> None:
    api = build_api(scene=None, time_provider=lambda: 0.0)
    for key in SCRIPT_API_HELPERS:
        assert key in api
    assert api["pi"] == math.pi
    assert api["tau"] == math.tau
