"""Tests for quaternion operations in :mod:`posecascade.utils.math3d`."""
from __future__ import annotations

import math

import numpy as np

from posecascade.utils.math3d import (
    quat_conjugate,
    quat_exp_map,
    quat_from_axis_angle,
    quat_identity,
    quat_inverse,
    quat_mul,
    quat_normalize,
    quat_rotate_vec,
    quat_to_axis_angle,
    vec3,
)


def test_quat_mul_identity_left() -> None:
    q = quat_from_axis_angle(vec3(0.0, 1.0, 0.0), math.pi / 3)
    np.testing.assert_allclose(quat_mul(quat_identity(), q), q, atol=1.0e-6)


def test_quat_mul_identity_right() -> None:
    q = quat_from_axis_angle(vec3(1.0, 0.0, 0.0), math.pi / 4)
    np.testing.assert_allclose(quat_mul(q, quat_identity()), q, atol=1.0e-6)


def test_quat_mul_compose_axis_angle() -> None:
    # Two 45° rotations about same axis = one 90°
    q = quat_from_axis_angle(vec3(0.0, 0.0, 1.0), math.pi / 4)
    composed = quat_mul(q, q)
    expected = quat_from_axis_angle(vec3(0.0, 0.0, 1.0), math.pi / 2)
    np.testing.assert_allclose(composed, expected, atol=1.0e-6)


def test_quat_inverse_round_trip() -> None:
    q = quat_from_axis_angle(vec3(1.0, 2.0, 3.0), 1.234)
    result = quat_mul(q, quat_inverse(q))
    np.testing.assert_allclose(result, quat_identity(), atol=1.0e-6)


def test_quat_conjugate_negates_vector_part() -> None:
    q = quat_from_axis_angle(vec3(0.0, 1.0, 0.0), math.pi / 5)
    c = quat_conjugate(q)
    np.testing.assert_allclose([c[0], c[1], c[2]], [-q[0], -q[1], -q[2]], atol=1.0e-6)
    np.testing.assert_allclose(c[3], q[3], atol=1.0e-6)


def test_quat_to_axis_angle_round_trip() -> None:
    axis_in = vec3(0.3, 0.7, 0.5)
    axis_in = axis_in / np.linalg.norm(axis_in)
    angle_in = 0.9
    q = quat_from_axis_angle(axis_in, angle_in)
    axis_out, angle_out = quat_to_axis_angle(q)
    assert angle_out == math.pi or abs(angle_out - angle_in) < 1.0e-5
    np.testing.assert_allclose(axis_out, axis_in, atol=1.0e-5)


def test_quat_to_axis_angle_identity() -> None:
    axis, angle = quat_to_axis_angle(quat_identity())
    assert angle == 0.0
    # axis is arbitrary for identity; just check it is a finite unit vector
    assert math.isclose(float(np.linalg.norm(axis)), 1.0, abs_tol=1.0e-6)


def test_quat_to_axis_angle_short_arc() -> None:
    # Quaternion with negative w should be flipped to short-arc form.
    q = quat_from_axis_angle(vec3(0.0, 1.0, 0.0), math.pi - 0.1)
    flipped = -q  # equivalent rotation, but w now negative if w>0 originally
    axis_a, angle_a = quat_to_axis_angle(q)
    axis_b, angle_b = quat_to_axis_angle(flipped)
    assert math.isclose(angle_a, angle_b, abs_tol=1.0e-6)
    np.testing.assert_allclose(axis_a, axis_b, atol=1.0e-6)


def test_quat_exp_map_zero_returns_identity() -> None:
    np.testing.assert_allclose(quat_exp_map(vec3(0.0, 0.0, 0.0)), quat_identity(), atol=1.0e-6)


def test_quat_exp_map_matches_axis_angle() -> None:
    # Angular displacement of ω·dt should produce same rotation as axis-angle.
    omega_dt = vec3(0.0, 0.0, math.pi / 6)
    q = quat_exp_map(omega_dt)
    expected = quat_from_axis_angle(vec3(0.0, 0.0, 1.0), math.pi / 6)
    np.testing.assert_allclose(q, expected, atol=1.0e-6)


def test_quat_rotate_vec_identity() -> None:
    v = vec3(1.0, 2.0, 3.0)
    np.testing.assert_allclose(quat_rotate_vec(quat_identity(), v), v, atol=1.0e-6)


def test_quat_rotate_vec_90deg_z() -> None:
    # Rotating +X by 90° around Z should produce +Y.
    q = quat_from_axis_angle(vec3(0.0, 0.0, 1.0), math.pi / 2)
    rotated = quat_rotate_vec(q, vec3(1.0, 0.0, 0.0))
    np.testing.assert_allclose(rotated, vec3(0.0, 1.0, 0.0), atol=1.0e-6)


def test_quat_rotate_vec_compose() -> None:
    # Rotate twice by q vs once by quat_mul(q, q).
    q = quat_from_axis_angle(vec3(0.0, 1.0, 0.0), 0.4)
    v = vec3(1.0, 0.0, 0.0)
    twice = quat_rotate_vec(q, quat_rotate_vec(q, v))
    once = quat_rotate_vec(quat_mul(q, q), v)
    np.testing.assert_allclose(twice, once, atol=1.0e-5)


def test_quat_normalize() -> None:
    q = np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32)  # not unit
    n = quat_normalize(q)
    assert math.isclose(float(np.linalg.norm(n)), 1.0, abs_tol=1.0e-6)


def test_quat_normalize_zero() -> None:
    q = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    n = quat_normalize(q)
    np.testing.assert_allclose(n, quat_identity(), atol=1.0e-6)
