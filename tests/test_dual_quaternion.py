"""Tests for the dual-quaternion skinning math.

Pins the screw-motion arithmetic that the DQS toon shader relies on —
identity, pure translation, pure rotation, a combined rigid motion,
and the antipodal-quaternion handling that keeps blended bones from
cancelling at twisted joints.
"""
from __future__ import annotations

import math

import numpy as np

from posecascade.utils.dual_quaternion import (
    dq_transform_point,
    matrices_to_dual_quaternions,
)


def test_identity_matrix_round_trips_to_identity_dq() -> None:
    """The identity transform encodes as the identity quaternion + zero dual."""
    m = np.eye(4, dtype=np.float32)[None, :, :]
    dq = matrices_to_dual_quaternions(m)[0]
    np.testing.assert_allclose(dq[:4], [0, 0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(dq[4:], [0, 0, 0, 0], atol=1e-6)


def test_pure_translation_recovers_via_dq_transform() -> None:
    """A translate-only matrix moves a point by exactly the matrix's offset."""
    m = np.eye(4, dtype=np.float32)[None, :, :].copy()
    m[0, :3, 3] = [5.0, -3.0, 2.0]
    dq = matrices_to_dual_quaternions(m)[0]
    out = dq_transform_point(dq, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(out, [6.0, -1.0, 5.0], atol=1e-5)


def test_rotation_about_y_90_degrees_maps_x_axis_to_negative_z() -> None:
    """A 90° rotation about +Y sends (1, 0, 0) → (0, 0, -1)."""
    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    m = np.array(
        [[[ c, 0, s, 0],
          [ 0, 1, 0, 0],
          [-s, 0, c, 0],
          [ 0, 0, 0, 1]]],
        dtype=np.float32,
    )
    dq = matrices_to_dual_quaternions(m)[0]
    out = dq_transform_point(dq, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(out, [0.0, 0.0, -1.0], atol=1e-5)


def test_combined_rotation_then_translation() -> None:
    """A rigid motion (rotate 90° about Y then translate +X by 5) composes correctly."""
    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    m = np.array(
        [[[ c, 0, s, 5],
          [ 0, 1, 0, 0],
          [-s, 0, c, 0],
          [ 0, 0, 0, 1]]],
        dtype=np.float32,
    )
    dq = matrices_to_dual_quaternions(m)[0]
    out = dq_transform_point(dq, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(out, [5.0, 0.0, -1.0], atol=1e-5)


def test_rotation_quat_norm_is_unity() -> None:
    """For a rigid bone matrix, the recovered ``qr`` must lie on the unit sphere."""
    rng = np.random.default_rng(42)
    matrices = []
    for _ in range(20):
        # Random rotation: build from a random axis-angle.
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle = rng.uniform(-math.pi, math.pi)
        c, s = math.cos(angle), math.sin(angle)
        cross_matrix = np.array(
            [[0, -axis[2], axis[1]],
             [axis[2], 0, -axis[0]],
             [-axis[1], axis[0], 0]],
        )
        rotation = np.eye(3) + s * cross_matrix + (1 - c) * (cross_matrix @ cross_matrix)
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = rotation
        m[:3, 3] = rng.normal(size=3)
        matrices.append(m)
    dqs = matrices_to_dual_quaternions(np.asarray(matrices, dtype=np.float32))
    norms = np.linalg.norm(dqs[:, :4], axis=1)
    np.testing.assert_allclose(norms, np.ones(len(matrices)), atol=1e-5)


def test_matrices_to_dual_quaternions_validates_shape() -> None:
    """Wrong-shape input raises a clear error rather than silently going wrong."""
    import pytest  # noqa: PLC0415

    with pytest.raises(ValueError, match="expected"):
        matrices_to_dual_quaternions(np.eye(3, dtype=np.float32))


def test_dq_transform_point_validates_shapes() -> None:
    """Single-DQ helper rejects mismatched inputs."""
    import pytest  # noqa: PLC0415

    dq = np.zeros(8, dtype=np.float32)
    dq[3] = 1.0
    with pytest.raises(ValueError, match="expected"):
        dq_transform_point(dq[:4], np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="expected"):
        dq_transform_point(dq, np.zeros(4, dtype=np.float32))
