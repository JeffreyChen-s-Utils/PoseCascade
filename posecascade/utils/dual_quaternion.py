"""Dual-quaternion utilities for skinning.

Linear blend skinning (LBS) interpolates bone matrices component-wise.
At a twisted joint the blend collapses toward zero halfway between
two opposing rotations — the classic "candy-wrapper" artefact at
shoulders / elbows / wrists. Dual-quaternion skinning (DQS)
represents each bone transform as a screw motion (rotation + screw
translation) and blends those screw motions, which preserves volume
at the joint.

This module's :func:`matrices_to_dual_quaternions` converts the bone
matrix array the renderer already computes into the dual-quaternion
form the DQS vertex shader expects. The conversion runs once per
frame per skinned mesh — same cost class as the existing LBS matrix
upload, just with different per-bone arithmetic.

Layout for each bone (8 floats, all ``float32``):

    [qr.x, qr.y, qr.z, qr.w, qd.x, qd.y, qd.z, qd.w]

where ``qr`` is the unit rotation quaternion (xyzw convention) and
``qd`` is the dual part encoding translation in the screw-motion
form ``qd = 0.5 * t * qr`` (quaternion multiplication, with ``t``
treated as a pure quaternion ``(tx, ty, tz, 0)``).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_NUM_COMPONENTS_PER_QUAT = 4
_NUM_COMPONENTS_PER_DQ = 8
_MATRIX_NDIM = 3  # (N, 4, 4) has three array dimensions


def matrices_to_dual_quaternions(
    matrices: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Convert ``(N, 4, 4)`` rigid-transform matrices to ``(N, 8)`` dual quaternions.

    Assumes each input is a rigid motion (rotation + translation) — the
    rotation part may carry uniform scale and the helper still recovers
    a valid quaternion, but a non-uniform scale would silently produce
    a non-unit ``qr`` and skew the screw-motion translation. Bone
    matrices in PoseCascade are always rigid, so this assumption holds.
    """
    matrices = np.asarray(matrices, dtype=np.float32)
    if matrices.ndim != _MATRIX_NDIM or matrices.shape[1:] != (4, 4):
        raise ValueError(
            f"expected (N, 4, 4) matrices, got shape {matrices.shape}",
        )
    rotation = matrices[:, :3, :3]
    translation = matrices[:, :3, 3]
    qr = _rotation_matrix_to_quaternion(rotation)
    qd = _build_dual_part(translation, qr)
    return np.concatenate([qr, qd], axis=1).astype(np.float32, copy=False)


def _build_dual_part(
    translation: NDArray[np.float32],
    rotation_quat: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute ``qd = 0.5 * t * qr`` for every bone in one vectorized pass.

    Expands the quaternion multiplication ``(tx, ty, tz, 0) * (qx, qy, qz, qw)``
    using the standard Hamilton product so we don't take a per-bone
    Python loop.
    """
    tx = translation[:, 0]
    ty = translation[:, 1]
    tz = translation[:, 2]
    qx = rotation_quat[:, 0]
    qy = rotation_quat[:, 1]
    qz = rotation_quat[:, 2]
    qw = rotation_quat[:, 3]
    half = np.float32(0.5)
    dx = half * (tx * qw + ty * qz - tz * qy)
    dy = half * (-tx * qz + ty * qw + tz * qx)
    dz = half * (tx * qy - ty * qx + tz * qw)
    dw = half * (-tx * qx - ty * qy - tz * qz)
    return np.stack([dx, dy, dz, dw], axis=1).astype(np.float32, copy=False)


def _rotation_matrix_to_quaternion(
    rotation: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Vectorized Shepperd's method: ``(N, 3, 3)`` → ``(N, 4)`` xyzw quaternions.

    Picks the diagonal entry with the largest absolute value as the
    pivot so the divisor in each branch stays away from zero — that's
    what keeps the extraction numerically stable when the rotation is
    near a 180° twist on one of the principal axes.
    """
    n = rotation.shape[0]
    m00 = rotation[:, 0, 0]
    m11 = rotation[:, 1, 1]
    m22 = rotation[:, 2, 2]
    m01 = rotation[:, 0, 1]
    m02 = rotation[:, 0, 2]
    m10 = rotation[:, 1, 0]
    m12 = rotation[:, 1, 2]
    m20 = rotation[:, 2, 0]
    m21 = rotation[:, 2, 1]
    trace = m00 + m11 + m22

    out = np.zeros((n, _NUM_COMPONENTS_PER_QUAT), dtype=np.float32)
    qx, qy, qz, qw = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    case1 = trace > 0.0
    case2 = (~case1) & (m00 >= m11) & (m00 >= m22)
    case3 = (~case1) & (~case2) & (m11 >= m22)
    case4 = ~(case1 | case2 | case3)

    if case1.any():
        s = np.sqrt(trace[case1] + 1.0) * 2.0
        qw[case1] = 0.25 * s
        qx[case1] = (m21[case1] - m12[case1]) / s
        qy[case1] = (m02[case1] - m20[case1]) / s
        qz[case1] = (m10[case1] - m01[case1]) / s

    if case2.any():
        s = np.sqrt(1.0 + m00[case2] - m11[case2] - m22[case2]) * 2.0
        qw[case2] = (m21[case2] - m12[case2]) / s
        qx[case2] = 0.25 * s
        qy[case2] = (m01[case2] + m10[case2]) / s
        qz[case2] = (m02[case2] + m20[case2]) / s

    if case3.any():
        s = np.sqrt(1.0 + m11[case3] - m00[case3] - m22[case3]) * 2.0
        qw[case3] = (m02[case3] - m20[case3]) / s
        qx[case3] = (m01[case3] + m10[case3]) / s
        qy[case3] = 0.25 * s
        qz[case3] = (m12[case3] + m21[case3]) / s

    if case4.any():
        s = np.sqrt(1.0 + m22[case4] - m00[case4] - m11[case4]) * 2.0
        qw[case4] = (m10[case4] - m01[case4]) / s
        qx[case4] = (m02[case4] + m20[case4]) / s
        qy[case4] = (m12[case4] + m21[case4]) / s
        qz[case4] = 0.25 * s

    return out


def dq_transform_point(
    dq: NDArray[np.float32], point: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Apply a single dual quaternion to a 3D point. Useful for tests.

    Implements the screw-motion form ``p' = R(p) + t`` where ``R`` is
    the rotation encoded by ``qr`` and ``t`` is derived from
    ``2 * (qr.w * qd.xyz - qd.w * qr.xyz + cross(qr.xyz, qd.xyz))``.
    The shader does the same operation per vertex.
    """
    dq = np.asarray(dq, dtype=np.float32)
    point = np.asarray(point, dtype=np.float32)
    if dq.shape != (_NUM_COMPONENTS_PER_DQ,):
        raise ValueError(f"expected (8,) dual quaternion, got shape {dq.shape}")
    if point.shape != (3,):
        raise ValueError(f"expected (3,) point, got shape {point.shape}")
    qr = dq[:_NUM_COMPONENTS_PER_QUAT]
    qd = dq[_NUM_COMPONENTS_PER_QUAT:]
    qr_xyz = qr[:3]
    qr_w = qr[3]
    qd_xyz = qd[:3]
    qd_w = qd[3]
    rotated = point + 2.0 * np.cross(
        qr_xyz, np.cross(qr_xyz, point) + qr_w * point,
    )
    translation = 2.0 * (
        qr_w * qd_xyz - qd_w * qr_xyz + np.cross(qr_xyz, qd_xyz)
    )
    return (rotated + translation).astype(np.float32, copy=False)
