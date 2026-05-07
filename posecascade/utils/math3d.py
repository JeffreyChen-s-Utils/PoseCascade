"""Lightweight 3D math helpers.

Vectors and quaternions are plain numpy arrays of dtype float32. Matrices are
column-major, 4x4, dtype float32 — matching OpenGL's expectation when uploaded
with ``glUniformMatrix4fv(transpose=GL_FALSE)``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vec3 = NDArray[np.float32]
Quat = NDArray[np.float32]
Mat4 = NDArray[np.float32]

_EPSILON = 1.0e-6


def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vec3:
    """Construct a 3-vector."""
    return np.array([x, y, z], dtype=np.float32)


def quat_identity() -> Quat:
    """Return the identity quaternion ``[x=0, y=0, z=0, w=1]``."""
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def mat4_identity() -> Mat4:
    """Return a 4x4 identity matrix (column-major)."""
    return np.eye(4, dtype=np.float32)


def normalize(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return ``vector`` divided by its length; identity when length is below epsilon."""
    length = float(np.linalg.norm(vector))
    if length < _EPSILON:
        return vector.copy()
    return (vector / length).astype(np.float32)


def compose_trs(translation: Vec3, rotation: Quat, scale: Vec3) -> Mat4:
    """Compose a 4x4 transform from translation, unit quaternion rotation, and scale."""
    qx, qy, qz, qw = (float(v) for v in rotation)
    sx, sy, sz = (float(v) for v in scale)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    tx, ty, tz = float(translation[0]), float(translation[1]), float(translation[2])
    return np.array(
        [
            [(1 - 2 * (yy + zz)) * sx, 2 * (xy - wz) * sy, 2 * (xz + wy) * sz, tx],
            [2 * (xy + wz) * sx, (1 - 2 * (xx + zz)) * sy, 2 * (yz - wx) * sz, ty],
            [2 * (xz - wy) * sx, 2 * (yz + wx) * sy, (1 - 2 * (xx + yy)) * sz, tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def decompose_trs(matrix: Mat4) -> tuple[Vec3, Quat, Vec3]:
    """Decompose a 4x4 transform into ``(translation, rotation, scale)``.

    Assumes a non-skewed affine transform; skew is silently absorbed into rotation.
    """
    translation = matrix[:3, 3].astype(np.float32).copy()
    sx = float(np.linalg.norm(matrix[:3, 0]))
    sy = float(np.linalg.norm(matrix[:3, 1]))
    sz = float(np.linalg.norm(matrix[:3, 2]))
    scale = np.array([sx, sy, sz], dtype=np.float32)
    rotation_basis = np.column_stack(
        [
            matrix[:3, 0] / max(sx, _EPSILON),
            matrix[:3, 1] / max(sy, _EPSILON),
            matrix[:3, 2] / max(sz, _EPSILON),
        ]
    )
    rotation = _matrix_to_quat(rotation_basis.astype(np.float32))
    return translation, rotation, scale


def _matrix_to_quat(rot: NDArray[np.float32]) -> Quat:
    """Convert a 3x3 rotation matrix to a unit quaternion (Shepperd's method)."""
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        s = 0.5 / float(np.sqrt(trace + 1.0))
        return np.array(
            [
                (rot[2, 1] - rot[1, 2]) * s,
                (rot[0, 2] - rot[2, 0]) * s,
                (rot[1, 0] - rot[0, 1]) * s,
                0.25 / s,
            ],
            dtype=np.float32,
        )
    return _matrix_to_quat_off_diagonal(rot)


def quat_mul(a: Quat, b: Quat) -> Quat:
    """Hamilton product ``a ⊗ b`` — apply ``b`` first, then ``a``.

    Both inputs are unit quaternions in ``[x, y, z, w]`` order; result is the
    same. Used to compose joint rotations during chain integration.
    """
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def quat_conjugate(q: Quat) -> Quat:
    """Return the conjugate ``[-x, -y, -z, w]`` — the inverse of a unit quaternion."""
    return np.array([-float(q[0]), -float(q[1]), -float(q[2]), float(q[3])], dtype=np.float32)


def quat_inverse(q: Quat) -> Quat:
    """Return ``q⁻¹``. For unit quaternions this is the conjugate; we normalise to be safe."""
    norm_sq = float(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    if norm_sq < _EPSILON:
        return quat_identity()
    return quat_conjugate(q) / norm_sq


def quat_from_axis_angle(axis: Vec3, angle_radians: float) -> Quat:
    """Build a unit quaternion from a rotation axis (3-vec) and angle in radians."""
    unit_axis = normalize(np.asarray(axis, dtype=np.float32))
    half = float(angle_radians) * 0.5
    sin_half = float(np.sin(half))
    return np.array(
        [
            float(unit_axis[0]) * sin_half,
            float(unit_axis[1]) * sin_half,
            float(unit_axis[2]) * sin_half,
            float(np.cos(half)),
        ],
        dtype=np.float32,
    )


def quat_to_axis_angle(q: Quat) -> tuple[Vec3, float]:
    """Decompose a unit quaternion into ``(axis, angle_radians)`` with ``angle ∈ [0, π]``.

    Returns the canonical short-arc form: if ``w < 0`` the quaternion is flipped first
    so the angle stays in ``[0, π]`` rather than wrapping past ``π`` to ``2π - θ``.
    """
    qx, qy, qz, qw = (float(v) for v in q)
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    sin_half_sq = qx * qx + qy * qy + qz * qz
    if sin_half_sq < _EPSILON:
        return vec3(1.0, 0.0, 0.0), 0.0
    sin_half = float(np.sqrt(sin_half_sq))
    angle = 2.0 * float(np.arctan2(sin_half, qw))
    inv = 1.0 / sin_half
    return vec3(qx * inv, qy * inv, qz * inv), angle


def quat_exp_map(omega_dt: Vec3) -> Quat:
    """Convert a rotation vector ``ω·dt`` (axis × angle) into a unit quaternion.

    Input is the angular displacement over a step. Used to integrate angular
    velocity into rotation: ``q_next = quat_exp_map(ω · dt) ⊗ q_current``.
    """
    angle = float(np.linalg.norm(omega_dt))
    if angle < _EPSILON:
        return quat_identity()
    axis = np.asarray(omega_dt, dtype=np.float32) / angle
    return quat_from_axis_angle(axis, angle)


def quat_rotate_vec(q: Quat, v: Vec3) -> Vec3:
    """Apply a unit quaternion rotation to a 3-vector (``v' = q · v · q⁻¹``)."""
    qx, qy, qz, qw = (float(c) for c in q)
    vx, vy, vz = (float(c) for c in v)
    # u = vector part, s = scalar part. v' = 2·dot(u,v)·u + (s²−dot(u,u))·v + 2·s·cross(u,v)
    dot_uv = qx * vx + qy * vy + qz * vz
    s_sq_minus_dot_uu = qw * qw - (qx * qx + qy * qy + qz * qz)
    cross_x = qy * vz - qz * vy
    cross_y = qz * vx - qx * vz
    cross_z = qx * vy - qy * vx
    return np.array(
        [
            2.0 * dot_uv * qx + s_sq_minus_dot_uu * vx + 2.0 * qw * cross_x,
            2.0 * dot_uv * qy + s_sq_minus_dot_uu * vy + 2.0 * qw * cross_y,
            2.0 * dot_uv * qz + s_sq_minus_dot_uu * vz + 2.0 * qw * cross_z,
        ],
        dtype=np.float32,
    )


def quat_to_euler_xyz(q: Quat) -> tuple[float, float, float]:
    """Decompose a unit quaternion into ``(rx, ry, rz)`` XYZ-order Euler radians.

    The convention is intrinsic XYZ — matches MMD's IK limit semantics where
    ``limit_min`` / ``limit_max`` are per-axis radian ranges in this same order.
    The middle (``ry``) angle is clamped to ``±π/2``; near the poles ``rx``
    and ``rz`` are not unique so the returned triple may differ from the
    triple that produced ``q`` even though both represent the same rotation.
    """
    qx, qy, qz, qw = (float(v) for v in q)
    sin_y = 2.0 * (qx * qz + qw * qy)
    sin_y = max(-1.0, min(1.0, sin_y))
    ry = float(np.arcsin(sin_y))
    if abs(sin_y) > 1.0 - _EPSILON:
        # Gimbal lock: rz is degenerate; fold it into rx.
        rx = float(np.arctan2(qx * qy + qw * qz, 0.5 - (qy * qy + qz * qz)))
        rz = 0.0
    else:
        rx = float(np.arctan2(-2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)))
        rz = float(np.arctan2(-2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qy * qy + qz * qz)))
    return rx, ry, rz


def quat_from_euler_xyz(rx: float, ry: float, rz: float) -> Quat:
    """Build a unit quaternion from ``(rx, ry, rz)`` XYZ-order Euler radians.

    Inverse of :func:`quat_to_euler_xyz`. Composition order is
    ``Rx * Ry * Rz`` (i.e., ``Rx`` is applied first to a vector, ``Rz`` last)
    so that a vector ``v`` is rotated as ``q ⋅ v ⋅ q⁻¹`` ≡ ``Rx(Ry(Rz(v)))``.
    """
    half_x, half_y, half_z = rx * 0.5, ry * 0.5, rz * 0.5
    cx, cy, cz = float(np.cos(half_x)), float(np.cos(half_y)), float(np.cos(half_z))
    sx, sy, sz = float(np.sin(half_x)), float(np.sin(half_y)), float(np.sin(half_z))
    return np.array(
        [
            sx * cy * cz + cx * sy * sz,
            cx * sy * cz - sx * cy * sz,
            cx * cy * sz + sx * sy * cz,
            cx * cy * cz - sx * sy * sz,
        ],
        dtype=np.float32,
    )


def quat_slerp(a: Quat, b: Quat, t: float) -> Quat:
    """Spherical linear interpolation between two unit quaternions.

    Picks the shorter arc by flipping ``b`` when ``a · b < 0`` and falls
    back to a normalised lerp when the two quaternions are nearly equal
    (slerp's ``sin(angle)`` denominator collapses there).
    """
    qax, qay, qaz, qaw = (float(v) for v in a)
    qbx, qby, qbz, qbw = (float(v) for v in b)
    dot = qax * qbx + qay * qby + qaz * qbz + qaw * qbw
    if dot < 0.0:
        qbx, qby, qbz, qbw = -qbx, -qby, -qbz, -qbw
        dot = -dot
    if dot > 1.0 - _EPSILON:
        # Lerp + normalise — the quaternions are nearly identical, so the
        # short-arc slerp formula's sin(theta) approaches zero.
        return quat_normalize(
            np.array(
                [
                    qax + (qbx - qax) * t,
                    qay + (qby - qay) * t,
                    qaz + (qbz - qaz) * t,
                    qaw + (qbw - qaw) * t,
                ],
                dtype=np.float32,
            )
        )
    theta = float(np.arccos(min(1.0, max(-1.0, dot))))
    sin_theta = float(np.sin(theta))
    factor_a = float(np.sin((1.0 - t) * theta)) / sin_theta
    factor_b = float(np.sin(t * theta)) / sin_theta
    return np.array(
        [
            qax * factor_a + qbx * factor_b,
            qay * factor_a + qby * factor_b,
            qaz * factor_a + qbz * factor_b,
            qaw * factor_a + qbw * factor_b,
        ],
        dtype=np.float32,
    )


def quat_normalize(q: Quat) -> Quat:
    """Return ``q`` re-scaled to unit length; identity if degenerate."""
    norm = float(np.linalg.norm(q))
    if norm < _EPSILON:
        return quat_identity()
    return (q / norm).astype(np.float32)


def quat_to_euler(q: Quat) -> tuple[float, float, float]:
    """Decompose a unit quaternion into ``(yaw, pitch, roll)`` Tait-Bryan ZYX radians.

    Inverse of :func:`posecascade.scripting.api.quat_from_euler`. Pitch is clamped
    to ``±π/2`` to avoid the gimbal-lock branch's NaN; near the poles, yaw and
    roll are not unique so the returned triple may differ from what produced
    ``q`` even though it represents the same rotation. Use this for UI display
    or debug — never as the canonical rotation storage.
    """
    qx, qy, qz, qw = (float(v) for v in q)
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    pitch = float(np.arcsin(sin_pitch))
    if abs(sin_pitch) > 1.0 - _EPSILON:
        # Gimbal lock: roll is degenerate; fold it into yaw.
        yaw = float(np.arctan2(-2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz)))
        roll = 0.0
    else:
        roll = float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
        yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    return yaw, pitch, roll


def _matrix_to_quat_off_diagonal(rot: NDArray[np.float32]) -> Quat:
    """Off-diagonal branches of Shepperd's quaternion-from-matrix algorithm."""
    if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]))
        return np.array(
            [
                0.25 * s,
                (rot[0, 1] + rot[1, 0]) / s,
                (rot[0, 2] + rot[2, 0]) / s,
                (rot[2, 1] - rot[1, 2]) / s,
            ],
            dtype=np.float32,
        )
    if rot[1, 1] > rot[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]))
        return np.array(
            [
                (rot[0, 1] + rot[1, 0]) / s,
                0.25 * s,
                (rot[1, 2] + rot[2, 1]) / s,
                (rot[0, 2] - rot[2, 0]) / s,
            ],
            dtype=np.float32,
        )
    s = 2.0 * float(np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]))
    return np.array(
        [
            (rot[0, 2] + rot[2, 0]) / s,
            (rot[1, 2] + rot[2, 1]) / s,
            0.25 * s,
            (rot[1, 0] - rot[0, 1]) / s,
        ],
        dtype=np.float32,
    )
