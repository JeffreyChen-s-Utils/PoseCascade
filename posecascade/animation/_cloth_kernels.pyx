# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython kernels for the cloth solver hot path.

Re-implements :func:`_solve_distance_constraints` and the two static-collider
projections (sphere, capsule) as typed Cython loops. The Python wrapper in
``cloth.py`` imports these unconditionally when the compiled extension is
available, and falls back to the NumPy implementations when it isn't (so
running from a source checkout without a build step still works).

Conventions:
- ``float_t`` is a fused type covering ``float32`` and ``float64``. Cython
  generates one specialisation per dtype and dispatches at call time based
  on the ``positions`` array's actual dtype. Every float argument in a
  given call must resolve to the same concrete type — the wrapper enforces
  this by reusing ``piece.positions.dtype`` for every per-piece array.
- Positions / prev_positions are mutated in place; the caller owns them.
- The signed-scale convention from the NumPy version is preserved: the
  first half of ``combined_scale`` is the (negative) a-side weighting, the
  second is the (positive) b-side. The kernel simply does
  ``positions[combined_idx[i]] += correction * combined_scale[i]`` so the
  sign collapses naturally into the scatter.
- The collider projection kernels take ``start`` / ``end`` index bounds so
  sub-AABB binning can dispatch each collider's projection to a subset of
  vertices rather than scanning the full mesh.
"""

import numpy as np

cimport cython
cimport numpy as cnp

from libc.math cimport sqrt


ctypedef fused float_t:
    cnp.float32_t
    cnp.float64_t


@cython.boundscheck(False)
@cython.wraparound(False)
def solve_distance_constraints(
    float_t[:, ::1] positions,
    cnp.intp_t[::1] a_idx,
    cnp.intp_t[::1] b_idx,
    float_t[::1] rest_lengths,
    cnp.intp_t[::1] combined_idx,
    float_t[::1] combined_scale,
    float_t stiffness,
    float_t numeric_eps,
) -> None:
    """One Jacobi-style PBD distance pass on a contiguous position array.

    Implements the same maths as the NumPy fallback: per-edge ``delta``
    direction + ``stiffness``-scaled length error, then scatter the per-
    axis correction onto the affected vertices through ``combined_idx`` /
    ``combined_scale``. Each constraint contributes exactly two scatters
    (a-side and b-side) — the signed scale folds the ``-=``/``+=`` split
    of the NumPy version into a single ``+=``.
    """
    cdef Py_ssize_t m = a_idx.shape[0]
    if m == 0:
        return
    cdef Py_ssize_t i, ia, ib
    cdef float_t ax, ay, az, bx, by, bz
    cdef float_t dx, dy, dz, length, safe_length, err, scale, cx, cy, cz
    for i in range(m):
        ia = a_idx[i]
        ib = b_idx[i]
        ax = positions[ia, 0]; ay = positions[ia, 1]; az = positions[ia, 2]
        bx = positions[ib, 0]; by = positions[ib, 1]; bz = positions[ib, 2]
        dx = ax - bx; dy = ay - by; dz = az - bz
        length = <float_t>sqrt(dx * dx + dy * dy + dz * dz)
        safe_length = length if length > numeric_eps else numeric_eps
        err = (length - rest_lengths[i]) * stiffness / safe_length
        cx = dx * err; cy = dy * err; cz = dz * err
        scale = combined_scale[i]
        positions[ia, 0] += cx * scale
        positions[ia, 1] += cy * scale
        positions[ia, 2] += cz * scale
        scale = combined_scale[m + i]
        positions[ib, 0] += cx * scale
        positions[ib, 1] += cy * scale
        positions[ib, 2] += cz * scale


@cython.boundscheck(False)
@cython.wraparound(False)
def project_static_sphere(
    float_t[:, ::1] positions,
    float_t[:, ::1] prev_positions,
    float_t[::1] inverse_masses,
    float_t cx, float_t cy, float_t cz,
    float_t radius,
    float_t tangent_retention,
    float_t numeric_eps,
    Py_ssize_t start,
    Py_ssize_t end,
) -> None:
    """Push movable verts in ``[start, end)`` outside a static sphere.

    ``start`` / ``end`` let the sub-AABB binning dispatch each collider's
    projection only to vertices whose bin overlaps the collider — for a
    skirt mostly out of contact with a hand-sphere, this typically cuts
    the per-call vert count by 3-4× without changing the projection
    maths.
    """
    cdef Py_ssize_t i
    cdef float_t dx, dy, dz, dist, inv_dist, nx, ny, nz
    cdef float_t old_x, old_y, old_z, prev_x, prev_y, prev_z
    cdef float_t vx, vy, vz, normal_v, tx, ty, tz
    cdef float_t new_x, new_y, new_z
    for i in range(start, end):
        if inverse_masses[i] == 0.0:
            continue
        old_x = positions[i, 0]; old_y = positions[i, 1]; old_z = positions[i, 2]
        dx = old_x - cx; dy = old_y - cy; dz = old_z - cz
        dist = <float_t>sqrt(dx * dx + dy * dy + dz * dz)
        if dist >= radius:
            continue
        inv_dist = 1.0 / (dist if dist > numeric_eps else numeric_eps)
        nx = dx * inv_dist; ny = dy * inv_dist; nz = dz * inv_dist
        new_x = cx + nx * radius
        new_y = cy + ny * radius
        new_z = cz + nz * radius
        prev_x = prev_positions[i, 0]
        prev_y = prev_positions[i, 1]
        prev_z = prev_positions[i, 2]
        vx = old_x - prev_x; vy = old_y - prev_y; vz = old_z - prev_z
        normal_v = vx * nx + vy * ny + vz * nz
        tx = (vx - normal_v * nx) * tangent_retention
        ty = (vy - normal_v * ny) * tangent_retention
        tz = (vz - normal_v * nz) * tangent_retention
        positions[i, 0] = new_x
        positions[i, 1] = new_y
        positions[i, 2] = new_z
        prev_positions[i, 0] = new_x - tx
        prev_positions[i, 1] = new_y - ty
        prev_positions[i, 2] = new_z - tz


@cython.boundscheck(False)
@cython.wraparound(False)
def project_static_capsule(
    float_t[:, ::1] positions,
    float_t[:, ::1] prev_positions,
    float_t[::1] inverse_masses,
    float_t ax, float_t ay, float_t az,
    float_t bx, float_t by, float_t bz,
    float_t radius,
    float_t tangent_retention,
    float_t numeric_eps,
    Py_ssize_t start,
    Py_ssize_t end,
) -> None:
    """Push movable verts in ``[start, end)`` outside a static capsule."""
    cdef Py_ssize_t i
    cdef float_t sx, sy, sz, seg_sq, rx, ry, rz, t
    cdef float_t closest_x, closest_y, closest_z
    cdef float_t dx, dy, dz, dist, inv_dist, nx, ny, nz
    cdef float_t old_x, old_y, old_z, prev_x, prev_y, prev_z
    cdef float_t vx, vy, vz, normal_v, tx, ty, tz
    cdef float_t new_x, new_y, new_z
    sx = bx - ax; sy = by - ay; sz = bz - az
    seg_sq = sx * sx + sy * sy + sz * sz
    if seg_sq < numeric_eps:
        return
    for i in range(start, end):
        if inverse_masses[i] == 0.0:
            continue
        old_x = positions[i, 0]; old_y = positions[i, 1]; old_z = positions[i, 2]
        rx = old_x - ax; ry = old_y - ay; rz = old_z - az
        t = (rx * sx + ry * sy + rz * sz) / seg_sq
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        closest_x = ax + t * sx
        closest_y = ay + t * sy
        closest_z = az + t * sz
        dx = old_x - closest_x; dy = old_y - closest_y; dz = old_z - closest_z
        dist = <float_t>sqrt(dx * dx + dy * dy + dz * dz)
        if dist >= radius:
            continue
        inv_dist = 1.0 / (dist if dist > numeric_eps else numeric_eps)
        nx = dx * inv_dist; ny = dy * inv_dist; nz = dz * inv_dist
        new_x = closest_x + nx * radius
        new_y = closest_y + ny * radius
        new_z = closest_z + nz * radius
        prev_x = prev_positions[i, 0]
        prev_y = prev_positions[i, 1]
        prev_z = prev_positions[i, 2]
        vx = old_x - prev_x; vy = old_y - prev_y; vz = old_z - prev_z
        normal_v = vx * nx + vy * ny + vz * nz
        tx = (vx - normal_v * nx) * tangent_retention
        ty = (vy - normal_v * ny) * tangent_retention
        tz = (vz - normal_v * nz) * tangent_retention
        positions[i, 0] = new_x
        positions[i, 1] = new_y
        positions[i, 2] = new_z
        prev_positions[i, 0] = new_x - tx
        prev_positions[i, 1] = new_y - ty
        prev_positions[i, 2] = new_z - tz
