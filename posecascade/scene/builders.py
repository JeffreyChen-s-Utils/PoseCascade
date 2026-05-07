"""Procedural geometry helpers — used as fallbacks when an asset isn't available."""
from __future__ import annotations

import numpy as np

from posecascade.assets.types import Mesh

_ROOM_FACE_QUAD_INDICES: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),  # floor (look up)
    (4, 7, 6, 5),  # ceiling (look down)
    (0, 4, 5, 1),  # +Z wall (look towards -Z)
    (2, 6, 7, 3),  # -Z wall (look towards +Z)
    (3, 7, 4, 0),  # -X wall (look towards +X)
    (1, 5, 6, 2),  # +X wall (look towards -X)
)

_ROOM_FACE_NORMALS: tuple[tuple[float, float, float], ...] = (
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
)


def make_box_room(
    name: str = "room",
    size: tuple[float, float, float] = (10.0, 4.0, 10.0),
) -> Mesh:
    """Build a 6-quad box room with inward-facing normals.

    Origin sits in the centre of the floor. ``size = (width, height, depth)``
    in metres. Backface culling is left to the caller — the demo renderer
    keeps it disabled, so the room is visible from outside as well.
    """
    half_x, full_y, half_z = size[0] * 0.5, size[1], size[2] * 0.5
    corners = np.array(
        [
            [-half_x, 0.0,  half_z],
            [ half_x, 0.0,  half_z],
            [ half_x, 0.0, -half_z],
            [-half_x, 0.0, -half_z],
            [-half_x, full_y,  half_z],
            [ half_x, full_y,  half_z],
            [ half_x, full_y, -half_z],
            [-half_x, full_y, -half_z],
        ],
        dtype=np.float32,
    )
    positions = np.empty((24, 3), dtype=np.float32)
    normals = np.empty((24, 3), dtype=np.float32)
    indices = np.empty(36, dtype=np.uint32)
    for face_index, (quad, normal) in enumerate(
        zip(_ROOM_FACE_QUAD_INDICES, _ROOM_FACE_NORMALS, strict=True)
    ):
        base = face_index * 4
        for j, corner_index in enumerate(quad):
            positions[base + j] = corners[corner_index]
            normals[base + j] = normal
        idx_base = face_index * 6
        indices[idx_base:idx_base + 6] = (base, base + 1, base + 2, base, base + 2, base + 3)
    return Mesh(name=name, positions=positions, indices=indices, normals=normals)
