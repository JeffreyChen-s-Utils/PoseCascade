"""Tests for :mod:`posecascade.scene.builders`."""
from __future__ import annotations

import numpy as np

from posecascade.scene.builders import make_box_room


def test_make_box_room_topology() -> None:
    mesh = make_box_room(size=(2.0, 1.0, 4.0))
    # 6 faces x 4 corners = 24 unique vertices, 6 x 6 = 36 indices.
    assert mesh.positions.shape == (24, 3)
    assert mesh.indices.shape == (36,)
    assert mesh.normals is not None and mesh.normals.shape == (24, 3)


def test_make_box_room_dimensions() -> None:
    width, height, depth = 6.0, 3.0, 8.0
    mesh = make_box_room(size=(width, height, depth))
    p = mesh.positions
    assert float(p[:, 0].min()) == -width / 2
    assert float(p[:, 0].max()) == width / 2
    assert float(p[:, 1].min()) == 0.0
    assert float(p[:, 1].max()) == height
    assert float(p[:, 2].min()) == -depth / 2
    assert float(p[:, 2].max()) == depth / 2


def test_make_box_room_face_normals_unit_length() -> None:
    mesh = make_box_room()
    norms = np.linalg.norm(mesh.normals, axis=-1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1.0e-6)


def test_make_box_room_indices_in_range() -> None:
    mesh = make_box_room()
    assert int(mesh.indices.min()) == 0
    assert int(mesh.indices.max()) == 23


def test_make_box_room_name() -> None:
    mesh = make_box_room(name="custom_room")
    assert mesh.name == "custom_room"
