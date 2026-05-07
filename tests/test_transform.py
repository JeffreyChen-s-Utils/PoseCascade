"""Tests for :class:`posecascade.scene.transform.Transform` and TRS math."""
from __future__ import annotations

import numpy as np

from posecascade.scene.transform import Transform
from posecascade.utils.math3d import (
    compose_trs,
    decompose_trs,
    quat_identity,
    vec3,
)


def test_identity_round_trip() -> None:
    translation = vec3(0.0, 0.0, 0.0)
    rotation = quat_identity()
    scale = vec3(1.0, 1.0, 1.0)
    matrix = compose_trs(translation, rotation, scale)
    np.testing.assert_allclose(matrix, np.eye(4, dtype=np.float32), atol=1.0e-6)


def test_compose_decompose_translation() -> None:
    t = vec3(1.0, 2.0, 3.0)
    r = quat_identity()
    s = vec3(1.0, 1.0, 1.0)
    decomposed_t, _, decomposed_s = decompose_trs(compose_trs(t, r, s))
    np.testing.assert_allclose(decomposed_t, t, atol=1.0e-6)
    np.testing.assert_allclose(decomposed_s, s, atol=1.0e-6)


def test_transform_to_dict_round_trip() -> None:
    original = Transform(
        translation=vec3(1.0, -2.0, 0.5),
        rotation=np.array([0.0, 0.7071, 0.0, 0.7071], dtype=np.float32),
        scale=vec3(2.0, 0.5, 1.0),
    )
    revived = Transform.from_dict(original.to_dict())
    np.testing.assert_allclose(revived.translation, original.translation, atol=1.0e-6)
    np.testing.assert_allclose(revived.rotation, original.rotation, atol=1.0e-6)
    np.testing.assert_allclose(revived.scale, original.scale, atol=1.0e-6)


def test_transform_setters_bump_version() -> None:
    transform = Transform()
    initial_version = transform.version
    transform.set_translation(vec3(1.0, 0.0, 0.0))
    assert transform.version == initial_version + 1
    transform.set_rotation(quat_identity())
    transform.set_scale(vec3(2.0, 2.0, 2.0))
    assert transform.version == initial_version + 3
