"""Tests for the bone post-IK resolver (append + fixed-axis)."""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pmx.importer import PmxImporter

from posecascade.animation.bone_resolver import (
    BoneAppendRule,
    BoneResolver,
    BoneResolverRules,
    FixedAxisRule,
    quat_project_to_axis,
)
from posecascade.scene.node import Node
from posecascade.scene.transform import Transform
from posecascade.utils.math3d import (
    quat_from_axis_angle,
    quat_from_euler_xyz,
    quat_identity,
    quat_to_euler_xyz,
    vec3,
)
from tests.fixtures.mmd.build import (
    FixtureBone,
    FixtureBuild,
    FixtureMaterial,
    FixtureVertex,
    _Bdef1,
    build_pmx,
)


def _two_bone_pair() -> tuple[Node, Node]:
    parent = Node(name="parent", transform=Transform(translation=vec3(0, 1, 0)))
    child = Node(name="child", transform=Transform(translation=vec3(1, 1, 0)))
    return parent, child


def _resolver_for(
    nodes: dict[int, Node],
    appends: tuple[BoneAppendRule, ...] = (),
    fixed_axes: tuple[FixedAxisRule, ...] = (),
) -> BoneResolver:
    rules = BoneResolverRules(
        deformation_order=tuple(sorted(nodes.keys())),
        appends=appends,
        fixed_axes=fixed_axes,
    )
    return BoneResolver.from_rules(rules, nodes)


# ----- append: rotation -------------------------------------------------
def test_append_rotation_at_full_weight_copies_parent_delta() -> None:
    parent, child = _two_bone_pair()
    nodes = {0: parent, 1: child}
    resolver = _resolver_for(
        nodes,
        appends=(
            BoneAppendRule(
                bone_index=1, parent_index=0, weight=1.0, inherit_rotation=True,
            ),
        ),
    )
    parent.transform.set_rotation(quat_from_euler_xyz(0.0, np.deg2rad(60), 0.0))
    resolver.resolve()
    rx, ry, rz = quat_to_euler_xyz(child.transform.rotation)
    assert np.rad2deg(ry) == pytest.approx(60.0, abs=0.5)


def test_append_rotation_half_weight_halves_parent_delta() -> None:
    parent, child = _two_bone_pair()
    nodes = {0: parent, 1: child}
    resolver = _resolver_for(
        nodes,
        appends=(
            BoneAppendRule(
                bone_index=1, parent_index=0, weight=0.5, inherit_rotation=True,
            ),
        ),
    )
    parent.transform.set_rotation(quat_from_euler_xyz(0.0, np.deg2rad(60), 0.0))
    resolver.resolve()
    rx, ry, rz = quat_to_euler_xyz(child.transform.rotation)
    assert np.rad2deg(ry) == pytest.approx(30.0, abs=0.5)


def test_append_rotation_zero_weight_is_noop() -> None:
    parent, child = _two_bone_pair()
    nodes = {0: parent, 1: child}
    resolver = _resolver_for(
        nodes,
        appends=(
            BoneAppendRule(
                bone_index=1, parent_index=0, weight=0.0, inherit_rotation=True,
            ),
        ),
    )
    parent.transform.set_rotation(quat_from_euler_xyz(0.0, np.deg2rad(60), 0.0))
    resolver.resolve()
    np.testing.assert_allclose(child.transform.rotation, quat_identity(), atol=1e-5)


def test_append_negative_weight_inverts_parent_delta() -> None:
    parent, child = _two_bone_pair()
    nodes = {0: parent, 1: child}
    resolver = _resolver_for(
        nodes,
        appends=(
            BoneAppendRule(
                bone_index=1, parent_index=0, weight=-1.0, inherit_rotation=True,
            ),
        ),
    )
    parent.transform.set_rotation(quat_from_euler_xyz(0.0, np.deg2rad(60), 0.0))
    resolver.resolve()
    rx, ry, rz = quat_to_euler_xyz(child.transform.rotation)
    assert np.rad2deg(ry) == pytest.approx(-60.0, abs=0.5)


# ----- append: translation ---------------------------------------------
def test_append_translation_only_does_not_touch_rotation() -> None:
    parent, child = _two_bone_pair()
    nodes = {0: parent, 1: child}
    resolver = _resolver_for(
        nodes,
        appends=(
            BoneAppendRule(
                bone_index=1, parent_index=0, weight=1.0, inherit_translation=True,
            ),
        ),
    )
    parent.transform.set_translation(vec3(0.5, 1.0, 0.0))
    parent.transform.set_rotation(quat_from_euler_xyz(0.0, np.deg2rad(60), 0.0))
    resolver.resolve()
    np.testing.assert_allclose(child.transform.translation, [1.5, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(child.transform.rotation, quat_identity(), atol=1e-5)


def test_append_both_rotation_and_translation() -> None:
    parent, child = _two_bone_pair()
    nodes = {0: parent, 1: child}
    resolver = _resolver_for(
        nodes,
        appends=(
            BoneAppendRule(
                bone_index=1, parent_index=0, weight=1.0,
                inherit_rotation=True, inherit_translation=True,
            ),
        ),
    )
    parent.transform.set_translation(vec3(0.0, 1.5, 0.0))
    parent.transform.set_rotation(quat_from_euler_xyz(np.deg2rad(20), 0.0, 0.0))
    resolver.resolve()
    np.testing.assert_allclose(child.transform.translation, [1.0, 1.5, 0.0], atol=1e-5)
    rx, _ry, _rz = quat_to_euler_xyz(child.transform.rotation)
    assert np.rad2deg(rx) == pytest.approx(20.0, abs=0.5)


# ----- append: missing parent / cycle ----------------------------------
def test_append_missing_parent_silently_skips() -> None:
    parent, child = _two_bone_pair()
    nodes = {0: parent, 1: child}
    resolver = _resolver_for(
        nodes,
        appends=(
            BoneAppendRule(
                bone_index=1, parent_index=99, weight=1.0, inherit_rotation=True,
            ),
        ),
    )
    parent.transform.set_rotation(quat_from_euler_xyz(0.0, np.deg2rad(60), 0.0))
    resolver.resolve()
    np.testing.assert_allclose(child.transform.rotation, quat_identity(), atol=1e-5)


# ----- deformation order -----------------------------------------------
def test_deformation_order_resolves_chained_appends_correctly() -> None:
    """``a → b → c``: ``b`` inherits from ``a``, ``c`` inherits from ``b``.

    With ``a`` rotated 60° around Y and both inheritance weights at 1.0,
    ``b`` and ``c`` should both end up at 60° around Y after one pass —
    proving that ``c``'s append picks up ``b``'s already-resolved
    rotation rather than ``b``'s initial identity.
    """
    a = Node(name="a", transform=Transform(translation=vec3(0, 0, 0)))
    b = Node(name="b", transform=Transform(translation=vec3(1, 0, 0)))
    c = Node(name="c", transform=Transform(translation=vec3(2, 0, 0)))
    nodes = {0: a, 1: b, 2: c}
    resolver = _resolver_for(
        nodes,
        appends=(
            BoneAppendRule(bone_index=1, parent_index=0, weight=1.0, inherit_rotation=True),
            BoneAppendRule(bone_index=2, parent_index=1, weight=1.0, inherit_rotation=True),
        ),
    )
    a.transform.set_rotation(quat_from_euler_xyz(0.0, np.deg2rad(60), 0.0))
    resolver.resolve()
    _rx_b, ry_b, _rz_b = quat_to_euler_xyz(b.transform.rotation)
    _rx_c, ry_c, _rz_c = quat_to_euler_xyz(c.transform.rotation)
    assert np.rad2deg(ry_b) == pytest.approx(60.0, abs=0.5)
    assert np.rad2deg(ry_c) == pytest.approx(60.0, abs=0.5)


# ----- fixed-axis projection -------------------------------------------
def test_fixed_axis_projects_yz_components_to_zero() -> None:
    """A rotation built from non-X-only Eulers, projected onto the X axis,
    must keep its X-axis component and drop the Y / Z parts."""
    bone = Node(name="bone", transform=Transform())
    bone.transform.set_rotation(
        quat_from_euler_xyz(np.deg2rad(30), np.deg2rad(45), np.deg2rad(20))
    )
    nodes = {0: bone}
    resolver = _resolver_for(
        nodes,
        fixed_axes=(FixedAxisRule(bone_index=0, axis=(1.0, 0.0, 0.0)),),
    )
    resolver.resolve()
    _rx, ry, rz = quat_to_euler_xyz(bone.transform.rotation)
    assert abs(np.rad2deg(ry)) < 0.5
    assert abs(np.rad2deg(rz)) < 0.5


def test_fixed_axis_preserves_axis_aligned_rotation_unchanged() -> None:
    """A rotation that already lies entirely on the fixed axis is preserved."""
    bone = Node(name="bone", transform=Transform())
    bone.transform.set_rotation(quat_from_axis_angle(vec3(1, 0, 0), np.deg2rad(45)))
    expected = bone.transform.rotation.copy()
    nodes = {0: bone}
    resolver = _resolver_for(
        nodes,
        fixed_axes=(FixedAxisRule(bone_index=0, axis=(1.0, 0.0, 0.0)),),
    )
    resolver.resolve()
    np.testing.assert_allclose(bone.transform.rotation, expected, atol=1e-5)


def test_fixed_axis_zero_axis_is_noop() -> None:
    bone = Node(name="bone", transform=Transform())
    bone.transform.set_rotation(quat_from_euler_xyz(np.deg2rad(30), 0.0, 0.0))
    expected = bone.transform.rotation.copy()
    nodes = {0: bone}
    resolver = _resolver_for(
        nodes,
        fixed_axes=(FixedAxisRule(bone_index=0, axis=(0.0, 0.0, 0.0)),),
    )
    resolver.resolve()
    np.testing.assert_allclose(bone.transform.rotation, expected, atol=1e-5)


def test_quat_project_to_axis_idempotent() -> None:
    """Projecting twice should give the same result as projecting once."""
    q = quat_from_euler_xyz(np.deg2rad(30), np.deg2rad(45), np.deg2rad(20))
    once = quat_project_to_axis(q, vec3(1, 0, 0))
    twice = quat_project_to_axis(once, vec3(1, 0, 0))
    np.testing.assert_allclose(once, twice, atol=1e-5)


# ----- PMX importer extracts rules -------------------------------------
def test_pmx_importer_picks_up_append_and_fixed_axis(tmp_path: Path) -> None:
    """A model with both an append bone and a fixed-axis bone must surface
    the rules in :attr:`ImportedScene.bone_resolver_rules`."""
    spec = FixtureBuild(
        name_jp="r", name_en="r",
        vertices=(
            FixtureVertex(position=(-1, -1, -1), deform=_Bdef1(bone=0)),
            FixtureVertex(position=(1, -1, -1), deform=_Bdef1(bone=0)),
            FixtureVertex(position=(0, 1, 0), deform=_Bdef1(bone=0)),
        ),
        indices=(0, 2, 1),
        materials=(FixtureMaterial(name_jp="m", face_index_count=3),),
        bones=(
            FixtureBone(name_jp="main", position=(0, 1, 0), parent_index=-1),
            FixtureBone(
                name_jp="follower", position=(1, 1, 0), parent_index=-1,
                inherit_parent_index=0, inherit_weight=0.5, inherit_rotation=True,
                deformation_depth=1,
            ),
            FixtureBone(
                name_jp="hair_tip", position=(0, 2, 0), parent_index=0,
                fixed_axis=(1.0, 0.0, 0.0), deformation_depth=2,
            ),
        ),
    )
    path = tmp_path / "rules.pmx"
    path.write_bytes(build_pmx(spec))
    scene = PmxImporter().load(path)
    rules = scene.bone_resolver_rules
    assert rules.deformation_order == (0, 1, 2)
    assert len(rules.appends) == 1
    append = rules.appends[0]
    assert append.bone_index == 1
    assert append.parent_index == 0
    assert append.weight == pytest.approx(0.5)
    assert append.inherit_rotation
    assert not append.inherit_translation
    assert len(rules.fixed_axes) == 1
    assert rules.fixed_axes[0].bone_index == 2


def test_pmx_importer_sorts_deformation_order_by_depth(tmp_path: Path) -> None:
    spec = FixtureBuild(
        name_jp="r", name_en="r",
        vertices=(
            FixtureVertex(position=(-1, -1, -1), deform=_Bdef1(bone=0)),
            FixtureVertex(position=(1, -1, -1), deform=_Bdef1(bone=0)),
            FixtureVertex(position=(0, 1, 0), deform=_Bdef1(bone=0)),
        ),
        indices=(0, 2, 1),
        materials=(FixtureMaterial(name_jp="m", face_index_count=3),),
        bones=(
            FixtureBone(name_jp="early", position=(0, 0, 0), parent_index=-1, deformation_depth=10),
            FixtureBone(name_jp="late", position=(1, 0, 0), parent_index=-1, deformation_depth=2),
            FixtureBone(name_jp="mid", position=(2, 0, 0), parent_index=-1, deformation_depth=5),
        ),
    )
    path = tmp_path / "depth.pmx"
    path.write_bytes(build_pmx(spec))
    scene = PmxImporter().load(path)
    # late (depth 2) → mid (5) → early (10)
    assert scene.bone_resolver_rules.deformation_order == (1, 2, 0)


# Keep the unused symbols load-bearing for IDE jumps.
__all__ = ["replace", "tempfile"]
