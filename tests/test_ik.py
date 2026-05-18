"""Tests for the CCD IK solver and PMX → IK extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pmx.importer import PmxImporter

from posecascade.animation.ik import (
    IkChain,
    IkLink,
    align_bone_axis_to_world,
    solve_chain,
    solve_two_bone,
    solve_two_bone_analytic,
)
from posecascade.scene.node import Node
from posecascade.scene.transform import Transform
from posecascade.scripting.api import _IK_SINGLETON
from posecascade.utils.math3d import (
    quat_from_euler_xyz,
    quat_identity,
    quat_to_euler_xyz,
    vec3,
)

_TINY_LEG_PMX = Path(__file__).resolve().parent / "fixtures" / "mmd" / "tiny_leg.pmx"


def _world_position(node: Node) -> np.ndarray:
    matrix = node.transform.to_matrix()
    parent = node.parent
    while parent is not None:
        matrix = parent.transform.to_matrix() @ matrix
        parent = parent.parent
    return np.array([matrix[0, 3], matrix[1, 3], matrix[2, 3]], dtype=np.float32)


# ----- single-link IK ---------------------------------------------------
def test_single_link_rotates_to_reach_driver() -> None:
    """A 1-link chain rotates the link bone until the effector lands at the driver."""
    root = Node(name="root", transform=Transform(translation=vec3(0, 0, 0)))
    effector = Node(name="effector", transform=Transform(translation=vec3(0, 1, 0)))
    driver = Node(name="driver", transform=Transform(translation=vec3(1, 0, 0)))
    root.add_child(effector)
    chain = IkChain(
        driver_bone_index=2, effector_bone_index=1,
        iterations=10, limit_radian=0.0,
        links=(IkLink(bone_index=0),),
    )
    solve_chain(chain, {0: root, 1: effector, 2: driver})
    np.testing.assert_allclose(_world_position(effector), [1, 0, 0], atol=1e-4)


def test_single_link_already_at_target_does_nothing() -> None:
    """When the effector already sits at the driver, the chain leaves the
    rotation untouched."""
    root = Node(name="root", transform=Transform(translation=vec3(0, 0, 0)))
    effector = Node(name="effector", transform=Transform(translation=vec3(1, 0, 0)))
    driver = Node(name="driver", transform=Transform(translation=vec3(1, 0, 0)))
    root.add_child(effector)
    chain = IkChain(
        driver_bone_index=2, effector_bone_index=1,
        iterations=10, limit_radian=0.0,
        links=(IkLink(bone_index=0),),
    )
    solve_chain(chain, {0: root, 1: effector, 2: driver})
    np.testing.assert_allclose(root.transform.rotation, quat_identity(), atol=1e-5)


def test_disabled_chain_does_not_rotate() -> None:
    root = Node(name="root", transform=Transform(translation=vec3(0, 0, 0)))
    effector = Node(name="effector", transform=Transform(translation=vec3(0, 1, 0)))
    driver = Node(name="driver", transform=Transform(translation=vec3(1, 0, 0)))
    root.add_child(effector)
    chain = IkChain(
        driver_bone_index=2, effector_bone_index=1,
        iterations=10, limit_radian=0.0,
        links=(IkLink(bone_index=0),),
        enabled=False,
    )
    solve_chain(chain, {0: root, 1: effector, 2: driver})
    np.testing.assert_allclose(root.transform.rotation, quat_identity(), atol=1e-5)


def test_missing_driver_or_effector_silently_skips() -> None:
    """A chain whose driver or effector index isn't in the lookup must not crash."""
    root = Node(name="root", transform=Transform(translation=vec3(0, 0, 0)))
    chain = IkChain(
        driver_bone_index=99, effector_bone_index=98,
        iterations=10, limit_radian=0.0,
        links=(IkLink(bone_index=0),),
    )
    solve_chain(chain, {0: root})
    np.testing.assert_allclose(root.transform.rotation, quat_identity(), atol=1e-5)


# ----- per-step angle limit ---------------------------------------------
def test_limit_radian_bounds_each_iteration_step() -> None:
    """With ``iterations=1`` and a small per-step ``limit_radian``, one pass
    cannot rotate more than ``limit_radian`` even if the goal is a full 90°.
    """
    root = Node(name="root", transform=Transform(translation=vec3(0, 0, 0)))
    effector = Node(name="effector", transform=Transform(translation=vec3(0, 1, 0)))
    driver = Node(name="driver", transform=Transform(translation=vec3(1, 0, 0)))
    root.add_child(effector)
    chain = IkChain(
        driver_bone_index=2, effector_bone_index=1,
        iterations=1, limit_radian=0.05,        # 0.05 rad ≈ 2.86°
        links=(IkLink(bone_index=0),),
    )
    solve_chain(chain, {0: root, 1: effector, 2: driver})
    rx, ry, rz = quat_to_euler_xyz(root.transform.rotation)
    assert abs(rz) <= 0.06, f"rotated {rz} rad exceeds per-step cap"


# ----- two-link knee chain ---------------------------------------------
def test_two_link_knee_chain_reaches_driver_via_canonical_fixture() -> None:
    """Loading the leg fixture + running the chain pulls the ankle close to the driver."""
    scene = PmxImporter().load(_TINY_LEG_PMX)
    chain = scene.ik_chains[0]
    lookup = dict(enumerate(scene.skins[0].joints))
    ankle = lookup[chain.effector_bone_index]
    driver = lookup[chain.driver_bone_index]
    target_world = _world_position(driver)
    solve_chain(chain, lookup)
    final = _world_position(ankle)
    distance = float(np.linalg.norm(final - target_world))
    # CCD with multiple links + an Euler-clamped knee converges to a small
    # residual rather than zero — the chain has limited reach. We assert
    # significant progress relative to the rest pose distance (which is
    # ~2.06 from ankle (0,0,0) to driver (0.5,0.5,0)).
    assert distance < 0.6, f"ankle still {distance:.3f} away from driver"


def test_knee_link_rotation_stays_within_x_axis_limit() -> None:
    """The knee's ``has_limit`` bracket is X-only; Y and Z must stay near zero."""
    scene = PmxImporter().load(_TINY_LEG_PMX)
    chain = scene.ik_chains[0]
    lookup = dict(enumerate(scene.skins[0].joints))
    knee = lookup[1]
    solve_chain(chain, lookup)
    rx, ry, rz = quat_to_euler_xyz(knee.transform.rotation)
    assert abs(ry) < 1e-4, f"knee Y rotation {ry} should be clamped to 0"
    assert abs(rz) < 1e-4, f"knee Z rotation {rz} should be clamped to 0"
    # The X bracket is [-π, 0], so the X angle must land within it.
    assert -3.1416 <= rx <= 0.0001, f"knee X rotation {rx} outside [-π, 0]"


def test_pmx_importer_extracts_one_chain_for_leg_fixture() -> None:
    scene = PmxImporter().load(_TINY_LEG_PMX)
    assert len(scene.ik_chains) == 1
    chain = scene.ik_chains[0]
    assert chain.effector_bone_index == 2     # ankle
    assert len(chain.links) == 2
    assert chain.links[0].bone_index == 1     # knee
    assert chain.links[0].has_limit


# ----- player integration -----------------------------------------------
def test_player_runs_ik_when_chain_enabled() -> None:
    """The unified player should pull the ankle toward the driver every apply()."""
    from posecascade.animation.player import VmdAnimationPlayer  # noqa: PLC0415
    from posecascade.animation.vmd_track import VmdMotionAsset  # noqa: PLC0415

    scene = PmxImporter().load(_TINY_LEG_PMX)
    motion = VmdMotionAsset(target_model_name="tiny")
    player = VmdAnimationPlayer.for_imported_scene(motion, scene)
    ankle = scene.skins[0].joints[2]
    driver = scene.skins[0].joints[3]
    target_world = _world_position(driver)
    pre = float(np.linalg.norm(_world_position(ankle) - target_world))
    player.apply(0.0)
    post = float(np.linalg.norm(_world_position(ankle) - target_world))
    assert post < pre, f"player did not reduce IK error (pre={pre:.3f}, post={post:.3f})"


# ----- two-bone wrapper -------------------------------------------------
def _make_two_bone() -> tuple[Node, Node, Node]:
    """Build a 2-bone chain: root at origin, mid 1 unit up, end 1 unit further up."""
    root = Node(name="root", transform=Transform(translation=vec3(0, 0, 0)))
    mid = Node(name="mid", transform=Transform(translation=vec3(0, 1, 0)))
    end = Node(name="end", transform=Transform(translation=vec3(0, 1, 0)))
    root.add_child(mid)
    mid.add_child(end)
    return root, mid, end


def test_solve_two_bone_reaches_target() -> None:
    """End lands close to a reachable target after the wrapper runs."""
    root, mid, end = _make_two_bone()
    target = vec3(1.5, 1.0, 0.0)
    solve_two_bone(root, mid, end, target, iterations=12)
    err = float(np.linalg.norm(_world_position(end) - target))
    assert err < 0.05, f"end did not reach target (err={err:.4f})"


def test_solve_two_bone_unreachable_target_clamps() -> None:
    """An unreachable target leaves the chain stretched toward it without raising."""
    root, mid, end = _make_two_bone()
    far = vec3(10.0, 0.0, 0.0)
    solve_two_bone(root, mid, end, far, iterations=12)
    # End should sit on the line from root toward target, near max reach (2.0)
    end_pos = _world_position(end)
    assert end_pos[0] > 1.5, f"chain did not stretch toward target (end_x={end_pos[0]:.3f})"


def test_solve_two_bone_hinge_limit_keeps_mid_on_axis() -> None:
    """With Y/Z mid limits clamped to 0, the mid joint only bends around X."""
    root, mid, end = _make_two_bone()
    target = vec3(0.5, 1.5, 0.5)
    solve_two_bone(
        root, mid, end, target,
        iterations=12,
        mid_limit_min=(-2.4, 0.0, 0.0),
        mid_limit_max=(0.1, 0.0, 0.0),
    )
    # Mid local rotation should have no Y or Z Euler component (hinge enforced).
    rx, ry, rz = quat_to_euler_xyz(mid.transform.rotation)
    assert abs(ry) < 1e-3, f"mid Y leaked under hinge limit: {ry:.4f}"
    assert abs(rz) < 1e-3, f"mid Z leaked under hinge limit: {rz:.4f}"


def test_solve_two_bone_rejects_non_node_targets() -> None:
    """Passing a non-Node for any of the three roles raises TypeError via the
    sandbox surface (api.IkApi)."""
    root, mid, _end = _make_two_bone()
    with pytest.raises(TypeError):
        _IK_SINGLETON.solve_two_bone(root, mid, "not_a_node", vec3(0, 0, 0))


# ----- analytical 2-bone IK (foot planting solver) ---------------------
def _make_analytic_chain() -> tuple[Node, Node, Node]:
    """Three-segment analytic chain: root at origin, mid 1u down, end 1u below mid."""
    root = Node(name="root", transform=Transform(translation=vec3(0.0, 0.0, 0.0)))
    mid = Node(name="mid", transform=Transform(translation=vec3(0.0, -1.0, 0.0)))
    end = Node(name="end", transform=Transform(translation=vec3(0.0, -1.0, 0.0)))
    root.add_child(mid)
    mid.add_child(end)
    return root, mid, end


def test_solve_two_bone_analytic_reaches_target_inside_annulus() -> None:
    """When the target is reachable, the foot lands EXACTLY on it.

    Closed-form IK has no convergence error to tune for; this test pins
    that property so a future refactor that re-introduces iteration
    will be caught.
    """
    root, mid, end = _make_analytic_chain()
    target = vec3(1.0, -1.0, 0.0)  # within annulus [0, 2]
    solve_two_bone_analytic(root, mid, end, target)
    np.testing.assert_allclose(_world_position(end), target, atol=1e-4)


def test_solve_two_bone_analytic_clamps_target_beyond_max_reach() -> None:
    """When the target is past L1+L2, the leg fully extends toward it
    and the foot lands at the closest reachable point along the
    hip→target line — never further than ``L1+L2`` from the root."""
    root, mid, end = _make_analytic_chain()
    target = vec3(0.0, -5.0, 0.0)  # 5 units down, max reach 2.0
    solve_two_bone_analytic(root, mid, end, target)
    end_pos = _world_position(end)
    distance = float(np.linalg.norm(end_pos))
    assert distance <= 2.0 + 1e-3, f"end at distance {distance} exceeds reach 2.0"
    # Direction should match: foot pointed straight down toward target.
    np.testing.assert_allclose(end_pos / distance, [0, -1, 0], atol=1e-3)


def test_solve_two_bone_analytic_clamps_target_inside_fold_hole() -> None:
    """A target closer than ``|L1-L2|`` from the root sits inside the
    chain's "minimum fold" — analytical IK places the foot at the
    nearest reachable distance, NOT at the unreachable target."""
    # L1=L2=1.0, fold hole = d < 0; degenerate (no inner hole). Make
    # an asymmetric chain so the inner hole has positive radius.
    root = Node(name="root", transform=Transform(translation=vec3(0.0, 0.0, 0.0)))
    mid = Node(name="mid", transform=Transform(translation=vec3(0.0, -0.5, 0.0)))
    end = Node(name="end", transform=Transform(translation=vec3(0.0, -2.0, 0.0)))
    root.add_child(mid)
    mid.add_child(end)
    # L1 = 0.5, L2 = 2.0. Inner fold hole = d < 1.5. Outer = d < 2.5.
    target = vec3(0.0, -1.0, 0.0)  # d=1.0 < 1.5 → inside fold hole.
    solve_two_bone_analytic(root, mid, end, target)
    end_pos = _world_position(end)
    distance = float(np.linalg.norm(end_pos))
    # Should land on the INNER reach boundary (≈ 1.5) along H→T direction.
    assert 1.4 < distance < 1.6, f"end at distance {distance} not on fold-hole boundary"


# ----- end-bone axis alignment (sole / palm flat to ground) ----------------
def _world_axis(node: Node, axis_local: tuple[float, float, float]) -> np.ndarray:
    matrix = node.transform.to_matrix()
    parent = node.parent
    while parent is not None:
        matrix = parent.transform.to_matrix() @ matrix
        parent = parent.parent
    a = np.asarray(axis_local, dtype=np.float32)
    return matrix[:3, :3] @ a


def test_align_bone_axis_to_world_rotates_to_world_down() -> None:
    """A bone whose local +Y starts at world +Y gets flipped to world -Y."""
    n = Node(name="palm", transform=Transform(rotation=quat_identity()))
    # Initial: local +Y -> world +Y. After alignment: should be world -Y.
    align_bone_axis_to_world(n, (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))
    out = _world_axis(n, (0.0, 1.0, 0.0))
    np.testing.assert_allclose(out, np.array([0.0, -1.0, 0.0]), atol=1e-4)


def test_align_bone_axis_handles_parented_node() -> None:
    """Alignment uses world rotation — respects the parent chain.

    Parent rotates the child by 90° around X (so child's local +Y now
    points world -Z), then align asks for world -Y; the result must
    rotate the child's local +Y from world -Z to world -Y, ignoring
    that the parent already moved it.
    """
    parent = Node(name="parent")
    child = Node(name="child")
    parent.add_child(child)
    parent.transform.set_rotation(
        quat_from_euler_xyz(np.pi / 2, 0.0, 0.0).astype(np.float32, copy=False),
    )
    align_bone_axis_to_world(child, (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))
    out = _world_axis(child, (0.0, 1.0, 0.0))
    np.testing.assert_allclose(out, np.array([0.0, -1.0, 0.0]), atol=1e-4)


def test_align_bone_axis_is_noop_when_already_aligned() -> None:
    """No correction needed when current axis already matches target."""
    n = Node(name="palm", transform=Transform(rotation=quat_identity()))
    rest = n.transform.rotation.copy()
    align_bone_axis_to_world(n, (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    np.testing.assert_allclose(n.transform.rotation, rest, atol=1e-6)


def test_solve_two_bone_analytic_uses_bend_hint_when_colinear() -> None:
    """When the current knee is colinear with the hip-target line (no
    bend direction to preserve), ``bend_hint`` selects which side the
    knee folds onto. With root at origin and end straight down at
    (0,-2,0), a target along the same line at (0,-1.5,0) is in the
    annulus but the knee has no preferred bend side — ``bend_hint``
    must pick one."""
    root, mid, end = _make_analytic_chain()
    target = vec3(0.0, -1.5, 0.0)  # colinear with rest leg axis
    solve_two_bone_analytic(root, mid, end, target, bend_hint=vec3(1.0, 0.0, 0.0))
    end_pos = _world_position(end)
    np.testing.assert_allclose(end_pos, target, atol=1e-4)
    knee_pos = _world_position(mid)
    assert knee_pos[0] > 0.1, (
        f"knee at {knee_pos} did not bend toward bend_hint=(+X)"
    )


def test_solve_two_bone_analytic_no_op_at_zero_distance() -> None:
    """Target colocated with root (degenerate H→T direction) returns
    early without crashing."""
    root, mid, end = _make_analytic_chain()
    rest_end = _world_position(end).copy()
    solve_two_bone_analytic(root, mid, end, vec3(0.0, 0.0, 0.0))
    # End should be unchanged — early return on |H→T| < epsilon.
    np.testing.assert_allclose(_world_position(end), rest_end, atol=1e-4)


# Keep `quat_from_euler_xyz` load-bearing so editors find it via cross-reference.
__all__ = ["quat_from_euler_xyz"]
