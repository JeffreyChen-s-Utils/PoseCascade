"""Tests for :class:`PhysicsWorld` — integration, modes, joints, frame jump."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pmx.importer import PmxImporter

from posecascade.physics import (
    Joint6DofSpring,
    PhysicsMode,
    PhysicsScene,
    PhysicsWorld,
    RigidBody,
    RigidShape,
)
from posecascade.physics.world import SCRUB_DT_CAP_SECONDS
from posecascade.scene.node import Node
from posecascade.scene.transform import Transform
from posecascade.utils.math3d import vec3
from tests.fixtures.mmd.build import (
    FixtureBone,
    FixtureBuild,
    FixtureJoint,
    FixtureMaterial,
    FixtureRigidBody,
    FixtureVertex,
    _Bdef1,
    build_pmx,
)


def _ball(
    name: str = "ball",
    *,
    position: tuple[float, float, float] = (0.0, 5.0, 0.0),
    mode: PhysicsMode = PhysicsMode.DYNAMIC,
    bone_index: int = -1,
    mass: float = 1.0,
) -> RigidBody:
    return RigidBody(
        name=name, bone_index=bone_index, group=0, non_collision_mask=0,
        shape=RigidShape.SPHERE, size=(0.25, 0.0, 0.0),
        position=position, rotation=(0.0, 0.0, 0.0),
        mass=mass, linear_damping=0.0, angular_damping=0.0,
        restitution=0.0, friction=0.5, physics_mode=mode,
    )


def _step_for_total(world: PhysicsWorld, total: float, *, chunk: float = 0.05) -> None:
    """Step ``total`` seconds in chunks below the scrub cap."""
    elapsed = 0.0
    while elapsed < total:
        slice_dt = min(chunk, total - elapsed)
        world.step(slice_dt)
        elapsed += slice_dt


# ----- free integration -------------------------------------------------
def test_dynamic_body_falls_under_gravity() -> None:
    world = PhysicsWorld(scene=PhysicsScene(bodies=(_ball(),)))
    _step_for_total(world, 1.0)
    final_y = float(world.body_position(0)[1])
    # Semi-implicit Euler at 60 Hz under -9.8 m/s² for 1 s ≈ 0.1 m above
    # the analytical solution. Allow ±0.5 m.
    assert -0.5 < final_y < 1.0, f"unexpected y after 1s: {final_y}"
    assert float(world.body_linear_velocity(0)[1]) < -8.0


def test_kinematic_body_does_not_move_without_bone() -> None:
    """A kinematic body without a bone Node stays at its rest pose."""
    body = _ball(mode=PhysicsMode.KINEMATIC)
    world = PhysicsWorld(scene=PhysicsScene(bodies=(body,)))
    _step_for_total(world, 0.5)
    np.testing.assert_allclose(world.body_position(0), [0, 5, 0], atol=1e-5)


def test_kinematic_body_follows_bone_node() -> None:
    bone = Node(name="follow", transform=Transform(translation=vec3(0, 0, 0)))
    body = _ball(position=(0.0, 0.0, 0.0), mode=PhysicsMode.KINEMATIC, bone_index=0)
    world = PhysicsWorld(
        scene=PhysicsScene(bodies=(body,)),
        bone_index_to_node={0: bone},
    )
    bone.transform.set_translation(vec3(2.0, 1.0, -1.0))
    _step_for_total(world, 0.05, chunk=0.05)
    np.testing.assert_allclose(world.body_position(0), [2.0, 1.0, -1.0], atol=1e-5)


def test_dynamic_bone_mode_keeps_position_at_bone() -> None:
    """Hybrid mode: physics may rotate freely but position stays at the bone."""
    bone = Node(name="hinge", transform=Transform(translation=vec3(1, 1, 0)))
    body = _ball(position=(1.0, 1.0, 0.0), mode=PhysicsMode.DYNAMIC_BONE, bone_index=0)
    world = PhysicsWorld(
        scene=PhysicsScene(bodies=(body,)),
        bone_index_to_node={0: bone},
    )
    _step_for_total(world, 1.0)
    # Without a joint the spinning is just gravity-driven angular drift —
    # the position hooks back to the bone every tick.
    np.testing.assert_allclose(world.body_position(0), [1.0, 1.0, 0.0], atol=1e-3)


# ----- joints -----------------------------------------------------------
def test_spring_joint_pulls_drifted_body_back_inside_limit() -> None:
    """A damped 6DOF spring whose linear range is 0 should converge the
    anchored body toward the joint origin. We add linear damping so the
    oscillation decays — without it a lossless spring just oscillates
    forever and a fixed-time sample lands at any phase."""
    body = RigidBody(
        name="ball", bone_index=-1, group=0, non_collision_mask=0,
        shape=RigidShape.SPHERE, size=(0.25, 0.0, 0.0),
        position=(0.5, 0.0, 0.0), rotation=(0.0, 0.0, 0.0),
        mass=1.0, linear_damping=4.0, angular_damping=0.0,
        restitution=0.0, friction=0.5, physics_mode=PhysicsMode.DYNAMIC,
    )
    joint = Joint6DofSpring(
        name="anchor", rigid_a_index=0, rigid_b_index=0,
        position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0),
        linear_lower=(0.0, 0.0, 0.0), linear_upper=(0.0, 0.0, 0.0),
        angular_lower=(0.0, 0.0, 0.0), angular_upper=(0.0, 0.0, 0.0),
        spring_linear=(60.0, 0.0, 0.0), spring_angular=(0.0, 0.0, 0.0),
    )
    world = PhysicsWorld(scene=PhysicsScene(bodies=(body,), joints=(joint,)))
    world.set_gravity((0.0, 0.0, 0.0))
    _step_for_total(world, 2.0)
    final_x = float(world.body_position(0)[0])
    assert abs(final_x) < 0.2, f"damped spring did not converge: |x|={abs(final_x):.3f}"


# ----- frame jump / scrub ----------------------------------------------
def test_step_above_scrub_cap_resets_to_rest() -> None:
    body = _ball()
    world = PhysicsWorld(scene=PhysicsScene(bodies=(body,)))
    _step_for_total(world, 0.4)
    moved_y = float(world.body_position(0)[1])
    assert moved_y < 5.0
    world.step(SCRUB_DT_CAP_SECONDS + 0.1)
    np.testing.assert_allclose(world.body_position(0), [0, 5, 0], atol=1e-5)


def test_reset_can_warm_up_dynamic_bodies() -> None:
    """``reset(warmup_steps=N)`` snaps + steps so the user doesn't see a
    discrete jump after a timeline scrub."""
    body = _ball()
    world = PhysicsWorld(scene=PhysicsScene(bodies=(body,)))
    world.reset(warmup_steps=30)                    # 30 ticks ≈ 0.5 s
    final_y = float(world.body_position(0)[1])
    assert final_y < 4.5, f"warm-up did not advance the simulation: {final_y}"


# ----- importer integration --------------------------------------------
def test_pmx_importer_extracts_bodies_and_joints(tmp_path: Path) -> None:
    spec = FixtureBuild(
        name_jp="p", name_en="p",
        vertices=(
            FixtureVertex(position=(-1, -1, -1), deform=_Bdef1(bone=0)),
            FixtureVertex(position=(1, -1, -1), deform=_Bdef1(bone=0)),
            FixtureVertex(position=(0, 1, 0), deform=_Bdef1(bone=0)),
        ),
        indices=(0, 2, 1),
        materials=(FixtureMaterial(name_jp="m", face_index_count=3),),
        bones=(
            FixtureBone(name_jp="root", position=(0, 1, 0), parent_index=-1),
        ),
        rigid_bodies=(
            FixtureRigidBody(name="ball", bone_index=0, position=(0, 1, 0), physics_mode=2),
            FixtureRigidBody(name="free", bone_index=-1, position=(0, 5, 0), physics_mode=1),
        ),
        joints=(
            FixtureJoint(
                name="link", rigid_a=0, rigid_b=1,
                position=(0, 3, 0),
                spring_linear=(50.0, 50.0, 50.0),
                spring_angular=(0.0, 0.0, 0.0),
            ),
        ),
    )
    path = tmp_path / "physics.pmx"
    path.write_bytes(build_pmx(spec))
    scene = PmxImporter().load(path)
    assert len(scene.physics_scene.bodies) == 2
    assert scene.physics_scene.bodies[0].physics_mode == PhysicsMode.DYNAMIC_BONE
    assert scene.physics_scene.bodies[1].physics_mode == PhysicsMode.DYNAMIC
    assert len(scene.physics_scene.joints) == 1


def test_player_steps_physics_only_after_initial_frame() -> None:
    """First :meth:`apply` after construction should snap, not integrate."""
    from posecascade.animation.player import VmdAnimationPlayer  # noqa: PLC0415
    from posecascade.animation.vmd_track import VmdMotionAsset  # noqa: PLC0415

    spec = FixtureBuild(
        name_jp="p", name_en="p",
        vertices=(
            FixtureVertex(position=(-1, -1, -1), deform=_Bdef1(bone=0)),
            FixtureVertex(position=(1, -1, -1), deform=_Bdef1(bone=0)),
            FixtureVertex(position=(0, 1, 0), deform=_Bdef1(bone=0)),
        ),
        indices=(0, 2, 1),
        materials=(FixtureMaterial(name_jp="m", face_index_count=3),),
        bones=(FixtureBone(name_jp="root", position=(0, 0, 0), parent_index=-1),),
        rigid_bodies=(
            FixtureRigidBody(
                name="free", bone_index=-1,
                position=(0, 5, 0), physics_mode=1,
            ),
        ),
    )
    pmx_path = pytest.importorskip("pathlib").Path("/tmp/_phys_player.pmx")
    pmx_path.parent.mkdir(parents=True, exist_ok=True)
    pmx_path.write_bytes(build_pmx(spec))
    scene = PmxImporter().load(pmx_path)
    motion = VmdMotionAsset(target_model_name="p")
    player = VmdAnimationPlayer.for_imported_scene(motion, scene)
    player.apply(0.0)
    initial_y = float(player._physics_world.body_position(0)[1])  # noqa: SLF001
    player.apply(0.5)
    later_y = float(player._physics_world.body_position(0)[1])    # noqa: SLF001
    assert later_y < initial_y, "physics step did not run on the second apply"
