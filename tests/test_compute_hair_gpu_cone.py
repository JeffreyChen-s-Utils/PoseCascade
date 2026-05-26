"""GPU compute shader end-to-end test for the angular-cone clamp.

Compiles ``shaders/hair/hair_spring.comp``, dispatches with a synthetic
single-joint chain under strong gravity, reads the joint state back,
and asserts the swing angle from rest was hard-clamped to
``max_swing_rad``. This is the GPU counterpart of
``test_spring_swing_cone.test_cone_clamps_swing_to_max_angle``; without
it, the shader path could silently regress.

Skips when GL 4.3 / compute support / offscreen context is unavailable
(headless CI without a GPU driver).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


def _try_make_offscreen_context():
    """Return (app, surface, context) or None if GL/compute isn't available."""
    try:
        from examples.mmd_demo import (  # noqa: PLC0415
            _make_offscreen_gl_context,
        )
    except ImportError:
        return None
    try:
        return _make_offscreen_gl_context(256, 256)
    except Exception:                                                        # noqa: BLE001
        return None


def test_gpu_cone_clamps_swing_under_gravity() -> None:
    ctx = _try_make_offscreen_context()
    if ctx is None:
        pytest.skip("no GL 4.3 / compute support in this environment")

    from posecascade.gl.compute_hair import (  # noqa: PLC0415
        HairComputeDispatcher,
        _pack_chain_info,
        _pack_chain_joints,
    )

    shader_path = Path(__file__).parent.parent / "shaders" / "hair" / "hair_spring.comp"
    dispatcher = HairComputeDispatcher.try_create(shader_path)
    if dispatcher is None:
        pytest.skip("hair_spring.comp failed to compile (no compute support)")

    # Build a synthetic single-joint chain by hand. Bone points +X with
    # 10 cm length, rest local rotation identity (chain rest direction =
    # parent X axis). max_swing_rad = 30°.
    max_swing_rad = math.radians(30.0)
    joint = type("J", (), {
        "world_rotation": np.array([0, 0, 0, 1], dtype=np.float32),
        "world_position": np.array([0.1, 0, 0], dtype=np.float32),
        "angular_velocity": np.array([0, 0, 0], dtype=np.float32),
        "rest_local_position": np.array([0.1, 0, 0], dtype=np.float32),
        "rest_local_rotation": np.array([0, 0, 0, 1], dtype=np.float32),
        "bone_vector_local": np.array([0.1, 0, 0], dtype=np.float32),
        "rest_in_anchor_frame": np.array([0, 0, 0, 1], dtype=np.float32),
        "inertia": 1.0,
    })()
    block = _pack_chain_joints((joint,))
    chain_row = _pack_chain_info(
        joint_start=0, joint_count=1, anchor_idx=0,
        stiffness=0.5, damping=0.1, inertia=1.0,
        max_swing_rad=max_swing_rad,
    )

    from OpenGL.GL import (  # noqa: PLC0415
        GL_DYNAMIC_DRAW,
        GL_SHADER_STORAGE_BUFFER,
        GL_STATIC_DRAW,
        glBindBuffer,
        glBufferData,
    )

    # Upload buffers manually (mimics register_chains for one chain).
    joint_block = np.ascontiguousarray(block.astype(np.float32))
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, dispatcher.joint_state_ssbo)
    glBufferData(
        GL_SHADER_STORAGE_BUFFER, joint_block.nbytes, joint_block,
        GL_DYNAMIC_DRAW,
    )
    chain_block = np.ascontiguousarray(chain_row.astype(np.float32))
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, dispatcher.chain_info_ssbo)
    glBufferData(
        GL_SHADER_STORAGE_BUFFER, chain_block.nbytes, chain_block,
        GL_STATIC_DRAW,
    )
    dispatcher.joint_capacity = 1
    dispatcher.chain_capacity = 1
    dispatcher.slots = [type("S", (), {
        "chain_name": "test", "joint_start": 0, "joint_count": 1,
        "anchor_idx": 0, "joints_cpu": (joint,),
    })()]

    # Anchor at identity.
    anchor_xforms = np.array([[0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.float32)
    spheres = np.zeros((0, 8), dtype=np.float32)
    capsules = np.zeros((0, 8), dtype=np.float32)
    # Drive 1000 small substeps with strong gravity to saturate the cone.
    gravity = np.array([0, -100.0, 0], dtype=np.float32)
    wind = np.array([0, 0, 0], dtype=np.float32)
    for _ in range(1000):
        dispatcher.dispatch(
            dt=1.0 / 240.0, gravity=gravity, wind=wind,
            anchor_xforms=anchor_xforms, spheres=spheres, capsules=capsules,
        )
    state = dispatcher.readback_to_cpu()
    world_quat = state[0, 0:4]
    # Angle = 2 × atan2(|sin_half|, cos_half).
    qw = float(world_quat[3])
    if qw < 0.0:
        world_quat = -world_quat
        qw = float(world_quat[3])
    sin_half = float(np.linalg.norm(world_quat[:3]))
    angle = 2.0 * math.atan2(sin_half, qw)
    assert angle <= max_swing_rad + math.radians(3.0), (
        f"GPU cone clamp failed: swing {math.degrees(angle):.1f}° > "
        f"{math.degrees(max_swing_rad):.0f}° + 3° tolerance"
    )

    dispatcher.destroy()
