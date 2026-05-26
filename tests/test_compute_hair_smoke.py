"""Smoke tests for the hair-spring GPU compute infrastructure.

No GL context required — verifies the shader source loads, packing
helpers produce the right buffer layout, and the dispatcher dataclass
defaults are sane. Real-dispatch tests with an offscreen context live
in ``test_compute_hair_offscreen.py`` (those skip cleanly when GL
isn't available).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from posecascade.gl.compute_hair import (
    ANCHOR_FLOATS,
    CAPSULE_FLOATS,
    CHAIN_INFO_FLOATS,
    JOINT_FLOATS,
    JOINT_STRIDE,
    SIM_CONSTANTS_FLOATS,
    SPHERE_FLOATS,
    HairChainSlot,
    HairComputeDispatcher,
    _pack_chain_info,
    _pack_chain_joints,
    _pack_sim_constants,
    pack_anchor_xforms,
    pack_capsules,
    pack_spheres,
)


def test_shader_source_loads() -> None:
    shader = (
        Path(__file__).resolve().parents[1]
        / "shaders" / "hair" / "hair_spring.comp"
    )
    src = shader.read_text(encoding="utf-8")
    assert "#version 430 core" in src
    assert "hair / spring chain physics" in src
    assert "JointStateBuffer" in src
    assert "SimConstants" in src
    assert "ChainInfoBuffer" in src


def test_joint_stride_matches_constant() -> None:
    assert JOINT_FLOATS == 32
    assert JOINT_STRIDE == JOINT_FLOATS * 4


def test_chain_info_floats() -> None:
    assert CHAIN_INFO_FLOATS == 12


def test_anchor_floats() -> None:
    assert ANCHOR_FLOATS == 8


def test_sphere_floats() -> None:
    assert SPHERE_FLOATS == 8


def test_capsule_floats() -> None:
    assert CAPSULE_FLOATS == 8


def test_sim_constants_floats() -> None:
    assert SIM_CONSTANTS_FLOATS == 12


def _stub_joint(
    world_rot: list[float],
    world_pos: list[float],
    rest_pos: list[float],
    bone_vec: list[float],
    inertia: float = 1.0,
) -> object:
    """Minimal duck-typed SpringJoint for packing-helper tests."""
    return type("J", (), {
        "world_rotation": np.array(world_rot, dtype=np.float32),
        "world_position": np.array(world_pos, dtype=np.float32),
        "angular_velocity": np.array([0, 0, 0], dtype=np.float32),
        "rest_local_position": np.array(rest_pos, dtype=np.float32),
        "rest_local_rotation": np.array([0, 0, 0, 1], dtype=np.float32),
        "bone_vector_local": np.array(bone_vec, dtype=np.float32),
        "rest_in_anchor_frame": np.array([0, 0, 0, 1], dtype=np.float32),
        "inertia": inertia,
    })()


def test_pack_chain_joints_layout() -> None:
    joints = (
        _stub_joint([0, 0, 0, 1], [0.1, 0.2, 0.3], [0, 0, 0], [0, 0.1, 0]),
        _stub_joint([0, 0, 0, 1], [0.1, 0.3, 0.3], [0, 0.1, 0], [0, 0.1, 0]),
    )
    packed = _pack_chain_joints(joints)
    assert packed.shape == (2, JOINT_FLOATS)
    assert packed[0, 3] == 1.0  # quaternion w
    assert packed[0, 4] == 0.1  # world_position x
    # bone length packed as the 4th element of the bone_vector_local vec4.
    assert packed[0, 23] == np.float32(0.1)


def test_pack_chain_info_preserves_integers() -> None:
    row = _pack_chain_info(
        joint_start=5,
        joint_count=10,
        anchor_idx=2,
        stiffness=8.0,
        damping=1.5,
        inertia=1.0,
        max_swing_rad=0.78539816,  # 45° — verifies the new slot.
    )
    assert row.shape == (CHAIN_INFO_FLOATS,)
    # Reinterpret the float bits back to uint to verify the integer
    # round-trip — same trick the shader does via ``floatBitsToUint``.
    assert np.uint32(row[0].view(np.uint32)) == 5
    assert np.uint32(row[1].view(np.uint32)) == 10
    assert np.uint32(row[2].view(np.uint32)) == 2
    assert row[3] == 8.0
    assert row[6] == np.float32(0.78539816)


def test_pack_anchor_xforms_layout() -> None:
    rot = np.array([0, 0, 0, 1], dtype=np.float32)
    pos = np.array([1, 2, 3], dtype=np.float32)
    packed = pack_anchor_xforms([(rot, pos), (rot, pos)])
    assert packed.shape == (2, ANCHOR_FLOATS)
    assert packed[0, 3] == 1.0
    assert packed[0, 4:7].tolist() == [1.0, 2.0, 3.0]


def test_pack_spheres_layout() -> None:
    center = np.array([1, 2, 3], dtype=np.float32)
    packed = pack_spheres([(center, 0.1, 0.01)])
    assert packed.shape == (1, SPHERE_FLOATS)
    assert packed[0, 0] == 1.0
    assert packed[0, 3] == np.float32(0.1)
    assert packed[0, 4] == np.float32(0.01)


def test_pack_capsules_layout() -> None:
    a = np.array([0, 0, 0], dtype=np.float32)
    b = np.array([1, 0, 0], dtype=np.float32)
    packed = pack_capsules([(a, b, 0.05, 0.005)])
    assert packed.shape == (1, CAPSULE_FLOATS)
    assert packed[0, 3] == np.float32(0.05)
    assert packed[0, 4:7].tolist() == [1.0, 0.0, 0.0]
    assert packed[0, 7] == np.float32(0.005)


def test_pack_sim_constants() -> None:
    ubo = _pack_sim_constants(
        dt=1.0 / 120.0,
        gravity=np.array([0, -9.8, 0], dtype=np.float32),
        wind=np.array([1, 0, 0], dtype=np.float32),
        sphere_count=5,
        capsule_count=10,
    )
    assert ubo.shape == (SIM_CONSTANTS_FLOATS,)
    assert ubo[0] == np.float32(1.0 / 120.0)
    assert ubo[5] == np.float32(-9.8)
    # sphere_count + capsule_count encoded as raw uint bit patterns.
    assert np.uint32(ubo[7].view(np.uint32)) == 5
    assert np.uint32(ubo[11].view(np.uint32)) == 10


def test_hair_chain_slot_default() -> None:
    slot = HairChainSlot(
        chain_name="BackHair_L",
        joint_start=0,
        joint_count=5,
        anchor_idx=0,
    )
    assert slot.joints_cpu == ()


def test_dispatcher_dataclass_has_destroy() -> None:
    # We can't instantiate without a GL context, but the type should
    # carry the lifecycle methods the renderer relies on.
    assert hasattr(HairComputeDispatcher, "try_create")
    assert hasattr(HairComputeDispatcher, "register_chains")
    assert hasattr(HairComputeDispatcher, "dispatch")
    assert hasattr(HairComputeDispatcher, "readback_to_cpu")
    assert hasattr(HairComputeDispatcher, "destroy")
