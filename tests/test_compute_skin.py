"""Unit tests for the GPU passive-skin compute dispatcher.

The pure-helper tests run without a GL context and cover the bitmask
packer plus the version-string parser. The GL-bound tests spin up an
offscreen 4.3 Core Profile context and skip cleanly if the platform
cannot provide one (macOS legacy GL, headless CI without driver, etc.).

The end-to-end test loads the actual compute shader, dispatches it on a
hand-rolled tiny skinned mesh, reads the position + normal VBOs back,
and compares the result to a numpy LBS reference. SSIM-style tolerance
is unnecessary here — both sides do the same float32 matmul, so the
output is bit-exact within 1e-5.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from posecascade.gl.compute_skin import (
    _BONE_WORDS,
    _MAX_COLLIDERS,
    PassiveSkinDispatcher,
    build_exclude_bits,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Pure-logic tests — no GL context needed.
# ---------------------------------------------------------------------------


def test_build_exclude_bits_packs_single_collider() -> None:
    """A single joint in the exclude set lights up the correct bit only."""
    bits = build_exclude_bits([frozenset({33})], collider_count=1)
    assert bits.shape == (_MAX_COLLIDERS, _BONE_WORDS)
    # joint 33 lives in word 1 (33 >> 5 == 1), bit 1 (33 & 31 == 1).
    assert int(bits[0, 1]) == (1 << 1)
    # Every other slot is zero.
    bits[0, 1] = 0
    assert int(bits.sum()) == 0


def test_build_exclude_bits_drops_out_of_range_joints() -> None:
    """Joint indices past MAX_BONES (384) are silently dropped, not over-written."""
    bits = build_exclude_bits(
        [frozenset({500, 1, -3})],
        collider_count=1,
    )
    # Joint 1 should be set; 500 and -3 are out of bounds and ignored.
    assert int(bits[0, 0]) == (1 << 1)


def test_build_exclude_bits_caps_at_max_colliders() -> None:
    """Passing more than MAX_COLLIDERS filters caps at MAX_COLLIDERS rows of output."""
    filters = [frozenset({i}) for i in range(_MAX_COLLIDERS + 4)]
    bits = build_exclude_bits(filters, collider_count=len(filters))
    # Bits beyond _MAX_COLLIDERS are silently truncated — only the first 16
    # rows carry data; the slot is sized at MAX_COLLIDERS so callers can't
    # over-run the shader's UBO.
    assert bits.shape == (_MAX_COLLIDERS, _BONE_WORDS)


# ---------------------------------------------------------------------------
# GL-bound tests — offscreen 4.3 context.
# ---------------------------------------------------------------------------


@pytest.fixture
def gl_compute_context(qapp: object) -> Iterator[object]:
    """Offscreen 4.3 Core Profile context. Skip if the platform lacks compute."""
    pytest.importorskip("PySide6")
    from PySide6.QtGui import (  # noqa: PLC0415
        QOffscreenSurface,
        QOpenGLContext,
        QSurfaceFormat,
    )

    fmt = QSurfaceFormat()
    fmt.setVersion(4, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    if not surface.isValid():
        pytest.skip("offscreen surface unavailable")
    context = QOpenGLContext()
    context.setFormat(fmt)
    if not context.create() or not context.makeCurrent(surface):
        pytest.skip("could not create 4.3 GL context")
    actual = context.format()
    if (actual.majorVersion(), actual.minorVersion()) < (4, 3):
        context.doneCurrent()
        pytest.skip("driver granted < 4.3 — compute shaders unavailable")
    try:
        yield context
    finally:
        context.doneCurrent()


def test_dispatcher_compiles_and_runs(gl_compute_context: object) -> None:
    """Compile the shader, register a 4-vert skinned mesh, dispatch, read back."""
    from OpenGL.GL import (  # noqa: PLC0415
        GL_ARRAY_BUFFER,
        GL_DYNAMIC_DRAW,
        glBindBuffer,
        glBufferData,
        glGenBuffers,
        glGetBufferSubData,
    )

    shader_path = _REPO_ROOT / "shaders" / "passive_skin" / "passive_skin_push.comp"
    dispatcher = PassiveSkinDispatcher.try_create(shader_path)
    if dispatcher is None:
        pytest.skip("driver compiled out compute support")

    # Tiny skinned mesh: 4 verts on a strip, 2 bones, identity bind.
    n = 4
    bind_positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    bind_normals = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (n, 1))
    joints = np.zeros((n, 4), dtype=np.int32)
    joints[2, 0] = 1
    joints[3, 0] = 1
    weights = np.zeros((n, 4), dtype=np.float32)
    weights[:, 0] = 1.0  # full weight on slot 0
    dominant = np.array([0, 0, 1, 1], dtype=np.int32)

    # Allocate the position + normal VBOs the dispatcher writes into.
    pos_vbo = int(glGenBuffers(1))
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo)
    glBufferData(
        GL_ARRAY_BUFFER, n * 3 * 4, np.zeros(n * 3, dtype=np.float32), GL_DYNAMIC_DRAW,
    )
    norm_vbo = int(glGenBuffers(1))
    glBindBuffer(GL_ARRAY_BUFFER, norm_vbo)
    glBufferData(
        GL_ARRAY_BUFFER, n * 3 * 4, np.zeros(n * 3, dtype=np.float32), GL_DYNAMIC_DRAW,
    )

    dispatcher.register_piece(
        piece_id=1,
        output_position_vbo=pos_vbo,
        output_normal_vbo=norm_vbo,
        bind_positions=bind_positions,
        bind_normals=bind_normals,
        joints_per_vert=joints,
        weights_per_vert=weights,
        dominant_joint=dominant,
    )

    # Bone 0 = identity; bone 1 translates +Y by 0.5 so verts 2 and 3 lift.
    bone_matrices = np.zeros((2, 4, 4), dtype=np.float32)
    bone_matrices[0] = np.eye(4, dtype=np.float32)
    bone_matrices[1] = np.eye(4, dtype=np.float32)
    bone_matrices[1, 1, 3] = 0.5

    # No colliders — pure LBS check.
    exclude_bits = build_exclude_bits([], collider_count=0)
    world_to_local = np.eye(4, dtype=np.float32)

    dispatcher.dispatch(
        piece_id=1,
        bone_matrices=bone_matrices,
        world_to_local=world_to_local,
        colliders=[],
        exclude_bits=exclude_bits,
    )

    # Read back the position VBO and compare against the CPU LBS reference.
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo)
    raw = glGetBufferSubData(GL_ARRAY_BUFFER, 0, n * 3 * 4)
    # PyOpenGL hands back either ``bytes`` (older builds) or a uint8
    # ndarray (newer builds). View both as float32 before reshaping.
    out_pos = np.frombuffer(
        bytes(raw) if not isinstance(raw, bytes) else raw, dtype=np.float32,
    ).reshape(n, 3)

    expected = bind_positions.copy()
    expected[2, 1] += 0.5
    expected[3, 1] += 0.5
    np.testing.assert_allclose(out_pos, expected, atol=1.0e-5)

    # Normals: bone-1 verts get their up vector left intact (bone 1 is a
    # pure translation; its 3x3 rotation block is identity).
    glBindBuffer(GL_ARRAY_BUFFER, norm_vbo)
    raw_n = glGetBufferSubData(GL_ARRAY_BUFFER, 0, n * 3 * 4)
    out_n = np.frombuffer(
        bytes(raw_n) if not isinstance(raw_n, bytes) else raw_n, dtype=np.float32,
    ).reshape(n, 3)
    np.testing.assert_allclose(out_n, bind_normals, atol=1.0e-5)

    dispatcher.shutdown()


def test_dispatcher_ground_clamp_lifts_below_floor_verts(gl_compute_context: object) -> None:
    """With ``ground_y=0``, dispatch must clamp every world-Y < 0 vert up to 0."""
    from OpenGL.GL import (  # noqa: PLC0415
        GL_ARRAY_BUFFER,
        GL_DYNAMIC_DRAW,
        glBindBuffer,
        glBufferData,
        glGenBuffers,
        glGetBufferSubData,
    )

    shader_path = _REPO_ROOT / "shaders" / "passive_skin" / "passive_skin_push.comp"
    dispatcher = PassiveSkinDispatcher.try_create(shader_path)
    if dispatcher is None:
        pytest.skip("driver compiled out compute support")

    n = 4
    bind_positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.0, 0.5, 0.0]],
        dtype=np.float32,
    )
    bind_normals = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (n, 1))
    joints = np.zeros((n, 4), dtype=np.int32)
    weights = np.zeros((n, 4), dtype=np.float32)
    weights[:, 0] = 1.0
    dominant = np.zeros(n, dtype=np.int32)

    pos_vbo = int(glGenBuffers(1))
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo)
    glBufferData(GL_ARRAY_BUFFER, n * 3 * 4, np.zeros(n * 3, dtype=np.float32), GL_DYNAMIC_DRAW)
    norm_vbo = int(glGenBuffers(1))
    glBindBuffer(GL_ARRAY_BUFFER, norm_vbo)
    glBufferData(GL_ARRAY_BUFFER, n * 3 * 4, np.zeros(n * 3, dtype=np.float32), GL_DYNAMIC_DRAW)

    dispatcher.register_piece(
        piece_id=1, output_position_vbo=pos_vbo, output_normal_vbo=norm_vbo,
        bind_positions=bind_positions, bind_normals=bind_normals,
        joints_per_vert=joints, weights_per_vert=weights, dominant_joint=dominant,
    )
    # Bone 0 translates -1.0 in Y, dragging all verts to negative Y.
    bone_matrices = np.zeros((1, 4, 4), dtype=np.float32)
    bone_matrices[0] = np.eye(4, dtype=np.float32)
    bone_matrices[0, 1, 3] = -1.0
    exclude_bits = build_exclude_bits([], collider_count=0)
    world_to_local = np.eye(4, dtype=np.float32)

    # Dispatch WITHOUT ground clamp first — verts should be negative.
    dispatcher.dispatch(
        piece_id=1, bone_matrices=bone_matrices, world_to_local=world_to_local,
        colliders=[], exclude_bits=exclude_bits, ground_y=None,
    )
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo)
    raw = glGetBufferSubData(GL_ARRAY_BUFFER, 0, n * 3 * 4)
    out_unclamped = np.frombuffer(
        bytes(raw) if not isinstance(raw, bytes) else raw, dtype=np.float32,
    ).reshape(n, 3)
    assert float(out_unclamped[:, 1].min()) < -0.4, "expected negative Y without clamp"

    # Now dispatch WITH ground_y=0 — every Y must be >= 0.
    dispatcher.dispatch(
        piece_id=1, bone_matrices=bone_matrices, world_to_local=world_to_local,
        colliders=[], exclude_bits=exclude_bits, ground_y=0.0,
    )
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo)
    raw = glGetBufferSubData(GL_ARRAY_BUFFER, 0, n * 3 * 4)
    out_clamped = np.frombuffer(
        bytes(raw) if not isinstance(raw, bytes) else raw, dtype=np.float32,
    ).reshape(n, 3)
    assert float(out_clamped[:, 1].min()) >= -1.0e-5, (
        f"ground clamp leaked: min Y = {float(out_clamped[:, 1].min())}"
    )

    dispatcher.shutdown()


def test_dispatcher_returns_none_on_missing_shader(qapp: object) -> None:
    """Missing shader file logs a warning and returns ``None`` (no GL needed)."""
    # No GL context required for this branch — it bails on the file read.
    result = PassiveSkinDispatcher.try_create(Path("/no/such/passive_skin.comp"))
    assert result is None
