"""GPU compute-shader LBS + collider push for passive skin-deform cloth pieces.

The CPU LBS path in ``ClothHost._update_skin_targets`` and the eligible-vert
collider projection in ``ClothHost._project_passive_pieces`` together cost
~9 ms / frame on the bundled The Herta body mesh (30 k verts, 354 bones,
7 active colliders). That floor is dominated by numpy doing one
``(N, 4, 4) @ (N, 4, 1)`` matmul per vertex and one boolean mask + index
gather per (piece, collider) — both embarrassingly parallel, both ideal for
a GPU compute shader.

This module owns the whole GPU path:

* :class:`PassiveSkinDispatcher` compiles
  ``shaders/passive_skin/passive_skin_push.comp`` once on construction and
  manages per-piece :class:`PassiveSkinSlot` state.
* :meth:`PassiveSkinDispatcher.register_piece` allocates the static SSBOs
  (bind_positions, joints, weights, dominant_joint) ONCE per piece and binds
  the existing mesh position VBO as the output SSBO — the compute shader
  writes the final mesh-local positions directly into the buffer the
  vertex shader will sample next, no CPU readback.
* :meth:`PassiveSkinDispatcher.dispatch` is called once per frame per piece;
  it uploads the per-frame bone matrices, collider records, and bone-filter
  bitmasks, then issues a ``glDispatchCompute`` with
  ``ceil(N / WORKGROUP_SIZE)`` workgroups. A
  ``glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)`` after dispatch
  guarantees the vertex shader sees the updated positions.

The dispatcher falls back to ``None`` when the active context is < GL 4.3 or
the shader fails to compile (some Intel drivers stub compute), in which case
the caller stays on the CPU path. The fallback is intentional —
``PoseCascade`` must run on macOS legacy GL (caps at 4.1) and on cheap
integrated parts that lie about compute support.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from OpenGL.GL import (
    GL_COMPILE_STATUS,
    GL_COMPUTE_SHADER,
    GL_DYNAMIC_DRAW,
    GL_LINK_STATUS,
    GL_SHADER_STORAGE_BARRIER_BIT,
    GL_SHADER_STORAGE_BUFFER,
    GL_STATIC_DRAW,
    GL_TRUE,
    GL_UNIFORM_BUFFER,
    GL_VERSION,
    GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT,
    glAttachShader,
    glBindBuffer,
    glBindBufferBase,
    glBufferData,
    glBufferSubData,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteBuffers,
    glDeleteProgram,
    glDeleteShader,
    glDispatchCompute,
    glGenBuffers,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetString,
    glGetUniformLocation,
    glLinkProgram,
    glMemoryBarrier,
    glShaderSource,
    glUniform1f,
    glUniform1ui,
    glUniformMatrix4fv,
    glUseProgram,
)

from posecascade.utils.logging import get_logger

_log = get_logger(__name__)

# Must match BONE_WORDS in the shader and _MAX_BONES (384) in the renderer.
_BONE_WORDS = 16
_MAX_BONES = _BONE_WORDS * 32  # 384
_MAX_COLLIDERS = 16
# Matches ``layout(local_size_x = 64)`` in passive_skin_push.comp.
_WORKGROUP_SIZE = 64
# std140 layout: 16-byte header (count + 3 uint pad) + N * 48-byte GpuCollider.
_COLLIDER_BYTES = 48
_UBO_HEADER_BYTES = 16


@dataclass
class PassiveSkinSlot:
    """Per-piece SSBO state for the GPU passive-skin path.

    Holds the GL handle of every static buffer (bind_positions, joints,
    weights, dominant_joint) plus a reference to the mesh's position VBO
    which doubles as the compute shader's output SSBO. The world-to-local
    matrix is captured per-frame from the bound node — no need to cache it.
    """

    piece_id: int
    vert_count: int
    output_position_vbo: int        # mesh position VBO; bound as SSBO 4
    output_normal_vbo: int          # mesh normal VBO; bound as SSBO 8
    bind_positions_ssbo: int
    bind_normals_ssbo: int
    joints_ssbo: int
    weights_ssbo: int
    dominant_joint_ssbo: int
    # The exclude-bits SSBO depends on the active collider list; held here
    # so we can reupload only when the collider set changes between frames.
    exclude_bits_ssbo: int
    # CPU-side cache of the most recently uploaded exclude-bits payload so
    # we can skip the upload when colliders haven't changed.
    last_exclude_signature: bytes = b""
    # Owned static buffer ids — used to free everything on ``release``.
    owned_buffers: tuple[int, ...] = field(default_factory=tuple)


class PassiveSkinDispatcher:
    """GPU LBS + collider-push compute dispatcher.

    Construct with :meth:`try_create` so the caller can react gracefully to
    a missing GL 4.3 context. Once constructed, register each passive-skin
    piece via :meth:`register_piece` and call :meth:`dispatch` per frame.
    """

    def __init__(
        self,
        program: int,
        loc_vert_count: int,
        loc_world_to_local: int,
        loc_ground_enabled: int,
        loc_ground_y: int,
        bone_matrices_ssbo: int,
        colliders_ubo: int,
    ) -> None:
        self._program = program
        self._loc_vert_count = loc_vert_count
        self._loc_world_to_local = loc_world_to_local
        self._loc_ground_enabled = loc_ground_enabled
        self._loc_ground_y = loc_ground_y
        self._bone_matrices_ssbo = bone_matrices_ssbo
        self._colliders_ubo = colliders_ubo
        self._slots: dict[int, PassiveSkinSlot] = {}
        # Pre-allocated CPU scratch — flat collider UBO payload.
        self._ubo_scratch = np.zeros(
            _UBO_HEADER_BYTES + _MAX_COLLIDERS * _COLLIDER_BYTES, dtype=np.uint8,
        )

    @classmethod
    def try_create(cls, shader_path: Path) -> PassiveSkinDispatcher | None:
        """Compile the compute shader and return a dispatcher, or ``None`` on failure.

        Returns ``None`` (and logs a warning) when:
          * the active GL context is older than 4.3
          * the compute shader fails to compile (driver doesn't support
            compute even if it advertises 4.3 — happens on some Intel iGPUs)
          * the shader file cannot be read

        The caller treats ``None`` as "stay on the CPU path"; nothing else
        should fall over.
        """
        if not _gl_supports_compute():
            _log.info("GPU passive-skin path disabled: GL context < 4.3")
            return None
        try:
            source = shader_path.read_text(encoding="utf-8")
        except OSError as err:
            _log.warning("GPU passive-skin path disabled: cannot read %s: %s", shader_path, err)
            return None
        program = _compile_compute(source)
        if program is None:
            return None
        # Pre-allocate the per-frame bone matrices SSBO at MAX size so the
        # buffer is never re-bound mid-dispatch — only the contents change.
        bone_ssbo = int(glGenBuffers(1))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bone_ssbo)
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            _MAX_BONES * 16 * 4,  # 384 mat4s, 16 floats each, 4 bytes each
            None, GL_DYNAMIC_DRAW,
        )
        ubo = int(glGenBuffers(1))
        glBindBuffer(GL_UNIFORM_BUFFER, ubo)
        glBufferData(
            GL_UNIFORM_BUFFER,
            _UBO_HEADER_BYTES + _MAX_COLLIDERS * _COLLIDER_BYTES,
            None, GL_DYNAMIC_DRAW,
        )
        loc_vert_count = int(glGetUniformLocation(program, "u_vertCount"))
        loc_world_to_local = int(glGetUniformLocation(program, "u_worldToLocal"))
        loc_ground_enabled = int(glGetUniformLocation(program, "u_groundEnabled"))
        loc_ground_y = int(glGetUniformLocation(program, "u_groundY"))
        return cls(
            program=program,
            loc_vert_count=loc_vert_count,
            loc_world_to_local=loc_world_to_local,
            loc_ground_enabled=loc_ground_enabled,
            loc_ground_y=loc_ground_y,
            bone_matrices_ssbo=bone_ssbo,
            colliders_ubo=ubo,
        )

    def has_slot(self, piece_id: int) -> bool:
        return piece_id in self._slots

    def register_piece(
        self,
        piece_id: int,
        output_position_vbo: int,
        output_normal_vbo: int,
        bind_positions: NDArray[np.float32],
        bind_normals: NDArray[np.float32],
        joints_per_vert: NDArray[np.int32],
        weights_per_vert: NDArray[np.float32],
        dominant_joint: NDArray[np.int32],
    ) -> None:
        """Allocate per-piece SSBOs and register the slot.

        ``output_position_vbo`` / ``output_normal_vbo`` are the GL handles
        of the mesh's position + normal VBOs. The compute shader writes
        float[3] per vert into each directly, so the existing
        ``glVertexAttribPointer`` bindings on the vertex shader's position
        and normal attributes Just Work on the next draw — no rebind, no
        CPU readback.

        Idempotent on ``piece_id``: re-registering an existing piece
        releases the previous slot first.
        """
        if piece_id in self._slots:
            self.release_piece(piece_id)
        bp = np.ascontiguousarray(bind_positions, dtype=np.float32)
        bn = np.ascontiguousarray(bind_normals, dtype=np.float32)
        # joints / weights arrive as (N, 4); the shader reads them as flat
        # arrays indexed by ``vid * 4 + slot``, so we contiguous-pack here.
        jv = np.ascontiguousarray(joints_per_vert, dtype=np.uint32).reshape(-1)
        wv = np.ascontiguousarray(weights_per_vert, dtype=np.float32).reshape(-1)
        dj = np.ascontiguousarray(dominant_joint, dtype=np.int32)

        bind_ssbo = _alloc_ssbo_with_data(bp, GL_STATIC_DRAW)
        bind_n_ssbo = _alloc_ssbo_with_data(bn, GL_STATIC_DRAW)
        joints_ssbo = _alloc_ssbo_with_data(jv, GL_STATIC_DRAW)
        weights_ssbo = _alloc_ssbo_with_data(wv, GL_STATIC_DRAW)
        dom_ssbo = _alloc_ssbo_with_data(dj, GL_STATIC_DRAW)
        # Exclude-bits is dynamic (depends on active collider set). Allocate
        # at max size so ``glBufferSubData`` updates never realloc.
        excl_ssbo = int(glGenBuffers(1))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, excl_ssbo)
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            _MAX_COLLIDERS * _BONE_WORDS * 4,
            None, GL_DYNAMIC_DRAW,
        )
        self._slots[piece_id] = PassiveSkinSlot(
            piece_id=piece_id,
            vert_count=int(bp.shape[0]),
            output_position_vbo=int(output_position_vbo),
            output_normal_vbo=int(output_normal_vbo),
            bind_positions_ssbo=bind_ssbo,
            bind_normals_ssbo=bind_n_ssbo,
            joints_ssbo=joints_ssbo,
            weights_ssbo=weights_ssbo,
            dominant_joint_ssbo=dom_ssbo,
            exclude_bits_ssbo=excl_ssbo,
            owned_buffers=(
                bind_ssbo, bind_n_ssbo, joints_ssbo, weights_ssbo,
                dom_ssbo, excl_ssbo,
            ),
        )

    def release_piece(self, piece_id: int) -> None:
        """Delete a piece's owned SSBOs and drop the slot."""
        slot = self._slots.pop(piece_id, None)
        if slot is None:
            return
        glDeleteBuffers(len(slot.owned_buffers), list(slot.owned_buffers))

    def dispatch(
        self,
        piece_id: int,
        bone_matrices: NDArray[np.float32],
        world_to_local: NDArray[np.float32],
        colliders: list[object],
        exclude_bits: NDArray[np.uint32],
        ground_y: float | None = None,
    ) -> bool:
        """Run the compute shader for one piece. Returns ``True`` if dispatched.

        ``bone_matrices`` is ``(J, 4, 4)``, the joint world × inverse-bind
        stack already computed by the cloth host (and shared with the
        renderer's per-frame bone matrix cache). ``world_to_local`` is the
        cloth-bound node's inverse world matrix.

        ``colliders`` is the active collider list (sphere or capsule); only
        the first ``MAX_COLLIDERS`` are sent — beyond that the compute path
        silently truncates with a warning so the user knows.

        ``exclude_bits`` is the packed ``(MAX_COLLIDERS, BONE_WORDS)``
        bitmask of joints each collider must NOT push (per-vert bone
        filter), built once on the host side from the collider→joint-set
        map.

        Returns ``False`` if the piece isn't registered — callers fall back
        to the CPU path for that piece in that case.
        """
        slot = self._slots.get(piece_id)
        if slot is None:
            return False
        collider_count = min(len(colliders), _MAX_COLLIDERS)
        if len(colliders) > _MAX_COLLIDERS:
            _log.warning(
                "passive-skin GPU dispatch: %d colliders > MAX_COLLIDERS=%d, truncating",
                len(colliders), _MAX_COLLIDERS,
            )

        glUseProgram(self._program)
        # Per-frame buffer uploads — bone matrices, colliders, exclude bits.
        self._upload_bone_matrices(bone_matrices)
        self._upload_colliders(colliders, collider_count)
        self._upload_exclude_bits(slot, exclude_bits, collider_count)

        # Uniforms: vertex count + world→local matrix.
        glUniform1ui(self._loc_vert_count, slot.vert_count)
        # GLSL is column-major; numpy mat4 is row-major. Pass transpose=GL_TRUE.
        contig = np.ascontiguousarray(world_to_local, dtype=np.float32)
        glUniformMatrix4fv(self._loc_world_to_local, 1, GL_TRUE, contig)
        # Ground clamp gate. Both uniforms always written so a previous
        # frame's enabled state can't leak in when the host turns ground off.
        if ground_y is None or self._loc_ground_enabled < 0 or self._loc_ground_y < 0:
            glUniform1ui(self._loc_ground_enabled, 0)
            glUniform1f(self._loc_ground_y, 0.0)
        else:
            glUniform1ui(self._loc_ground_enabled, 1)
            glUniform1f(self._loc_ground_y, float(ground_y))

        # SSBO bindings. Outputs (bindings 4, 8) reuse the mesh's existing
        # position + normal VBOs; the compute shader writes to them in
        # place so the very next draw call sees the deformed values.
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, slot.bind_positions_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, slot.joints_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, slot.weights_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self._bone_matrices_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, slot.output_position_vbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, slot.dominant_joint_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, slot.exclude_bits_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, slot.bind_normals_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, slot.output_normal_vbo)
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, self._colliders_ubo)

        groups = (slot.vert_count + _WORKGROUP_SIZE - 1) // _WORKGROUP_SIZE
        glDispatchCompute(groups, 1, 1)
        # The mesh's position VBO is read by the vertex shader on the next
        # draw call — we need both the vertex-attrib barrier (for the draw)
        # and the SSBO barrier (in case any same-frame readback or compute
        # follow-up touches it).
        glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT)
        glUseProgram(0)
        return True

    def shutdown(self) -> None:
        """Delete every owned GL object. Idempotent."""
        for piece_id in list(self._slots):
            self.release_piece(piece_id)
        if self._bone_matrices_ssbo:
            glDeleteBuffers(1, [self._bone_matrices_ssbo])
            self._bone_matrices_ssbo = 0
        if self._colliders_ubo:
            glDeleteBuffers(1, [self._colliders_ubo])
            self._colliders_ubo = 0
        if self._program:
            glDeleteProgram(self._program)
            self._program = 0

    def _upload_bone_matrices(self, bone_matrices: NDArray[np.float32]) -> None:
        """Stream ``(J, 4, 4)`` joint matrices into the shared SSBO."""
        # GLSL std430 mat4 layout is column-major; numpy stores row-major.
        # Transpose per-matrix so the shader reads correct columns.
        transposed = np.transpose(bone_matrices, (0, 2, 1))
        contig = np.ascontiguousarray(transposed, dtype=np.float32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._bone_matrices_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, contig.nbytes, contig)

    def _upload_colliders(self, colliders: list[object], count: int) -> None:
        """Pack the active colliders into the std140 UBO payload."""
        from posecascade.animation.cloth import (  # noqa: PLC0415 — avoid heavy import at module top
            CapsuleCollider,
            SphereCollider,
        )

        payload = self._ubo_scratch
        payload.fill(0)
        # std140 header: uint count + 3 uint pad. We write only the first
        # uint; the rest stays zero from the fill().
        np.frombuffer(payload, dtype=np.uint32, count=1)[0] = np.uint32(count)
        # Each GpuCollider occupies 48 bytes (3 * vec4). data1 = (x, y, z, r);
        # data2 = (x2, y2, z2, kind); extras = (skin_offset, 0, 0, 0).
        offset_floats = _UBO_HEADER_BYTES // 4
        flat = np.frombuffer(payload, dtype=np.float32)
        for i in range(count):
            c = colliders[i]
            base = offset_floats + i * (_COLLIDER_BYTES // 4)
            if isinstance(c, SphereCollider):
                flat[base + 0] = c.center[0]
                flat[base + 1] = c.center[1]
                flat[base + 2] = c.center[2]
                flat[base + 3] = c.radius
                flat[base + 7] = 0.0           # kind = sphere
                flat[base + 8] = c.skin_offset
            elif isinstance(c, CapsuleCollider):
                flat[base + 0] = c.a[0]
                flat[base + 1] = c.a[1]
                flat[base + 2] = c.a[2]
                flat[base + 3] = c.radius
                flat[base + 4] = c.b[0]
                flat[base + 5] = c.b[1]
                flat[base + 6] = c.b[2]
                flat[base + 7] = 1.0           # kind = capsule
                flat[base + 8] = c.skin_offset
            else:
                _log.warning(
                    "passive-skin GPU dispatch: unsupported collider type %s — skipped",
                    type(c).__name__,
                )
        glBindBuffer(GL_UNIFORM_BUFFER, self._colliders_ubo)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, payload.nbytes, payload)

    def _upload_exclude_bits(
        self,
        slot: PassiveSkinSlot,
        exclude_bits: NDArray[np.uint32],
        collider_count: int,
    ) -> None:
        """Stream the (collider × bone-word) exclude bitmask into the SSBO.

        Skips the upload when the payload matches the last frame's — the
        collider→joint-filter mapping is set at scene-build time and only
        changes when the user adds / removes a collider, so most frames hit
        the cached fast path.
        """
        contig = np.ascontiguousarray(
            exclude_bits[:collider_count], dtype=np.uint32,
        )
        signature = contig.tobytes()
        if signature == slot.last_exclude_signature:
            return
        slot.last_exclude_signature = signature
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, slot.exclude_bits_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, contig.nbytes, contig)


def build_exclude_bits(
    collider_filters: list[frozenset[int]], collider_count: int,
) -> NDArray[np.uint32]:
    """Pack per-collider excluded-joint sets into a ``(MAX_COLLIDERS, BONE_WORDS)`` bitmask.

    Each ``frozenset[int]`` is the joint-index set to NOT push for that
    collider. Joints beyond ``_MAX_BONES`` are silently dropped — anything
    past index 384 cannot be expressed in the fixed-width bitfield and
    would over-run the shader's array bounds.
    """
    out = np.zeros((_MAX_COLLIDERS, _BONE_WORDS), dtype=np.uint32)
    capped = min(collider_count, _MAX_COLLIDERS)
    for i in range(capped):
        excluded = collider_filters[i] if i < len(collider_filters) else frozenset()
        for joint_idx in excluded:
            if 0 <= joint_idx < _MAX_BONES:
                word = joint_idx >> 5
                bit = joint_idx & 31
                out[i, word] |= np.uint32(1 << bit)
    return out


def _gl_supports_compute() -> bool:
    """True iff the current GL context is at least 4.3 (where compute became core)."""
    try:
        raw = glGetString(GL_VERSION)
    except Exception:                                                   # noqa: BLE001
        return False
    if raw is None:
        return False
    text = bytes(raw).decode("ascii", errors="replace") if isinstance(raw, bytes) else str(raw)
    head = text.split()[0] if text else ""
    parts = head.split(".")
    # GL version strings always parse as ``MAJOR.MINOR[.PATCH] [vendor]`` —
    # at minimum we need two dotted components to read major/minor.
    min_components = 2
    if len(parts) < min_components:
        return False
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except ValueError:
        return False
    return (major, minor) >= (4, 3)


def _compile_compute(source: str) -> int | None:
    """Compile + link a compute-only program. Returns ``None`` on failure (with log)."""
    shader = int(glCreateShader(GL_COMPUTE_SHADER))
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = _decode(glGetShaderInfoLog(shader))
        _log.warning("passive-skin compute shader compile failed: %s", log)
        glDeleteShader(shader)
        return None
    program = int(glCreateProgram())
    glAttachShader(program, shader)
    glLinkProgram(program)
    glDeleteShader(shader)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = _decode(glGetProgramInfoLog(program))
        _log.warning("passive-skin compute program link failed: %s", log)
        glDeleteProgram(program)
        return None
    return program


def _alloc_ssbo_with_data(data: np.ndarray, usage: int) -> int:
    """Generate an SSBO and upload ``data`` with the given usage hint."""
    contig = np.ascontiguousarray(data)
    ssbo = int(glGenBuffers(1))
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, contig.nbytes, contig, usage)
    return ssbo


def _decode(log: object) -> str:
    if isinstance(log, bytes):
        return log.decode("utf-8", errors="replace").strip()
    return str(log).strip()


# ``ctypes`` is imported to keep the buffer-binding pattern symmetric with
# ``mesh_uploader.py`` and to make it trivial to extend with mapped-buffer
# fast paths if the per-frame SubData uploads ever become hot.
_ = ctypes
