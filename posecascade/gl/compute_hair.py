"""GPU compute-shader spring-chain physics dispatcher.

Mirrors :class:`posecascade.gl.compute_skin.PassiveSkinDispatcher`'s
pattern: compile + link the ``shaders/hair/hair_spring.comp`` shader
once at construction, allocate per-chain SSBOs lazily on the first
``register_chains`` call, and dispatch + read-back once per frame from
the render thread.

Falls back to ``None`` on GL < 4.3, so the calling renderer stays on
the CPU spring solver when compute isn't available.
"""
from __future__ import annotations

import logging
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
    GL_UNIFORM_BUFFER,
    GL_VERSION,
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
    glGetBufferSubData,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetString,
    glLinkProgram,
    glMemoryBarrier,
    glShaderSource,
    glUseProgram,
)

_log = logging.getLogger(__name__)

# Layout constants — must match ``shaders/hair/hair_spring.comp``.
SSBO_JOINT_STATE = 0
SSBO_CHAIN_INFO = 1
SSBO_ANCHOR = 2
SSBO_SPHERES = 3
SSBO_CAPSULES = 4
UBO_SIM_CONSTANTS = 0

# 8 vec4s per Joint, std430 = 128 bytes (32 floats). Joints carry world
# state + rest pose + rest-in-anchor (for cone clamp); the 8th vec4 is
# reserved padding so the per-joint stride is a power of 2 — empirically
# avoids a multi-dim-upload bug on AMD drivers where odd-stride 2D
# numpy arrays uploaded via PyOpenGL's glBufferData ended up writing
# garbage values into the SSBO past the first row.
JOINT_FLOATS = 32
JOINT_STRIDE = JOINT_FLOATS * 4

# 12 floats per ChainInfo (start, count, anchor_idx, stiffness, damping,
# inertia, max_swing_rad, pad, gravity_override.xyzw).
CHAIN_INFO_FLOATS = 12

# 8 floats per AnchorXform (rotation vec4 + position vec4).
ANCHOR_FLOATS = 8

# 8 floats per SphereC / CapsuleC (two vec4s each).
SPHERE_FLOATS = 8
CAPSULE_FLOATS = 8

# 12 floats for SimConstants UBO (std140: vec3s padded to vec4).
SIM_CONSTANTS_FLOATS = 12


@dataclass
class HairChainSlot:
    """Per-chain bookkeeping: where this chain's joints live in the
    joint-state SSBO, and references to the CPU-side SpringJoint
    objects so the readback step can write rotations back into the
    scene graph."""

    chain_name: str
    joint_start: int
    joint_count: int
    anchor_idx: int
    joints_cpu: tuple[object, ...] = field(default_factory=tuple)


@dataclass
class HairComputeDispatcher:
    """Owns the compute program + SSBO arena + per-frame upload buffers.

    Lifecycle:
      1. ``try_create(shader_path)`` compiles the program; returns ``None``
         on GL < 4.3 / driver bug / link failure.
      2. ``register_chains(chains)`` packs initial joint state, chain info,
         anchor transforms into SSBOs. Idempotent: re-running rebuilds
         from scratch.
      3. ``dispatch(dt, gravity, wind, anchor_xforms, spheres, capsules)``
         uploads per-frame state, runs ``glDispatchCompute``, blocks on a
         memory barrier, reads back rotations into ``joints_cpu``.
    """

    program: int
    shader_source_hash: str
    joint_state_ssbo: int = 0
    chain_info_ssbo: int = 0
    anchor_ssbo: int = 0
    sphere_ssbo: int = 0
    capsule_ssbo: int = 0
    sim_constants_ubo: int = 0
    joint_capacity: int = 0
    chain_capacity: int = 0
    sphere_capacity: int = 0
    capsule_capacity: int = 0
    slots: list[HairChainSlot] = field(default_factory=list)
    # Per-buffer capacity tracking for glBufferSubData fast-path. When
    # the new upload fits in the existing allocation, we update in
    # place; only when it grows past capacity do we re-allocate via
    # glBufferData. Cuts per-frame upload cost by ~10× on warm buffers.
    _anchor_cap: int = 0
    _sphere_cap: int = 0
    _capsule_cap: int = 0

    @classmethod
    def try_create(cls, shader_path: Path) -> HairComputeDispatcher | None:
        if not _has_compute_support():
            return None
        try:
            source = shader_path.read_text(encoding="utf-8")
        except OSError as err:
            _log.warning("hair compute shader read failed: %s", err)
            return None
        program = _compile_compute(source)
        if program is None:
            return None
        instance = cls(
            program=program,
            shader_source_hash=str(hash(source)),
        )
        instance._allocate_static_buffers()                                  # noqa: SLF001
        return instance

    def _allocate_static_buffers(self) -> None:
        """Reserve the SSBO + UBO ids that persist for the dispatcher's
        whole lifetime. Buffer storage gets (re)allocated whenever the
        chain set's size grows."""
        ids = glGenBuffers(6)
        # On AMD drivers ``glGenBuffers(n)`` returns a numpy array; on
        # Mesa it returns a list. Normalise to plain Python ints.
        ids = [int(x) for x in (ids if hasattr(ids, "__iter__") else [ids])]
        (
            self.joint_state_ssbo,
            self.chain_info_ssbo,
            self.anchor_ssbo,
            self.sphere_ssbo,
            self.capsule_ssbo,
            self.sim_constants_ubo,
        ) = ids

    def destroy(self) -> None:
        if self.joint_state_ssbo:
            glDeleteBuffers(6, [
                self.joint_state_ssbo,
                self.chain_info_ssbo,
                self.anchor_ssbo,
                self.sphere_ssbo,
                self.capsule_ssbo,
                self.sim_constants_ubo,
            ])
            self.joint_state_ssbo = 0
        if self.program:
            glDeleteProgram(self.program)
            self.program = 0
        self.slots.clear()

    def register_chains(
        self,
        chains: list[object],
    ) -> None:
        """Pack ``chains`` (list of SpringChain) into the joint + chain
        SSBOs.

        Joint state is computed from each ``SpringJoint``'s rest pose
        snapshot — ``world_rotation`` / ``world_position`` / ``angular_velocity``
        seed from whatever the CPU spring solver has stored (typically
        identity rotation + chain anchor position for an unevaluated
        chain, or the last-frame state for a hand-off from CPU).
        """
        if not chains:
            self.slots.clear()
            self.joint_capacity = 0
            self.chain_capacity = 0
            return
        slots: list[HairChainSlot] = []
        joint_blocks: list[NDArray[np.float32]] = []
        chain_info_rows: list[NDArray[np.float32]] = []
        cursor = 0
        for chain_idx, chain in enumerate(chains):
            joints = tuple(chain.joints)
            n = len(joints)
            block = _pack_chain_joints(joints)
            joint_blocks.append(block)
            chain_info_rows.append(_pack_chain_info(
                joint_start=cursor,
                joint_count=n,
                anchor_idx=chain_idx,
                stiffness=float(chain.stiffness),
                damping=float(chain.damping),
                inertia=float(joints[0].inertia) if joints else 1.0,
                max_swing_rad=float(chain.max_swing_rad),
                gravity_override=getattr(chain, "gravity_override", None),
                static_drape=bool(getattr(chain, "static_drape", False)),
            ))
            slots.append(HairChainSlot(
                chain_name=str(chain.name),
                joint_start=cursor,
                joint_count=n,
                anchor_idx=chain_idx,
                joints_cpu=joints,
            ))
            cursor += n
        joint_state = np.concatenate(joint_blocks, axis=0).astype(np.float32)
        chain_info = np.stack(chain_info_rows, axis=0).astype(np.float32)

        # Flatten to 1-D before uploading — PyOpenGL's glBufferData
        # treats multi-dim arrays inconsistently across versions and
        # driver vendors; a contiguous 1-D float32 buffer is the one
        # form every backend agrees on.
        joint_flat = np.ascontiguousarray(joint_state.reshape(-1))
        chain_flat = np.ascontiguousarray(chain_info.reshape(-1))

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.joint_state_ssbo)
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            joint_flat.nbytes,
            joint_flat,
            GL_DYNAMIC_DRAW,
        )
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.chain_info_ssbo)
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            chain_flat.nbytes,
            chain_flat,
            GL_STATIC_DRAW,
        )
        self.joint_capacity = cursor
        self.chain_capacity = len(chains)
        self.slots = slots

    def dispatch(
        self,
        dt: float,
        gravity: NDArray[np.float32],
        wind: NDArray[np.float32],
        anchor_xforms: NDArray[np.float32],
        spheres: NDArray[np.float32],
        capsules: NDArray[np.float32],
    ) -> None:
        """Run the compute shader for the registered chains.

        Parameters
        ----------
        dt
            Substep duration in seconds. Matches the CPU solver's
            ``_DEFAULT_FIXED_DT`` (~8.3 ms at 120 Hz).
        gravity, wind
            World-space force vectors. Shape ``(3,)`` float32.
        anchor_xforms
            ``(chain_count, ANCHOR_FLOATS)`` per-chain rotation + position
            of the anchor bone at the current frame.
        spheres
            ``(N_sphere, SPHERE_FLOATS)`` per-collider center+radius+offset.
        capsules
            ``(N_capsule, CAPSULE_FLOATS)`` per-collider a/b/radius/offset.
        """
        if not self.slots:
            return
        # Per-frame upload path: ``glBufferSubData`` reuses the already-
        # allocated storage (avoids the GPU driver reallocating + syncing
        # every frame on glBufferData). Only re-allocate via glBufferData
        # when the upload's byte size exceeds the current capacity.
        self._upload_per_frame(self.anchor_ssbo, anchor_xforms, "_anchor_cap")
        if len(spheres):
            self._upload_per_frame(
                self.sphere_ssbo, spheres, "_sphere_cap",
            )
        else:
            zero = np.zeros(SPHERE_FLOATS, dtype=np.float32)
            self._upload_per_frame(
                self.sphere_ssbo, zero, "_sphere_cap",
            )
        if len(capsules):
            self._upload_per_frame(
                self.capsule_ssbo, capsules, "_capsule_cap",
            )
        else:
            zero = np.zeros(CAPSULE_FLOATS, dtype=np.float32)
            self._upload_per_frame(
                self.capsule_ssbo, zero, "_capsule_cap",
            )
        # Sim constants UBO — fixed 12 floats, fits in one re-allocation
        # so plain glBufferData is fine (cost is sub-microsecond).
        sim = _pack_sim_constants(dt, gravity, wind, len(spheres), len(capsules))
        glBindBuffer(GL_UNIFORM_BUFFER, self.sim_constants_ubo)
        glBufferData(
            GL_UNIFORM_BUFFER,
            sim.nbytes,
            np.ascontiguousarray(sim),
            GL_DYNAMIC_DRAW,
        )

        glUseProgram(self.program)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBO_JOINT_STATE, self.joint_state_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBO_CHAIN_INFO, self.chain_info_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBO_ANCHOR, self.anchor_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBO_SPHERES, self.sphere_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBO_CAPSULES, self.capsule_ssbo)
        glBindBufferBase(GL_UNIFORM_BUFFER, UBO_SIM_CONSTANTS, self.sim_constants_ubo)
        glDispatchCompute(len(self.slots), 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glUseProgram(0)

    def _upload_per_frame(
        self,
        ssbo: int,
        data: NDArray[np.float32],
        cap_attr: str,
    ) -> None:
        """Upload ``data`` to ``ssbo``, reusing existing storage when
        the new payload fits. Re-allocates with ``glBufferData`` only
        when capacity is exceeded.
        """
        contig = np.ascontiguousarray(data)
        nbytes = int(contig.nbytes)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        current_cap = int(getattr(self, cap_attr))
        if nbytes <= current_cap:
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, nbytes, contig)
            return
        # Grow with 1.5× headroom so subsequent frames hit the fast path.
        new_cap = max(int(nbytes * 1.5), nbytes)
        glBufferData(
            GL_SHADER_STORAGE_BUFFER, new_cap, None, GL_DYNAMIC_DRAW,
        )
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, nbytes, contig)
        setattr(self, cap_attr, new_cap)

    def readback_to_cpu(self) -> NDArray[np.float32]:
        """Read the entire joint-state SSBO back into a numpy array.

        ``(joint_capacity, JOINT_FLOATS)``. Caller is responsible for
        applying the world rotations back into the CPU SpringJoint
        objects.

        Uses a ctypes buffer as the readback target then copies into
        numpy. PyOpenGL's ``glGetBufferSubData`` does NOT correctly
        write into a numpy array argument on AMD drivers — the call
        appears to succeed but the array stays uninitialised — so we
        fall back to the ctypes path which is the version every
        backend handles consistently.
        """
        import ctypes  # noqa: PLC0415

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.joint_state_ssbo)
        n_floats = self.joint_capacity * JOINT_FLOATS
        ctype_buf = (ctypes.c_float * n_floats)()
        glGetBufferSubData(
            GL_SHADER_STORAGE_BUFFER, 0, n_floats * 4, ctype_buf,
        )
        buf = np.frombuffer(ctype_buf, dtype=np.float32).copy()
        return buf.reshape(self.joint_capacity, JOINT_FLOATS)


# --- packing helpers --------------------------------------------------------


def _pack_chain_joints(joints: tuple[object, ...]) -> NDArray[np.float32]:
    """Pack one chain's joints into ``(N, JOINT_FLOATS)`` float32.

    Slots 28..31 are reserved padding (see :data:`JOINT_FLOATS` rationale).
    """
    n = len(joints)
    out = np.zeros((n, JOINT_FLOATS), dtype=np.float32)
    for i, j in enumerate(joints):
        out[i, 0:4]   = j.world_rotation
        out[i, 4:7]   = j.world_position
        out[i, 8:11]  = j.angular_velocity
        out[i, 12:15] = j.rest_local_position
        out[i, 16:20] = j.rest_local_rotation
        out[i, 20:23] = j.bone_vector_local
        out[i, 23]    = float(np.linalg.norm(j.bone_vector_local))
        out[i, 24:28] = j.rest_in_anchor_frame
    return out


def _pack_chain_info(
    joint_start: int,
    joint_count: int,
    anchor_idx: int,
    stiffness: float,
    damping: float,
    inertia: float,
    max_swing_rad: float,
    gravity_override: NDArray[np.float32] | None = None,
    static_drape: bool = False,
) -> NDArray[np.float32]:
    row = np.zeros(CHAIN_INFO_FLOATS, dtype=np.float32)
    # std430 doesn't enforce uint alignment for our use; reinterpret
    # the integer fields as float-bit patterns the shader can read
    # via ``floatBitsToUint`` (the shader code uses ``uint`` fields
    # directly so we pack them as raw uint32 → reinterpret as float32).
    row[0] = np.uint32(joint_start).view(np.float32)
    row[1] = np.uint32(joint_count).view(np.float32)
    row[2] = np.uint32(anchor_idx).view(np.float32)
    row[3] = stiffness
    row[4] = damping
    row[5] = inertia
    row[6] = max_swing_rad
    row[7] = 1.0 if static_drape else 0.0
    if gravity_override is not None:
        row[8:11] = np.asarray(gravity_override, dtype=np.float32)
        row[11] = 1.0  # enabled flag
    # else row[8..11] stay zero → flag 0 → shader uses global gravity.
    return row


def pack_anchor_xforms(
    anchors: list[tuple[NDArray[np.float32], NDArray[np.float32]]],
) -> NDArray[np.float32]:
    """Pack ``[(rotation_quat, position_vec3), ...]`` into a flat
    ``(N, 8)`` float32 array matching the shader's ``AnchorXform``."""
    n = len(anchors)
    out = np.zeros((n, ANCHOR_FLOATS), dtype=np.float32)
    for i, (rot, pos) in enumerate(anchors):
        out[i, 0:4] = rot
        out[i, 4:7] = pos
    return out


def pack_spheres(
    spheres: list[tuple[NDArray[np.float32], float, float]],
) -> NDArray[np.float32]:
    """Pack ``[(center, radius, skin_offset), ...]`` into ``(N, 8)``."""
    n = len(spheres)
    out = np.zeros((n, SPHERE_FLOATS), dtype=np.float32)
    for i, (center, radius, skin_offset) in enumerate(spheres):
        out[i, 0:3] = center
        out[i, 3] = radius
        out[i, 4] = skin_offset
    return out


def pack_capsules(
    capsules: list[tuple[NDArray[np.float32], NDArray[np.float32], float, float]],
) -> NDArray[np.float32]:
    """Pack ``[(a, b, radius, skin_offset), ...]`` into ``(N, 8)``."""
    n = len(capsules)
    out = np.zeros((n, CAPSULE_FLOATS), dtype=np.float32)
    for i, (a, b, radius, skin_offset) in enumerate(capsules):
        out[i, 0:3] = a
        out[i, 3] = radius
        out[i, 4:7] = b
        out[i, 7] = skin_offset
    return out


def _pack_sim_constants(
    dt: float,
    gravity: NDArray[np.float32],
    wind: NDArray[np.float32],
    sphere_count: int,
    capsule_count: int,
) -> NDArray[np.float32]:
    buf = np.zeros(SIM_CONSTANTS_FLOATS, dtype=np.float32)
    buf[0] = float(dt)
    buf[4:7] = gravity[:3]
    buf[8:11] = wind[:3]
    # sphere_count + capsule_count packed as floats (reinterpret in shader
    # via ``floatBitsToUint`` if needed; the shader uses ``uint`` fields,
    # so write the bit pattern directly).
    buf[7] = np.uint32(sphere_count).view(np.float32)
    buf[11] = np.uint32(capsule_count).view(np.float32)
    return buf


def _has_compute_support() -> bool:
    try:
        raw = glGetString(GL_VERSION)
    except Exception:                                                        # noqa: BLE001
        return False
    if raw is None:
        return False
    text = bytes(raw).decode("ascii", "replace") if isinstance(raw, bytes) else str(raw)
    parts = text.split()[0].split(".") if text else []
    if len(parts) < 2:                                                       # noqa: PLR2004
        return False
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except ValueError:
        return False
    return (major, minor) >= (4, 3)


def _compile_compute(source: str) -> int | None:
    shader = int(glCreateShader(GL_COMPUTE_SHADER))
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = _decode(glGetShaderInfoLog(shader))
        _log.warning("hair compute shader compile failed: %s", log)
        glDeleteShader(shader)
        return None
    program = int(glCreateProgram())
    glAttachShader(program, shader)
    glLinkProgram(program)
    glDeleteShader(shader)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = _decode(glGetProgramInfoLog(program))
        _log.warning("hair compute program link failed: %s", log)
        glDeleteProgram(program)
        return None
    return program


def _decode(log: object) -> str:
    if isinstance(log, bytes):
        return log.decode("utf-8", "replace").strip()
    return str(log).strip()


__all__ = [
    "ANCHOR_FLOATS",
    "CAPSULE_FLOATS",
    "CHAIN_INFO_FLOATS",
    "HairChainSlot",
    "HairComputeDispatcher",
    "JOINT_FLOATS",
    "JOINT_STRIDE",
    "SIM_CONSTANTS_FLOATS",
    "SPHERE_FLOATS",
    "pack_anchor_xforms",
    "pack_capsules",
    "pack_spheres",
]
