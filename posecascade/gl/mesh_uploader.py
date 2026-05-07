"""Upload :class:`~posecascade.assets.types.Mesh` into a VAO + VBO + EBO bundle."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_DYNAMIC_DRAW,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_STATIC_DRAW,
    GL_UNSIGNED_SHORT,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glBufferSubData,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glVertexAttribIPointer,
    glVertexAttribPointer,
)

from posecascade.assets.types import Mesh
from posecascade.gl.uniforms import (
    A_JOINTS_0,
    A_NORMAL,
    A_POSITION,
    A_TEXCOORD_0,
    A_WEIGHTS_0,
)
from posecascade.render.material import MMDMaterial

# Index of the normal VBO inside :attr:`GLMesh.vbos` when uploaded with normals.
_NORMAL_VBO_INDEX = 1


@dataclass
class GLMesh:
    """GPU-resident mesh handle. Owns its VAO, attribute VBOs, and EBO."""

    vao: int
    vbos: tuple[int, ...]
    ebo: int
    index_count: int
    aabb_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    aabb_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_color: tuple[float, float, float, float] | None = None
    base_color_texture_id: int | None = None  # GL texture id; None means use white fallback
    # MMD-specific extras: present iff the importer attached an MMDMaterial.
    # The renderer routes such meshes through the toon + outline passes.
    mmd_material: MMDMaterial | None = None
    sphere_texture_id: int | None = None
    toon_texture_id: int | None = None
    # VBO-slot indices the upload step records explicitly (otherwise downstream
    # streamers must guess at where each attribute lives). ``None`` means the
    # mesh was uploaded without that channel.
    position_vbo_index: int = 0
    normal_vbo_index: int | None = None
    texcoord_vbo_index: int | None = None

    def delete(self) -> None:
        glDeleteVertexArrays(1, [self.vao])
        if self.vbos:
            glDeleteBuffers(len(self.vbos), list(self.vbos))
        glDeleteBuffers(1, [self.ebo])


def upload_mesh(mesh: Mesh) -> GLMesh:
    """Allocate GL buffers and upload ``mesh`` data. Caller must hold the GL context."""
    vao = int(glGenVertexArrays(1))
    glBindVertexArray(vao)

    vbos: list[int] = []
    vbos.append(_upload_attribute(mesh.positions, A_POSITION, components=3))
    normal_index: int | None = None
    if mesh.normals is not None:
        normal_index = len(vbos)
        vbos.append(_upload_attribute(mesh.normals, A_NORMAL, components=3))
    texcoord_index: int | None = None
    if mesh.texcoords_0 is not None:
        texcoord_index = len(vbos)
        vbos.append(_upload_attribute(mesh.texcoords_0, A_TEXCOORD_0, components=2))
    if mesh.joints_0 is not None:
        vbos.append(_upload_int_attribute(mesh.joints_0, A_JOINTS_0, components=4))
    if mesh.weights_0 is not None:
        vbos.append(_upload_attribute(mesh.weights_0, A_WEIGHTS_0, components=4))

    ebo = int(glGenBuffers(1))
    indices = np.ascontiguousarray(mesh.indices, dtype=np.uint32)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glBindVertexArray(0)
    aabb_min, aabb_max = _mesh_local_aabb(mesh.positions)
    return GLMesh(
        vao=vao,
        vbos=tuple(vbos),
        ebo=ebo,
        index_count=int(indices.size),
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        base_color=mesh.base_color,
        mmd_material=mesh.mmd_material,
        normal_vbo_index=normal_index,
        texcoord_vbo_index=texcoord_index,
    )


def _mesh_local_aabb(
    positions: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute the axis-aligned bounding box of mesh-local vertex positions."""
    if positions.size == 0:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    arr = np.asarray(positions, dtype=np.float32)
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    return (
        (float(mn[0]), float(mn[1]), float(mn[2])),
        (float(mx[0]), float(mx[1]), float(mx[2])),
    )


def _upload_attribute(data: np.ndarray, location: int, components: int) -> int:
    contig = np.ascontiguousarray(data, dtype=np.float32)
    vbo = int(glGenBuffers(1))
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, contig.nbytes, contig, GL_STATIC_DRAW)
    glEnableVertexAttribArray(location)
    glVertexAttribPointer(
        location,
        components,
        GL_FLOAT,
        GL_FALSE,
        components * ctypes.sizeof(ctypes.c_float),
        ctypes.c_void_p(0),
    )
    return vbo


def reupload_position_vbo(gl_mesh: GLMesh, positions: np.ndarray) -> None:
    """Stream new vertex positions into the position VBO (``vbos[0]``).

    Caller must hold the GL context. Uses ``glBufferSubData`` so the buffer's
    storage hint is preserved — works whether the buffer was created with
    ``GL_STATIC_DRAW`` or ``GL_DYNAMIC_DRAW``. If the new data is larger than
    the existing allocation, the buffer is reallocated with ``GL_DYNAMIC_DRAW``.
    """
    contig = np.ascontiguousarray(positions, dtype=np.float32)
    glBindBuffer(GL_ARRAY_BUFFER, gl_mesh.vbos[0])
    glBufferData(GL_ARRAY_BUFFER, contig.nbytes, contig, GL_DYNAMIC_DRAW)


def reupload_normal_vbo(gl_mesh: GLMesh, normals: np.ndarray) -> None:
    """Stream new vertex normals into the normal VBO if it exists.

    No-op when the mesh was uploaded without normals.
    """
    if gl_mesh.normal_vbo_index is None:
        return
    contig = np.ascontiguousarray(normals, dtype=np.float32)
    glBindBuffer(GL_ARRAY_BUFFER, gl_mesh.vbos[gl_mesh.normal_vbo_index])
    glBufferSubData(GL_ARRAY_BUFFER, 0, contig.nbytes, contig)


def reupload_texcoords_vbo(gl_mesh: GLMesh, texcoords: np.ndarray) -> None:
    """Stream new primary-UV coordinates into the texcoord VBO if it exists."""
    if gl_mesh.texcoord_vbo_index is None:
        return
    contig = np.ascontiguousarray(texcoords, dtype=np.float32)
    glBindBuffer(GL_ARRAY_BUFFER, gl_mesh.vbos[gl_mesh.texcoord_vbo_index])
    glBufferData(GL_ARRAY_BUFFER, contig.nbytes, contig, GL_DYNAMIC_DRAW)


def _upload_int_attribute(data: np.ndarray, location: int, components: int) -> int:
    """Upload an unsigned integer vertex attribute (e.g. joint indices) as ``uvec4``.

    Uses :func:`glVertexAttribIPointer` so the shader receives raw integers
    rather than GL converting to float.
    """
    contig = np.ascontiguousarray(data, dtype=np.uint16)
    vbo = int(glGenBuffers(1))
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, contig.nbytes, contig, GL_STATIC_DRAW)
    glEnableVertexAttribArray(location)
    glVertexAttribIPointer(
        location,
        components,
        GL_UNSIGNED_SHORT,
        components * ctypes.sizeof(ctypes.c_uint16),
        ctypes.c_void_p(0),
    )
    return vbo
