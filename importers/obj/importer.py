"""Wavefront OBJ importer.

Reads ``.obj`` text files into :class:`~posecascade.assets.types.ImportedScene`.
Supports positions, normals, UVs, and triangulated faces (fan triangulation
for n-gons). ``mtllib`` references resolve via
:func:`posecascade.assets.path_safety.resolve_safe`; the actual MTL parser is
deferred — material names are recorded on the mesh so a later pass can bind
them once a material library is wired up.

Skeleton-mode: animations, skinning, and per-face material splits are not
emitted yet. The vertex stream is unique-keyed on ``(v_idx, vt_idx, vn_idx)``
so duplicate corners share an output index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from posecascade.assets.path_safety import resolve_safe
from posecascade.assets.types import ImportedScene, Mesh
from posecascade.errors import MalformedAssetError
from posecascade.scene.component import MeshRefComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene

_VEC3_MIN_COMPONENTS = 3
_VEC2_MIN_COMPONENTS = 2
_FACE_MIN_CORNERS = 3


@dataclass
class _ParseState:
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    normals: list[tuple[float, float, float]] = field(default_factory=list)
    texcoords: list[tuple[float, float]] = field(default_factory=list)
    faces: list[list[tuple[int, int, int]]] = field(default_factory=list)
    object_name: str = "obj"


class ObjImporter:
    """Loads ``.obj`` files into :class:`ImportedScene`."""

    supported_extensions: tuple[str, ...] = (".obj",)

    def load(self, path: Path) -> ImportedScene:
        path = path.resolve()
        if not path.is_file():
            raise MalformedAssetError(f"OBJ file not found: {path}")
        state = _parse_obj(path)
        mesh = _build_mesh(state)
        node = Node(name=state.object_name)
        node.add_component(MeshRefComponent(mesh_indices=(0,)))
        scene = Scene(name=path.stem)
        scene.root.add_child(node)
        return ImportedScene(meshes=(mesh,), scene=scene)


def _parse_obj(path: Path) -> _ParseState:
    state = _ParseState(object_name=path.stem)
    asset_root = path.parent
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            _dispatch_line(line, state, asset_root)
    return state


def _dispatch_line(line: str, state: _ParseState, asset_root: Path) -> None:
    keyword, _, rest = line.partition(" ")
    if keyword == "v":
        state.positions.append(_parse_vec3(rest))
    elif keyword == "vn":
        state.normals.append(_parse_vec3(rest))
    elif keyword == "vt":
        state.texcoords.append(_parse_vec2(rest))
    elif keyword == "f":
        state.faces.append(_parse_face(rest))
    elif keyword == "o":
        state.object_name = rest.strip() or state.object_name
    elif keyword == "mtllib":
        # Path-safety check; MTL parsing is deferred.
        for token in rest.split():
            resolve_safe(asset_root, token)


def _parse_vec3(rest: str) -> tuple[float, float, float]:
    parts = rest.split()
    if len(parts) < _VEC3_MIN_COMPONENTS:
        raise MalformedAssetError(f"vec3 expects >={_VEC3_MIN_COMPONENTS} values, got {rest!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _parse_vec2(rest: str) -> tuple[float, float]:
    parts = rest.split()
    if len(parts) < _VEC2_MIN_COMPONENTS:
        raise MalformedAssetError(f"vec2 expects >={_VEC2_MIN_COMPONENTS} values, got {rest!r}")
    return float(parts[0]), float(parts[1])


def _parse_face(rest: str) -> list[tuple[int, int, int]]:
    """Return a list of (v_idx, vt_idx, vn_idx) per face corner; -1 for missing."""
    corners: list[tuple[int, int, int]] = []
    for token in rest.split():
        v_idx, vt_idx, vn_idx = _parse_face_corner(token)
        corners.append((v_idx, vt_idx, vn_idx))
    if len(corners) < _FACE_MIN_CORNERS:
        raise MalformedAssetError(f"face needs >={_FACE_MIN_CORNERS} corners, got {rest!r}")
    return corners


def _parse_face_corner(token: str) -> tuple[int, int, int]:
    parts = token.split("/")
    v_str = parts[0]
    vt_str = parts[1] if len(parts) > 1 else ""
    vn_str = parts[2] if len(parts) > _VEC2_MIN_COMPONENTS else ""
    return _to_index(v_str), _to_index(vt_str), _to_index(vn_str)


def _to_index(text: str) -> int:
    """OBJ indices are 1-based and may be negative (relative to current count)."""
    if not text:
        return -1
    value = int(text)
    return value - 1 if value > 0 else value


def _build_mesh(state: _ParseState) -> Mesh:
    if not state.positions:
        raise MalformedAssetError("OBJ contains no vertex positions")
    positions_src = np.asarray(state.positions, dtype=np.float32)
    normals_src = np.asarray(state.normals, dtype=np.float32) if state.normals else None
    texcoords_src = np.asarray(state.texcoords, dtype=np.float32) if state.texcoords else None

    unique_corners: dict[tuple[int, int, int], int] = {}
    out_positions: list[tuple[float, float, float]] = []
    out_normals: list[tuple[float, float, float]] = []
    out_texcoords: list[tuple[float, float]] = []
    out_indices: list[int] = []

    for face in state.faces:
        for v_idx, vt_idx, vn_idx in _triangulate_fan(face):
            key = (v_idx, vt_idx, vn_idx)
            existing = unique_corners.get(key)
            if existing is None:
                existing = len(unique_corners)
                unique_corners[key] = existing
                out_positions.append(tuple(positions_src[v_idx]))  # type: ignore[arg-type]
                if normals_src is not None and vn_idx >= 0:
                    out_normals.append(tuple(normals_src[vn_idx]))  # type: ignore[arg-type]
                if texcoords_src is not None and vt_idx >= 0:
                    out_texcoords.append(tuple(texcoords_src[vt_idx]))  # type: ignore[arg-type]
            out_indices.append(existing)

    positions_array = np.asarray(out_positions, dtype=np.float32)
    indices_array = np.asarray(out_indices, dtype=np.uint32)
    normals_array = np.asarray(out_normals, dtype=np.float32) if out_normals else None
    if normals_array is not None and len(normals_array) != len(positions_array):
        normals_array = None
    texcoords_array = np.asarray(out_texcoords, dtype=np.float32) if out_texcoords else None
    if texcoords_array is not None and len(texcoords_array) != len(positions_array):
        texcoords_array = None
    return Mesh(
        name=state.object_name,
        positions=positions_array,
        indices=indices_array,
        normals=normals_array,
        texcoords_0=texcoords_array,
    )


def _triangulate_fan(face: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Fan-triangulate an n-gon, anchored at the first corner."""
    triangles: list[tuple[int, int, int]] = []
    anchor = face[0]
    for i in range(1, len(face) - 1):
        triangles.append(anchor)
        triangles.append(face[i])
        triangles.append(face[i + 1])
    return triangles
