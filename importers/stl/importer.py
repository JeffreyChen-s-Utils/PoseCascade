"""STL importer — both ASCII and binary variants.

STL is a triangle soup: every triangle ships its own three vertices and a face
normal. This importer emits one unique vertex per face corner (no welding) and
sets per-vertex normals to the face normal so the existing forward shader
renders flat-shaded surfaces correctly.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from posecascade.assets.types import ImportedScene, Mesh
from posecascade.errors import MalformedAssetError
from posecascade.scene.component import MeshRefComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene

_BINARY_HEADER_BYTES = 80
_BINARY_TRIANGLE_BYTES = 50
_BINARY_COUNT_BYTES = 4
_TRIANGLE_VERTEX_COUNT = 3
_FACET_NORMAL_TOKEN_COUNT = 4
_VEC3_FLOAT_COUNT = 3


class StlImporter:
    """Loads ``.stl`` files into :class:`ImportedScene`."""

    supported_extensions: tuple[str, ...] = (".stl",)

    def load(self, path: Path) -> ImportedScene:
        path = path.resolve()
        if not path.is_file():
            raise MalformedAssetError(f"STL file not found: {path}")
        payload = path.read_bytes()
        positions, normals = _decode_stl(payload)
        indices = np.arange(len(positions), dtype=np.uint32)
        mesh = Mesh(
            name=path.stem,
            positions=positions,
            indices=indices,
            normals=normals,
        )
        node = Node(name=path.stem)
        node.add_component(MeshRefComponent(mesh_indices=(0,)))
        scene = Scene(name=path.stem)
        scene.root.add_child(node)
        return ImportedScene(meshes=(mesh,), scene=scene)


def _decode_stl(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    if _looks_like_binary(payload):
        return _decode_binary(payload)
    return _decode_ascii(payload.decode("utf-8", errors="replace"))


def _looks_like_binary(payload: bytes) -> bool:
    """Detect binary STL by checking the size matches the embedded triangle count."""
    if len(payload) < _BINARY_HEADER_BYTES + _BINARY_COUNT_BYTES:
        return False
    triangle_count = struct.unpack_from("<I", payload, _BINARY_HEADER_BYTES)[0]
    expected = (
        _BINARY_HEADER_BYTES
        + _BINARY_COUNT_BYTES
        + triangle_count * _BINARY_TRIANGLE_BYTES
    )
    return len(payload) == expected


def _decode_binary(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    triangle_count = struct.unpack_from("<I", payload, _BINARY_HEADER_BYTES)[0]
    positions = np.empty((triangle_count * 3, 3), dtype=np.float32)
    normals = np.empty((triangle_count * 3, 3), dtype=np.float32)
    offset = _BINARY_HEADER_BYTES + _BINARY_COUNT_BYTES
    for i in range(triangle_count):
        nx, ny, nz, *vertex_floats = struct.unpack_from("<12f", payload, offset)
        positions[i * 3 + 0] = vertex_floats[0:3]
        positions[i * 3 + 1] = vertex_floats[3:6]
        positions[i * 3 + 2] = vertex_floats[6:9]
        face_normal = (nx, ny, nz)
        normals[i * 3 + 0] = face_normal
        normals[i * 3 + 1] = face_normal
        normals[i * 3 + 2] = face_normal
        offset += _BINARY_TRIANGLE_BYTES
    return positions, normals


def _decode_ascii(text: str) -> tuple[np.ndarray, np.ndarray]:
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    current_normal: tuple[float, float, float] | None = None
    pending_vertices: list[tuple[float, float, float]] = []
    for raw_line in text.splitlines():
        keyword, parts = _split_keyword(raw_line)
        if keyword == "facet":
            current_normal = _parse_facet_normal(parts)
        elif keyword == "vertex":
            pending_vertices.append(_parse_vec3(parts))
        elif keyword == "endfacet":
            if current_normal is None or len(pending_vertices) != _TRIANGLE_VERTEX_COUNT:
                raise MalformedAssetError("malformed ASCII STL facet")
            positions.extend(pending_vertices)
            normals.extend([current_normal, current_normal, current_normal])
            current_normal = None
            pending_vertices = []
    if not positions:
        raise MalformedAssetError("ASCII STL contains no triangles")
    return np.asarray(positions, dtype=np.float32), np.asarray(normals, dtype=np.float32)


def _split_keyword(raw_line: str) -> tuple[str, list[str]]:
    tokens = raw_line.strip().split()
    if not tokens:
        return "", []
    return tokens[0].lower(), tokens[1:]


def _parse_facet_normal(parts: list[str]) -> tuple[float, float, float]:
    if len(parts) < _FACET_NORMAL_TOKEN_COUNT or parts[0].lower() != "normal":
        raise MalformedAssetError("ASCII STL facet missing normal")
    return float(parts[1]), float(parts[2]), float(parts[3])


def _parse_vec3(parts: list[str]) -> tuple[float, float, float]:
    if len(parts) < _VEC3_FLOAT_COUNT:
        raise MalformedAssetError(f"ASCII STL vertex needs {_VEC3_FLOAT_COUNT} floats")
    return float(parts[0]), float(parts[1]), float(parts[2])
