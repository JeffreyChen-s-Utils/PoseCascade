"""PLY (Stanford Polygon) importer — ASCII variant.

Supports the common ``element vertex`` block (positions + optional normals +
optional vertex colours) and ``element face`` with a polygon list property.
Faces are triangulated fan-style.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from posecascade.assets.types import ImportedScene, Mesh
from posecascade.errors import MalformedAssetError
from posecascade.scene.component import MeshRefComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene

_VERTEX_AXIS_PROPS = ("x", "y", "z")
_NORMAL_PROPS = ("nx", "ny", "nz")
_TEXCOORD_PROPS = ("s", "t")


@dataclass
class _ElementSpec:
    name: str
    count: int
    properties: list[tuple[str, str, bool, str | None]] = field(default_factory=list)
    # Each property is (name, scalar_type, is_list, list_length_type).


class PlyImporter:
    """Loads ASCII ``.ply`` files into :class:`ImportedScene`."""

    supported_extensions: tuple[str, ...] = (".ply",)

    def load(self, path: Path) -> ImportedScene:
        path = path.resolve()
        if not path.is_file():
            raise MalformedAssetError(f"PLY file not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        elements, body = _parse_header(text)
        mesh = _parse_body(elements, body, path.stem)
        node = Node(name=path.stem)
        node.add_component(MeshRefComponent(mesh_indices=(0,)))
        scene = Scene(name=path.stem)
        scene.root.add_child(node)
        return ImportedScene(meshes=(mesh,), scene=scene)


def _parse_header(text: str) -> tuple[list[_ElementSpec], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "ply":
        raise MalformedAssetError("PLY missing magic header")
    elements: list[_ElementSpec] = []
    end_index: int | None = None
    fmt_seen = False
    for index, raw in enumerate(lines[1:], start=1):
        line = raw.strip()
        if line == "end_header":
            end_index = index
            break
        keyword, _, rest = line.partition(" ")
        if keyword == "format":
            fmt_seen = True
            if not rest.startswith("ascii"):
                raise MalformedAssetError("only ASCII PLY is supported in skeleton importer")
        elif keyword == "element":
            name, count = rest.split()
            elements.append(_ElementSpec(name=name, count=int(count)))
        elif keyword == "property":
            elements[-1].properties.append(_parse_property(rest))
        # comments and other tokens are ignored
    if end_index is None or not fmt_seen:
        raise MalformedAssetError("PLY header missing end_header or format line")
    body = "\n".join(lines[end_index + 1:])
    return elements, body


def _parse_property(rest: str) -> tuple[str, str, bool, str | None]:
    tokens = rest.split()
    if tokens[0] == "list":
        # property list <count_type> <value_type> <name>
        return tokens[3], tokens[2], True, tokens[1]
    return tokens[1], tokens[0], False, None


def _parse_body(elements: list[_ElementSpec], body: str, name: str) -> Mesh:
    cursor = iter(body.splitlines())
    positions: np.ndarray | None = None
    normals: np.ndarray | None = None
    indices: np.ndarray | None = None
    for spec in elements:
        if spec.name == "vertex":
            positions, normals = _read_vertex_block(spec, cursor)
        elif spec.name == "face":
            indices = _read_face_block(spec, cursor)
        else:
            for _ in range(spec.count):
                next(cursor, "")
    if positions is None or indices is None:
        raise MalformedAssetError("PLY missing vertex or face element")
    return Mesh(name=name, positions=positions, indices=indices, normals=normals)


def _read_vertex_block(
    spec: _ElementSpec,
    cursor,
) -> tuple[np.ndarray, np.ndarray | None]:
    prop_names = [p[0] for p in spec.properties]
    pos_idx = [prop_names.index(axis) for axis in _VERTEX_AXIS_PROPS if axis in prop_names]
    if len(pos_idx) != len(_VERTEX_AXIS_PROPS):
        raise MalformedAssetError("PLY vertex element missing x/y/z properties")
    have_normals = all(p in prop_names for p in _NORMAL_PROPS)
    nrm_idx = [prop_names.index(p) for p in _NORMAL_PROPS] if have_normals else []
    positions = np.empty((spec.count, 3), dtype=np.float32)
    normals = np.empty((spec.count, 3), dtype=np.float32) if have_normals else None
    for i in range(spec.count):
        line = next(cursor, "").strip()
        if not line:
            raise MalformedAssetError("PLY vertex block truncated")
        tokens = line.split()
        positions[i] = [float(tokens[j]) for j in pos_idx]
        if normals is not None:
            normals[i] = [float(tokens[j]) for j in nrm_idx]
    return positions, normals


def _read_face_block(spec: _ElementSpec, cursor) -> np.ndarray:
    indices: list[int] = []
    for _ in range(spec.count):
        line = next(cursor, "").strip()
        if not line:
            raise MalformedAssetError("PLY face block truncated")
        tokens = line.split()
        corner_count = int(tokens[0])
        corners = [int(tokens[1 + j]) for j in range(corner_count)]
        for j in range(1, corner_count - 1):
            indices.extend((corners[0], corners[j], corners[j + 1]))
    return np.asarray(indices, dtype=np.uint32)
