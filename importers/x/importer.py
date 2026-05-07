"""DirectX .x (text) importer.

The .x format is a header-tagged tree of templates separated by braces;
this importer parses the static-mesh subset that MMD-era tools actually
emit (no anim sets, no skinning) and yields :class:`ImportedScene`.

Supported templates:

- ``Frame`` — coordinate-frame node, may carry a ``FrameTransformMatrix``
  and any number of nested Frames / Meshes.
- ``FrameTransformMatrix`` — 4×4 row-major float matrix.
- ``Mesh`` — vertex array + face list (n-gons triangulated fan-style).
- ``MeshNormals`` — per-vertex normal array (normal-faces ignored; we
  rely on the engine's later normal recompute if the per-face normals
  diverge from per-vertex).
- ``MeshTextureCoords`` — per-vertex (u, v).
- ``MeshMaterialList`` — material count + per-face material index,
  followed by ``Material`` records. The first material's diffuse colour
  becomes the mesh's :attr:`Mesh.base_color`.
- ``Material`` — face / specular / emissive colours plus optional
  ``TextureFilename`` (the path is captured but the texture is *not*
  loaded — keeps the importer dependency-free; assets/path_safety can
  resolve it later when texture decoding lands).

Templates the importer doesn't recognise are silently skipped. The .x
format is loose enough that strict parsing breaks on every other
exporter's quirks; lenient skip-on-unknown is what every viable
runtime does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from posecascade.assets.types import ImportedScene, Mesh
from posecascade.errors import MalformedAssetError
from posecascade.scene.component import MeshRefComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.scene.transform import Transform
from posecascade.utils.math3d import decompose_trs

_HEADER_PREFIX = "xof "
_MATRIX_FLOAT_COUNT = 16
# Punctuation tokens promoted to module-level constants so the
# parser's brace / separator comparisons read as grammar-of-the-format,
# not as accidental string-literal sniffing (and ruff's S105 doesn't
# flag them as embedded credentials).
_BRACE_OPEN = "{"
_BRACE_CLOSE = "}"
_SEPARATOR_TOKENS = (",", ";")


@dataclass
class _RawMesh:
    name: str
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    normals: list[tuple[float, float, float]] | None = None
    texcoords: list[tuple[float, float]] | None = None
    base_color: tuple[float, float, float, float] | None = None


@dataclass
class _RawFrame:
    name: str
    transform_matrix: NDArray[np.float32] | None = None
    meshes: list[_RawMesh] = field(default_factory=list)
    children: list[_RawFrame] = field(default_factory=list)


class XImporter:
    """Loads ``.x`` text files into :class:`ImportedScene`."""

    supported_extensions: tuple[str, ...] = (".x",)

    def load(self, path: Path) -> ImportedScene:
        path = path.resolve()
        if not path.is_file():
            raise MalformedAssetError(f".x file not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.lstrip().startswith(_HEADER_PREFIX):
            raise MalformedAssetError("not a DirectX .x text file (missing 'xof' header)")
        tokens = _tokenize(_strip_header(text))
        cursor = _Cursor(tokens)
        frames, top_meshes = _parse_top_level(cursor)
        return _build_imported_scene(path.stem, frames, top_meshes)


# ----- top-level glue ---------------------------------------------------
def _build_imported_scene(
    stem: str,
    frames: list[_RawFrame],
    top_meshes: list[_RawMesh],
) -> ImportedScene:
    """Flatten the parsed Frame tree into meshes + a scene graph."""
    flat_meshes: list[Mesh] = []
    scene = Scene(name=stem)
    # Top-level meshes (not inside any Frame) get attached to the scene root.
    for raw in top_meshes:
        _attach_raw_mesh(raw, flat_meshes, scene.root)
    for frame in frames:
        scene.root.add_child(_build_frame_node(frame, flat_meshes))
    if not flat_meshes:
        raise MalformedAssetError(".x file contained no usable Mesh blocks")
    return ImportedScene(meshes=tuple(flat_meshes), scene=scene)


def _build_frame_node(frame: _RawFrame, flat_meshes: list[Mesh]) -> Node:
    transform = _transform_from_matrix(frame.transform_matrix)
    node = Node(name=frame.name, transform=transform)
    for raw_mesh in frame.meshes:
        _attach_raw_mesh(raw_mesh, flat_meshes, node)
    for child in frame.children:
        node.add_child(_build_frame_node(child, flat_meshes))
    return node


def _attach_raw_mesh(
    raw: _RawMesh, flat_meshes: list[Mesh], parent: Node,
) -> None:
    """Materialise a :class:`Mesh` from ``raw`` and attach to ``parent``."""
    positions = np.asarray(raw.positions, dtype=np.float32)
    indices = np.asarray(raw.indices, dtype=np.uint32)
    normals = (
        np.asarray(raw.normals, dtype=np.float32) if raw.normals else None
    )
    texcoords = (
        np.asarray(raw.texcoords, dtype=np.float32) if raw.texcoords else None
    )
    if normals is not None and normals.shape[0] != positions.shape[0]:
        normals = None    # mismatched per-face normals — let the renderer fall back
    if texcoords is not None and texcoords.shape[0] != positions.shape[0]:
        texcoords = None
    mesh = Mesh(
        name=raw.name,
        positions=positions,
        indices=indices,
        normals=normals,
        texcoords_0=texcoords,
        base_color=raw.base_color,
    )
    mesh_index = len(flat_meshes)
    flat_meshes.append(mesh)
    mesh_node = Node(name=raw.name)
    mesh_node.add_component(MeshRefComponent(mesh_indices=(mesh_index,)))
    parent.add_child(mesh_node)


def _transform_from_matrix(matrix: NDArray[np.float32] | None) -> Transform:
    if matrix is None:
        return Transform()
    translation, rotation, scale = decompose_trs(matrix)
    return Transform(translation=translation, rotation=rotation, scale=scale)


# ----- tokenizer --------------------------------------------------------
@dataclass
class _Cursor:
    tokens: list[str]
    index: int = 0

    def peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self) -> str:
        tok = self.peek()
        if tok is None:
            raise MalformedAssetError(".x parser ran out of tokens")
        self.index += 1
        return tok

    def expect(self, expected: str) -> None:
        actual = self.take()
        if actual != expected:
            raise MalformedAssetError(
                f".x parser expected {expected!r}, got {actual!r}",
            )

    def consume_optional(self, expected: str) -> bool:
        if self.peek() == expected:
            self.index += 1
            return True
        return False


def _strip_header(text: str) -> str:
    """Drop the ``xof 0303txt 0032`` magic line."""
    newline = text.find("\n")
    if newline < 0:
        raise MalformedAssetError(".x file has no body after the xof header")
    return text[newline + 1:]


def _tokenize(text: str) -> list[str]:
    """Split ``text`` into tokens.

    Tokens are: identifiers, numbers, strings, and the punctuation
    ``{`` / ``}`` / ``;`` / ``,``. Comments (``//`` or ``#`` to end of
    line) are dropped. Quoted strings keep their quotes so the parser
    can distinguish them from identifiers.
    """
    tokens: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        char = text[i]
        if char.isspace():
            i += 1
            continue
        if char == "/" and i + 1 < length and text[i + 1] == "/":
            i = _skip_to_eol(text, i)
            continue
        if char == "#":
            i = _skip_to_eol(text, i)
            continue
        if char in "{};,":
            tokens.append(char)
            i += 1
            continue
        if char == '"':
            end = text.find('"', i + 1)
            if end < 0:
                raise MalformedAssetError(".x string literal not terminated")
            tokens.append(text[i:end + 1])
            i = end + 1
            continue
        # Run of non-special characters = identifier or number.
        end = i
        while end < length and not text[end].isspace() and text[end] not in '{};,/#"':
            end += 1
        tokens.append(text[i:end])
        i = end
    return tokens


def _skip_to_eol(text: str, start: int) -> int:
    end = text.find("\n", start)
    if end < 0:
        return len(text)
    return end + 1


# ----- recursive descent -----------------------------------------------
def _parse_top_level(cursor: _Cursor) -> tuple[list[_RawFrame], list[_RawMesh]]:
    frames: list[_RawFrame] = []
    meshes: list[_RawMesh] = []
    while cursor.peek() is not None:
        keyword = cursor.take()
        name = _take_optional_name(cursor)
        if keyword == "Frame":
            frames.append(_parse_frame_body(cursor, name or "frame"))
        elif keyword == "Mesh":
            meshes.append(_parse_mesh_body(cursor, name or "mesh"))
        elif keyword == "template":
            _skip_block(cursor)         # template definitions — schema only
        else:
            _skip_block(cursor)
    return frames, meshes


def _take_optional_name(cursor: _Cursor) -> str | None:
    """Consume the optional identifier that follows a template keyword."""
    next_token = cursor.peek()
    if next_token is None or next_token == _BRACE_OPEN:
        return None
    return cursor.take()


def _parse_frame_body(cursor: _Cursor, name: str) -> _RawFrame:
    cursor.expect(_BRACE_OPEN)
    frame = _RawFrame(name=name)
    while True:
        token = cursor.peek()
        if token is None:
            raise MalformedAssetError(f"Frame {name!r} not closed")
        if token == _BRACE_CLOSE:
            cursor.take()
            return frame
        keyword = cursor.take()
        child_name = _take_optional_name(cursor)
        if keyword == "FrameTransformMatrix":
            frame.transform_matrix = _parse_transform_matrix(cursor)
        elif keyword == "Mesh":
            frame.meshes.append(_parse_mesh_body(cursor, child_name or "mesh"))
        elif keyword == "Frame":
            frame.children.append(_parse_frame_body(cursor, child_name or "frame"))
        else:
            _skip_block(cursor)


def _parse_transform_matrix(cursor: _Cursor) -> NDArray[np.float32]:
    cursor.expect(_BRACE_OPEN)
    floats = _read_floats(cursor, _MATRIX_FLOAT_COUNT)
    _gobble_separators(cursor)
    cursor.expect("}")
    matrix = np.asarray(floats, dtype=np.float32).reshape(4, 4)
    # FrameTransformMatrix is column-major in DirectX; transpose so the
    # rest of the engine's row-major helpers see a familiar layout.
    return matrix.T.astype(np.float32, copy=False)


def _parse_mesh_body(cursor: _Cursor, name: str) -> _RawMesh:
    cursor.expect(_BRACE_OPEN)
    raw = _RawMesh(name=name)
    raw.positions = _read_vertex_array(cursor)
    raw.indices = _read_face_array(cursor)
    while True:
        token = cursor.peek()
        if token is None:
            raise MalformedAssetError(f"Mesh {name!r} not closed")
        if token == _BRACE_CLOSE:
            cursor.take()
            return raw
        keyword = cursor.take()
        _take_optional_name(cursor)
        if keyword == "MeshNormals":
            raw.normals = _parse_normal_block(cursor)
        elif keyword == "MeshTextureCoords":
            raw.texcoords = _parse_texcoord_block(cursor)
        elif keyword == "MeshMaterialList":
            colour = _parse_material_list(cursor)
            if colour is not None and raw.base_color is None:
                raw.base_color = colour
        else:
            _skip_block(cursor)


def _read_vertex_array(cursor: _Cursor) -> list[tuple[float, float, float]]:
    count = _read_integer(cursor)
    _gobble_separators(cursor)
    vertices: list[tuple[float, float, float]] = []
    for _ in range(count):
        x = _read_float(cursor)
        _gobble_separators(cursor)
        y = _read_float(cursor)
        _gobble_separators(cursor)
        z = _read_float(cursor)
        _gobble_separators(cursor)
        vertices.append((x, y, z))
    return vertices


def _read_face_array(cursor: _Cursor) -> list[int]:
    count = _read_integer(cursor)
    _gobble_separators(cursor)
    indices: list[int] = []
    for _ in range(count):
        corner_count = _read_integer(cursor)
        _gobble_separators(cursor)
        corners: list[int] = []
        for _ in range(corner_count):
            corners.append(_read_integer(cursor))
            _gobble_separators(cursor)
        for j in range(1, corner_count - 1):
            indices.extend((corners[0], corners[j], corners[j + 1]))
    return indices


def _parse_normal_block(cursor: _Cursor) -> list[tuple[float, float, float]]:
    cursor.expect(_BRACE_OPEN)
    normals = _read_vertex_array(cursor)
    # Discard the per-face normal index list — engine recomputes /
    # interpolates per-vertex normals; matching them face-by-face is
    # rarely worth the extra complexity at this importer tier.
    _read_face_array(cursor)
    _skip_to_close_brace(cursor)
    return normals


def _parse_texcoord_block(cursor: _Cursor) -> list[tuple[float, float]]:
    cursor.expect(_BRACE_OPEN)
    count = _read_integer(cursor)
    _gobble_separators(cursor)
    coords: list[tuple[float, float]] = []
    for _ in range(count):
        u = _read_float(cursor)
        _gobble_separators(cursor)
        v = _read_float(cursor)
        _gobble_separators(cursor)
        coords.append((u, v))
    _skip_to_close_brace(cursor)
    return coords


def _parse_material_list(
    cursor: _Cursor,
) -> tuple[float, float, float, float] | None:
    cursor.expect(_BRACE_OPEN)
    material_count = _read_integer(cursor)
    _gobble_separators(cursor)
    face_count = _read_integer(cursor)
    _gobble_separators(cursor)
    # Skip the per-face material index list — runtime treats every face
    # as drawn with material 0 (we only capture the first material's
    # diffuse colour anyway).
    for _ in range(face_count):
        _read_integer(cursor)
        _gobble_separators(cursor)
    first_diffuse: tuple[float, float, float, float] | None = None
    materials_seen = 0
    while True:
        token = cursor.peek()
        if token is None:
            raise MalformedAssetError("MeshMaterialList not closed")
        if token == _BRACE_CLOSE:
            cursor.take()
            return first_diffuse
        keyword = cursor.take()
        _take_optional_name(cursor)
        if keyword == "Material":
            colour = _parse_material(cursor)
            if first_diffuse is None and materials_seen < material_count:
                first_diffuse = colour
            materials_seen += 1
        else:
            _skip_block(cursor)


def _parse_material(cursor: _Cursor) -> tuple[float, float, float, float]:
    cursor.expect(_BRACE_OPEN)
    diffuse_floats = _read_floats(cursor, 4)
    _gobble_separators(cursor)
    # specular_power, specular(rgb), emissive(rgb) — we read them so the
    # cursor advances but only the diffuse colour ends up on the mesh.
    _read_float(cursor)
    _gobble_separators(cursor)
    _read_floats(cursor, 3)
    _gobble_separators(cursor)
    _read_floats(cursor, 3)
    _gobble_separators(cursor)
    _skip_to_close_brace(cursor)
    return tuple(float(v) for v in diffuse_floats)  # type: ignore[return-value]


# ----- low-level token helpers -----------------------------------------
def _skip_block(cursor: _Cursor) -> None:
    """Skip an entire ``{ ... }`` block, supporting nested braces."""
    # The opening brace may not have been consumed yet — consume tokens
    # until we hit it, then balance.
    depth = 0
    while True:
        token = cursor.peek()
        if token is None:
            return
        if token == _BRACE_OPEN:
            cursor.take()
            depth += 1
            continue
        if depth == 0:
            return
        if token == _BRACE_CLOSE:
            cursor.take()
            depth -= 1
            if depth == 0:
                return
            continue
        cursor.take()


def _skip_to_close_brace(cursor: _Cursor) -> None:
    """Skip tokens until the matching ``}`` for the currently-open block."""
    depth = 1
    while depth > 0:
        token = cursor.peek()
        if token is None:
            raise MalformedAssetError(".x block not closed")
        if token == _BRACE_OPEN:
            depth += 1
        elif token == _BRACE_CLOSE:
            depth -= 1
            if depth == 0:
                cursor.take()
                return
        cursor.take()


def _read_integer(cursor: _Cursor) -> int:
    token = cursor.take()
    try:
        return int(token)
    except ValueError as err:
        raise MalformedAssetError(f"expected integer, got {token!r}") from err


def _read_float(cursor: _Cursor) -> float:
    token = cursor.take()
    try:
        return float(token)
    except ValueError as err:
        raise MalformedAssetError(f"expected float, got {token!r}") from err


def _read_floats(cursor: _Cursor, count: int) -> list[float]:
    floats: list[float] = []
    for index in range(count):
        floats.append(_read_float(cursor))
        if index < count - 1:
            _gobble_separators(cursor)
    return floats


def _gobble_separators(cursor: _Cursor) -> None:
    """Consume every consecutive ``,`` / ``;`` token at the cursor.

    The .x grammar uses the two punctuation marks interchangeably as
    "advance to next value", which makes precise position-based parsing
    fragile across exporters. Since the array sizes are always declared
    up front, this importer reads exactly N values and gobbles whatever
    separators sit between / after them.
    """
    while cursor.peek() in _SEPARATOR_TOKENS:
        cursor.take()
