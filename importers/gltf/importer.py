"""glTF 2.0 / GLB importer.

Walks a ``GLTF2`` document into :class:`~posecascade.assets.types.ImportedScene`:
positions, normals, UVs, indices, multiple primitives per mesh, the node
hierarchy with TRS, and basic baseColor materials. Skinning and animation are
left for the next pass.

External buffer ``uri`` references resolve through
:func:`posecascade.assets.path_safety.resolve_safe` so a hostile glTF cannot
read files outside the asset root.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pygltflib import GLTF2

from posecascade.animation.rig_normalize import (
    collapse_wrapper_empties,
    normalize_character_orientation,
)
from posecascade.animation.spring import (
    detect_chains,
    resolve_chain_params_or_none,
)
from posecascade.assets.path_safety import resolve_safe
from posecascade.assets.types import ImportedScene, Mesh, Skin, Texture
from posecascade.errors import MalformedAssetError
from posecascade.render.constants import (
    MAX_EMBEDDED_BUFFER_BYTES,
    MAX_MESH_INDICES,
    MAX_MESH_VERTICES,
    MAX_TEXTURE_DIMENSION,
)
from posecascade.scene.component import (
    MeshRefComponent,
    SkinRefComponent,
    SpringChainComponent,
)
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.scene.transform import Transform
from posecascade.utils.math3d import quat_identity, vec3

_DATA_URI_PREFIX = "data:application/octet-stream;base64,"
_DATA_URI_GLTF_BUFFER_PREFIX = "data:application/gltf-buffer;base64,"

_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


@dataclass(frozen=True)
class _LoadedBuffers:
    blobs: tuple[bytes, ...]


class GltfImporter:
    """Loads ``.gltf`` and ``.glb`` files into :class:`ImportedScene`."""

    supported_extensions: tuple[str, ...] = (".gltf", ".glb")

    def load(self, path: Path) -> ImportedScene:
        path = path.resolve()
        if not path.is_file():
            raise MalformedAssetError(f"glTF file not found: {path}")
        gltf = GLTF2().load(str(path))
        if gltf is None:
            raise MalformedAssetError(f"pygltflib could not parse {path}")
        buffers = _load_buffers(gltf, path)
        textures = _load_textures(gltf, buffers, path.parent)
        material_to_image = _material_base_color_image(gltf)
        meshes, mesh_index_map = _build_meshes(gltf, buffers, material_to_image)
        scene, all_nodes = _build_scene(gltf, path.stem, mesh_index_map)
        skins = _build_skins(gltf, buffers, all_nodes)
        # Now that skins exist, attach SkinRefComponents to nodes that reference one.
        _attach_skins(gltf, all_nodes, skins)
        # Auto-detect spring-physics chains in the joint set (e.g. ``hair_C_0..3``)
        # and tag the chain anchor with a SpringChainComponent. The PhysicsHost
        # picks these up at scene-load time without any user scripting.
        _attach_spring_chains(skins)
        # Fold wrapper EMPTY chains (Sketchfab/VRoid/FBX nest several of
        # them) into a single transform on the first content node. User
        # scripts targeting the character root see one rest rotation
        # instead of a stack that double-rotates when composed with yaw.
        collapse_wrapper_empties(scene, skins)
        # Auto-rotate the top wrapper so each skin's first significant
        # joint ends up with cumulative rotation = identity, AND lift
        # the resulting rig along +Y so the lowest joint sits at world
        # Y=0. The lift compensates for the case where rotating the
        # rig (without rebinding) puts joints below the floor — common
        # for VRoid characters where hip's translation goes +Y in the
        # bone frame but +Y is "below" once the chain is normalized.
        normalize_character_orientation(scene, skins)
        return ImportedScene(
            meshes=tuple(meshes), textures=textures, skins=skins, scene=scene,
        )


def _load_buffers(gltf: GLTF2, gltf_path: Path) -> _LoadedBuffers:
    asset_root = gltf_path.parent
    blobs: list[bytes] = []
    glb_blob = gltf.binary_blob() if gltf_path.suffix.lower() == ".glb" else None
    for buffer_index, buffer in enumerate(gltf.buffers or ()):
        blobs.append(_resolve_buffer(buffer, buffer_index, glb_blob, asset_root))
    return _LoadedBuffers(blobs=tuple(blobs))


def _resolve_buffer(buffer, buffer_index: int, glb_blob: bytes | None, asset_root: Path) -> bytes:
    uri = getattr(buffer, "uri", None)
    if uri is None:
        if buffer_index == 0 and glb_blob is not None:
            return glb_blob
        raise MalformedAssetError(f"buffer {buffer_index} has no uri and no GLB blob available")
    if uri.startswith(_DATA_URI_PREFIX):
        return _decode_data_uri(uri[len(_DATA_URI_PREFIX):])
    if uri.startswith(_DATA_URI_GLTF_BUFFER_PREFIX):
        return _decode_data_uri(uri[len(_DATA_URI_GLTF_BUFFER_PREFIX):])
    if uri.startswith("data:"):
        raise MalformedAssetError(f"unsupported data URI scheme in buffer {buffer_index}")
    resolved = resolve_safe(asset_root, uri)
    return resolved.read_bytes()


def _decode_data_uri(b64: str) -> bytes:
    payload = base64.b64decode(b64, validate=True)
    if len(payload) > MAX_EMBEDDED_BUFFER_BYTES:
        raise MalformedAssetError(
            f"embedded buffer exceeds MAX_EMBEDDED_BUFFER_BYTES "
            f"({len(payload)} > {MAX_EMBEDDED_BUFFER_BYTES})"
        )
    return payload


def _accessor_array(gltf: GLTF2, buffers: _LoadedBuffers, accessor_index: int) -> np.ndarray:
    accessor = gltf.accessors[accessor_index]
    if accessor.bufferView is None:
        raise MalformedAssetError(f"accessor {accessor_index} has no bufferView")
    view = gltf.bufferViews[accessor.bufferView]
    blob = buffers.blobs[view.buffer]
    dtype = _COMPONENT_DTYPE.get(accessor.componentType)
    if dtype is None:
        raise MalformedAssetError(f"unsupported componentType {accessor.componentType}")
    components = _TYPE_COMPONENTS.get(accessor.type)
    if components is None:
        raise MalformedAssetError(f"unsupported accessor type {accessor.type!r}")
    base_offset = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    byte_stride = view.byteStride or (np.dtype(dtype).itemsize * components)
    return _slice_strided(blob, base_offset, byte_stride, accessor.count, components, dtype)


def _slice_strided(
    blob: bytes,
    base_offset: int,
    stride: int,
    count: int,
    components: int,
    dtype: np.dtype,
) -> np.ndarray:
    item_size = np.dtype(dtype).itemsize * components
    if stride == item_size:
        view = np.frombuffer(blob, dtype=dtype, count=count * components, offset=base_offset)
        return view.reshape(count, components) if components > 1 else view.copy()
    out = np.empty((count, components), dtype=dtype)
    for i in range(count):
        offset = base_offset + i * stride
        out[i] = np.frombuffer(blob, dtype=dtype, count=components, offset=offset)
    return out if components > 1 else out.reshape(-1)


def _material_base_color_image(gltf: GLTF2) -> dict[int, int]:
    """Map ``material_index → image_index`` for base-colour textures.

    Returns an empty dict when the document has no materials or textures.
    Materials without a ``baseColorTexture`` are simply absent from the map.
    """
    out: dict[int, int] = {}
    materials = gltf.materials or ()
    textures = gltf.textures or ()
    if not materials or not textures:
        return out
    for mat_idx, material in enumerate(materials):
        pbr = getattr(material, "pbrMetallicRoughness", None)
        base_color_texture = getattr(pbr, "baseColorTexture", None) if pbr else None
        if base_color_texture is None:
            continue
        tex_idx = base_color_texture.index
        if tex_idx is None or tex_idx >= len(textures):
            continue
        source = getattr(textures[tex_idx], "source", None)
        if source is None:
            continue
        out[mat_idx] = source
    return out


def _load_textures(
    gltf: GLTF2,
    buffers: _LoadedBuffers,
    asset_root: Path,
) -> tuple[Texture, ...]:
    """Decode each glTF image into an RGBA :class:`Texture`.

    Images that fail to decode (corrupt bytes, unsupported format) are
    replaced with a 1×1 white placeholder so the renderer still has a valid
    slot to bind.
    """
    images = gltf.images or ()
    if not images:
        return ()
    from io import BytesIO  # noqa: PLC0415 — lazy stdlib import

    from PIL import Image, UnidentifiedImageError  # noqa: PLC0415 — lazy heavy import

    out: list[Texture] = []
    for image_index, image in enumerate(images):
        name = getattr(image, "name", None) or f"image_{image_index}"
        try:
            data = _extract_image_bytes(gltf, image, buffers, asset_root)
            pil_image = Image.open(BytesIO(data)).convert("RGBA")
            if pil_image.width > MAX_TEXTURE_DIMENSION or pil_image.height > MAX_TEXTURE_DIMENSION:
                raise MalformedAssetError(
                    f"texture {name!r} exceeds MAX_TEXTURE_DIMENSION "
                    f"({pil_image.width}x{pil_image.height})"
                )
            pixels = np.array(pil_image, dtype=np.uint8)
            out.append(Texture(name=name, pixels=pixels, srgb=True))
        except (OSError, UnidentifiedImageError, MalformedAssetError, ValueError):
            placeholder = np.full((1, 1, 4), 255, dtype=np.uint8)
            out.append(Texture(name=name, pixels=placeholder, srgb=True))
    return tuple(out)


def _extract_image_bytes(
    gltf: GLTF2,
    image,
    buffers: _LoadedBuffers,
    asset_root: Path,
) -> bytes:
    """Get the raw image bytes from a glTF image (data URI / external file / bufferView)."""
    uri = getattr(image, "uri", None)
    if uri:
        if uri.startswith("data:"):
            comma = uri.find(",")
            if comma < 0:
                raise MalformedAssetError("malformed data URI in image")
            payload = base64.b64decode(uri[comma + 1:], validate=True)
            if len(payload) > MAX_EMBEDDED_BUFFER_BYTES:
                raise MalformedAssetError("embedded image exceeds MAX_EMBEDDED_BUFFER_BYTES")
            return payload
        return resolve_safe(asset_root, uri).read_bytes()
    buffer_view = getattr(image, "bufferView", None)
    if buffer_view is None:
        raise MalformedAssetError("glTF image has no uri and no bufferView")
    view = gltf.bufferViews[buffer_view]
    blob = buffers.blobs[view.buffer]
    offset = view.byteOffset or 0
    return blob[offset:offset + view.byteLength]


def _build_meshes(
    gltf: GLTF2,
    buffers: _LoadedBuffers,
    material_to_image: dict[int, int] | None = None,
) -> tuple[list[Mesh], dict[int, tuple[int, ...]]]:
    """Decode every glTF mesh's primitives into our flat :class:`Mesh` list.

    The second return value maps a glTF mesh index to the tuple of indices
    that mesh occupies in the flat list — needed to populate
    :class:`MeshRefComponent` on the nodes that reference it.
    """
    material_lookup = material_to_image or {}
    meshes: list[Mesh] = []
    mesh_index_map: dict[int, tuple[int, ...]] = {}
    for mesh_index, mesh in enumerate(gltf.meshes or ()):
        flat_indices: list[int] = []
        for primitive_index, primitive in enumerate(mesh.primitives):
            flat_indices.append(len(meshes))
            meshes.append(
                _build_primitive(
                    gltf, buffers, mesh, primitive,
                    mesh_index, primitive_index, material_lookup,
                )
            )
        mesh_index_map[mesh_index] = tuple(flat_indices)
    return meshes, mesh_index_map


def _build_primitive(
    gltf: GLTF2,
    buffers: _LoadedBuffers,
    mesh,
    primitive,
    mesh_index: int,
    primitive_index: int,
    material_to_image: dict[int, int],
) -> Mesh:
    attributes = primitive.attributes
    positions = _read_attribute(gltf, buffers, attributes.POSITION, "POSITION", required=True)
    if positions.shape[0] > MAX_MESH_VERTICES:
        raise MalformedAssetError(f"mesh exceeds MAX_MESH_VERTICES: {positions.shape[0]}")
    normals = _read_attribute(gltf, buffers, attributes.NORMAL, "NORMAL", required=False)
    texcoords = _read_attribute(gltf, buffers, attributes.TEXCOORD_0, "TEXCOORD_0", required=False)
    joints_0 = _read_attribute(gltf, buffers, attributes.JOINTS_0, "JOINTS_0", required=False)
    weights_0 = _read_attribute(gltf, buffers, attributes.WEIGHTS_0, "WEIGHTS_0", required=False)
    indices = _read_indices(gltf, buffers, primitive.indices, positions.shape[0])
    if indices.shape[0] > MAX_MESH_INDICES:
        raise MalformedAssetError(f"mesh exceeds MAX_MESH_INDICES: {indices.shape[0]}")
    name = mesh.name or f"mesh_{mesh_index}_{primitive_index}"
    base_color, base_color_texture_index = _resolve_material(
        gltf, primitive, material_to_image,
    )
    return Mesh(
        name=name,
        positions=positions.astype(np.float32, copy=False),
        indices=indices.astype(np.uint32, copy=False),
        normals=normals.astype(np.float32, copy=False) if normals is not None else None,
        texcoords_0=texcoords.astype(np.float32, copy=False) if texcoords is not None else None,
        joints_0=joints_0.astype(np.uint16, copy=False) if joints_0 is not None else None,
        weights_0=weights_0.astype(np.float32, copy=False) if weights_0 is not None else None,
        base_color=base_color,
        base_color_texture_index=base_color_texture_index,
    )


def _resolve_material(
    gltf: GLTF2,
    primitive,
    material_to_image: dict[int, int],
) -> tuple[tuple[float, float, float, float] | None, int | None]:
    """Pull baseColorFactor and base-colour image index from the primitive's material."""
    material_index = getattr(primitive, "material", None)
    if material_index is None or material_index >= len(gltf.materials or ()):
        return None, None
    material = gltf.materials[material_index]
    pbr = getattr(material, "pbrMetallicRoughness", None)
    factor = getattr(pbr, "baseColorFactor", None) if pbr else None
    base_color: tuple[float, float, float, float] | None = None
    rgba_components = 4
    if factor is not None and len(factor) >= rgba_components:
        base_color = (float(factor[0]), float(factor[1]), float(factor[2]), float(factor[3]))
    return base_color, material_to_image.get(material_index)


def _read_attribute(
    gltf: GLTF2,
    buffers: _LoadedBuffers,
    accessor_index: int | None,
    label: str,
    *,
    required: bool,
) -> np.ndarray | None:
    if accessor_index is None:
        if required:
            raise MalformedAssetError(f"primitive missing required attribute {label}")
        return None
    return _accessor_array(gltf, buffers, accessor_index)


def _read_indices(
    gltf: GLTF2,
    buffers: _LoadedBuffers,
    accessor_index: int | None,
    vertex_count: int,
) -> np.ndarray:
    if accessor_index is None:
        return np.arange(vertex_count, dtype=np.uint32)
    return _accessor_array(gltf, buffers, accessor_index).astype(np.uint32, copy=False)


def _build_scene(
    gltf: GLTF2,
    name: str,
    mesh_index_map: dict[int, tuple[int, ...]],
) -> tuple[Scene, list[Node | None]]:
    """Build the scene tree and return ``(scene, all_nodes)``.

    ``all_nodes[i]`` is the :class:`Node` corresponding to ``gltf.nodes[i]``,
    or ``None`` if that node was not reachable from the active scene
    (skin-only nodes get materialised by :func:`_build_skins` lazily).
    """
    scene = Scene(name=name)
    all_nodes: list[Node | None] = [None] * len(gltf.nodes or ())
    if not gltf.scenes:
        return scene, all_nodes
    default_index = gltf.scene if gltf.scene is not None else 0
    gltf_scene = gltf.scenes[default_index]
    for node_index in gltf_scene.nodes or ():
        scene.root.add_child(_build_node(gltf, node_index, mesh_index_map, all_nodes))
    return scene, all_nodes


def _build_node(
    gltf: GLTF2,
    node_index: int,
    mesh_index_map: dict[int, tuple[int, ...]],
    all_nodes: list[Node | None],
) -> Node:
    node = gltf.nodes[node_index]
    transform = _node_transform(node)
    out = Node(name=node.name or f"node_{node_index}", transform=transform)
    all_nodes[node_index] = out
    if node.mesh is not None:
        flat_indices = mesh_index_map.get(node.mesh, ())
        if flat_indices:
            out.add_component(MeshRefComponent(mesh_indices=flat_indices))
    for child_index in node.children or ():
        out.add_child(_build_node(gltf, child_index, mesh_index_map, all_nodes))
    return out


def _build_skins(
    gltf: GLTF2,
    buffers: _LoadedBuffers,
    all_nodes: list[Node | None],
) -> tuple[Skin, ...]:
    """Decode every glTF skin into a :class:`Skin` (joint nodes + inverse-bind matrices)."""
    if not gltf.skins:
        return ()
    skins: list[Skin] = []
    for skin_index, gltf_skin in enumerate(gltf.skins):
        joints = tuple(
            _ensure_node(gltf, joint_idx, all_nodes) for joint_idx in (gltf_skin.joints or ())
        )
        ibms = _read_inverse_bind_matrices(gltf, buffers, gltf_skin, len(joints))
        name = gltf_skin.name or f"skin_{skin_index}"
        skins.append(Skin(name=name, joints=joints, inverse_bind_matrices=ibms))
    return tuple(skins)


def _ensure_node(gltf: GLTF2, node_index: int, all_nodes: list[Node | None]) -> Node:
    """Look up or lazily materialise the :class:`Node` for a glTF node index.

    Skin joint nodes are sometimes outside the active scene tree; those still
    need a :class:`Node` so we can read their TRS for bone-matrix uploads.
    """
    if 0 <= node_index < len(all_nodes) and all_nodes[node_index] is not None:
        return all_nodes[node_index]
    gltf_node = gltf.nodes[node_index]
    out = Node(
        name=gltf_node.name or f"node_{node_index}",
        transform=_node_transform(gltf_node),
    )
    all_nodes[node_index] = out
    return out


def _read_inverse_bind_matrices(
    gltf: GLTF2,
    buffers: _LoadedBuffers,
    gltf_skin,
    joint_count: int,
) -> np.ndarray:
    """Read ``inverseBindMatrices`` accessor → ``(J, 4, 4)`` row-major float32.

    Falls back to identity matrices if the accessor is missing.
    """
    if gltf_skin.inverseBindMatrices is None:
        return np.tile(np.eye(4, dtype=np.float32), (joint_count, 1, 1))
    raw = _accessor_array(gltf, buffers, gltf_skin.inverseBindMatrices)
    flat = np.asarray(raw, dtype=np.float32).reshape(-1, 16)
    if flat.shape[0] != joint_count:
        # Truncate or pad with identity if the model is malformed.
        out = np.tile(np.eye(4, dtype=np.float32), (joint_count, 1, 1))
        copy_count = min(flat.shape[0], joint_count)
        # glTF stores column-major; numpy reshape gives row-major. Transpose each.
        out[:copy_count] = flat[:copy_count].reshape(copy_count, 4, 4).transpose(0, 2, 1)
        return out
    matrices = flat.reshape(joint_count, 4, 4).transpose(0, 2, 1)
    return matrices.astype(np.float32, copy=False)


def _attach_skins(
    gltf: GLTF2,
    all_nodes: list[Node | None],
    skins: tuple[Skin, ...],
) -> None:
    """Attach :class:`SkinRefComponent` to every node that references a skin."""
    if not skins:
        return
    for node_index, gltf_node in enumerate(gltf.nodes or ()):
        skin_index = getattr(gltf_node, "skin", None)
        if skin_index is None or skin_index >= len(skins):
            continue
        node = all_nodes[node_index]
        if node is None:
            continue
        node.add_component(SkinRefComponent(skin=skins[skin_index]))


def _attach_spring_chains(skins: tuple[Skin, ...]) -> None:
    """For every skin, detect named chains and tag each chain's anchor node.

    Only chains whose prefix matches a known profile in
    :data:`DEFAULT_CHAIN_PROFILES` get rigged. The chain detector
    catches every ``<prefix><digits>`` joint group, including
    structural bones (``Spine1_M_037``/``Spine2_M_044``) and rig-only
    helpers (twist deformers, weapon attachments) that look like
    chains but should never sway. The strict resolver returns
    ``None`` for those so they're skipped — auto-rigging Spine as
    swing physics would warp the torso every time the body moved.

    Idempotent: skipping anchors that already carry a
    :class:`SpringChainComponent` with the same chain name (so
    re-importing the same scene won't double-attach).
    """
    for skin in skins:
        for chain in detect_chains(skin.joints):
            if _has_chain_component(chain.anchor, chain.name):
                continue
            params = resolve_chain_params_or_none(chain.name)
            if params is None:
                continue
            chain.anchor.add_component(
                SpringChainComponent(
                    chain_name=chain.name,
                    joints=chain.joints,
                    stiffness=params.stiffness,
                    damping=params.damping,
                    inertia=params.inertia,
                )
            )


def _has_chain_component(anchor: Node, chain_name: str) -> bool:
    return any(
        isinstance(c, SpringChainComponent) and c.chain_name == chain_name
        for c in anchor.components
    )


def _node_transform(node) -> Transform:
    if node.matrix:
        # glTF stores matrix column-major; reshape gives row-major, so transpose.
        # Skeleton bone matrices are often skewed/non-Euclidean — keep them raw
        # rather than running them through decompose_trs (which collapses to
        # near-zero scale and zeroes the whole sub-tree).
        matrix = np.asarray(node.matrix, dtype=np.float32).reshape(4, 4).T
        return Transform.from_raw_matrix(matrix)
    translation = np.asarray(node.translation or [0.0, 0.0, 0.0], dtype=np.float32)
    rotation = np.asarray(node.rotation or [0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    scale = np.asarray(node.scale or [1.0, 1.0, 1.0], dtype=np.float32)
    if translation.shape != (3,) or rotation.shape != (4,) or scale.shape != (3,):
        return Transform(
            translation=vec3(0.0, 0.0, 0.0),
            rotation=quat_identity(),
            scale=vec3(1.0, 1.0, 1.0),
        )
    return Transform(translation=translation, rotation=rotation, scale=scale)
