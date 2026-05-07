"""PMD (legacy MikuMikuDance Polygon Model Data) binary parser.

PMD predates PMX. The format is fixed-width records with Shift-JIS text and
no per-index byte-size globals — every index is a ``uint16`` and the
"no parent" sentinel is ``0xFFFF``. The reader emits the same
:class:`~pmx.types.PmxDocument` so the rest of the importer adapter does
not need to know which dialect produced the data.

Phase 1 supports the core sections (header / vertices / faces / textures
expressed as material file names / materials / bones). PMD-specific IK,
morphs, and physics still parse via :func:`parse_pmd` so the file is
walked end-to-end, but they are converted to the PMX schema where the
mapping is unambiguous and dropped where it is not.
"""
from __future__ import annotations

from pmx.encoding import Cursor, read_pmd_text
from pmx.types import (
    PmxBdef2,
    PmxBone,
    PmxDocument,
    PmxHeader,
    PmxMaterial,
    PmxNames,
    PmxSphereMode,
    PmxTextEncoding,
    PmxToonMode,
    PmxVertex,
)

from posecascade.errors import MalformedAssetError
from posecascade.render.constants import (
    MAX_PMX_BONE_COUNT,
    MAX_PMX_MATERIAL_COUNT,
    MAX_PMX_VERTEX_COUNT,
)

_PMD_MAGIC = b"Pmd"
_PMD_VERSION_SUPPORTED = (1.0,)
_PMD_NAME_BYTES = 20
_PMD_COMMENT_BYTES = 256
_PMD_TEXTURE_BYTES = 20
_PMD_BONE_NAME_BYTES = 20
_PMD_PARENT_NONE = 0xFFFF


def parse_pmd(data: bytes) -> PmxDocument:
    """Parse a PMD byte buffer into the unified :class:`PmxDocument`.

    The vertex / face / material / bone counts are validated against the
    same hard caps that PMX uses (PMD models are smaller in practice, so
    nothing PMD-shaped will ever bump up against them).
    """
    cursor = Cursor(data=data)
    _read_header(cursor)
    names = _read_names(cursor)
    vertices = _read_vertices(cursor)
    indices = _read_indices(cursor, len(vertices))
    materials, textures = _read_materials(cursor)
    bones = _read_bones(cursor)
    return PmxDocument(
        header=PmxHeader(
            version=2.0,
            text_encoding=PmxTextEncoding.UTF8,   # synthetic — PMD is SJIS, but the
                                                  # PmxDocument consumer only inspects
                                                  # text fields, never the flag.
            additional_uv_count=0,
            vertex_index_size=4,
            texture_index_size=4,
            material_index_size=4,
            bone_index_size=4,
            morph_index_size=4,
            rigid_index_size=4,
        ),
        names=names,
        vertices=vertices,
        indices=indices,
        textures=textures,
        materials=materials,
        bones=bones,
    )


def _read_header(cursor: Cursor) -> None:
    magic = cursor.read_bytes(len(_PMD_MAGIC))
    if magic != _PMD_MAGIC:
        raise MalformedAssetError(f"not a PMD file (magic={magic!r})")
    version = cursor.read_float()
    if version not in _PMD_VERSION_SUPPORTED:
        raise MalformedAssetError(f"unsupported PMD version {version}")


def _read_names(cursor: Cursor) -> PmxNames:
    name = read_pmd_text(cursor, _PMD_NAME_BYTES)
    comment = read_pmd_text(cursor, _PMD_COMMENT_BYTES)
    return PmxNames(name_jp=name, name_en=name, comment_jp=comment, comment_en=comment)


def _read_vertices(cursor: Cursor) -> tuple[PmxVertex, ...]:
    count = cursor.read_uint32()
    if count > MAX_PMX_VERTEX_COUNT:
        raise MalformedAssetError(
            f"vertex count out of range: {count} (cap {MAX_PMX_VERTEX_COUNT})"
        )
    out: list[PmxVertex] = []
    for _ in range(count):
        position = cursor.read_vec3()
        normal = cursor.read_vec3()
        uv = cursor.read_vec2()
        bone1 = cursor.read_uint16()
        bone2 = cursor.read_uint16()
        weight_byte = cursor.read_uint8()
        cursor.read_uint8()                    # edge flag — ignored for Phase 1
        weight1 = float(weight_byte) / 100.0
        out.append(
            PmxVertex(
                position=position, normal=normal, uv=uv, additional_uvs=(),
                deform_type=_pmd_deform_type(bone1, bone2),
                deform=PmxBdef2(bone1=bone1, bone2=bone2, weight1=weight1),
                edge_ratio=1.0,
            )
        )
    return tuple(out)


def _pmd_deform_type(bone1: int, bone2: int):  # noqa: ANN202 — tiny local helper
    from pmx.types import PmxDeformType  # noqa: PLC0415 — avoid circular at module load
    if bone1 == bone2:
        return PmxDeformType.BDEF1
    return PmxDeformType.BDEF2


def _read_indices(cursor: Cursor, vertex_count: int) -> tuple[int, ...]:
    count = cursor.read_uint32()
    if count % 3 != 0:
        raise MalformedAssetError(f"face index count {count} is not a multiple of 3")
    out: list[int] = []
    for _ in range(count):
        idx = cursor.read_uint16()
        if idx >= vertex_count:
            raise MalformedAssetError(
                f"face index {idx} out of range (vertex count {vertex_count})"
            )
        out.append(idx)
    return tuple(out)


def _read_materials(cursor: Cursor) -> tuple[tuple[PmxMaterial, ...], tuple[str, ...]]:
    """Decode PMD materials and the implicit textures they reference.

    PMD inlines the diffuse texture filename inside each material record, so
    we collect them into a deduplicated texture table and rewrite each
    material's ``texture_index`` to point into it. The result mirrors how
    PMX exposes textures.
    """
    count = cursor.read_uint32()
    if count > MAX_PMX_MATERIAL_COUNT:
        raise MalformedAssetError(
            f"material count out of range: {count} (cap {MAX_PMX_MATERIAL_COUNT})"
        )
    materials: list[PmxMaterial] = []
    textures: list[str] = []
    texture_lookup: dict[str, int] = {}
    for _ in range(count):
        mat, tex_path = _read_one_material(cursor)
        if tex_path:
            tex_index = texture_lookup.get(tex_path)
            if tex_index is None:
                tex_index = len(textures)
                textures.append(tex_path)
                texture_lookup[tex_path] = tex_index
            materials.append(_with_texture_index(mat, tex_index))
        else:
            materials.append(mat)
    return tuple(materials), tuple(textures)


def _read_one_material(cursor: Cursor) -> tuple[PmxMaterial, str]:
    """Parse one PMD material record + return the diffuse texture path it
    referenced (empty string when the material has none)."""
    diffuse = cursor.read_vec4()
    specular_factor = cursor.read_float()
    specular = cursor.read_vec3()
    ambient = cursor.read_vec3()
    toon_index = cursor.read_uint8()
    cursor.read_uint8()                    # edge flag — ignored
    face_index_count = cursor.read_uint32()
    raw_texture = read_pmd_text(cursor, _PMD_TEXTURE_BYTES)
    diffuse_texture = _split_pmd_texture(raw_texture)
    mat = PmxMaterial(
        name_jp="", name_en="",
        diffuse=diffuse, specular=specular, specular_factor=specular_factor,
        ambient=ambient, flags=0, edge_color=(0.0, 0.0, 0.0, 1.0), edge_size=1.0,
        texture_index=-1, sphere_texture_index=-1,
        sphere_mode=PmxSphereMode.DISABLED,
        toon_mode=PmxToonMode.INTERNAL, toon_reference=int(toon_index),
        memo="", face_index_count=int(face_index_count),
    )
    return mat, diffuse_texture


def _split_pmd_texture(raw: str) -> str:
    """Strip the optional ``"*sphere"`` suffix PMD uses to chain a sphere map.

    Phase 1 only consumes the diffuse half — sphere textures are revisited
    in Phase 2 (toon render path). Returning the empty string when no
    diffuse is present is fine: the importer will fall back to an untextured
    material.
    """
    if "*" in raw:
        diffuse, _, _sphere = raw.partition("*")
        return diffuse
    return raw


def _with_texture_index(mat: PmxMaterial, texture_index: int) -> PmxMaterial:
    """Return a copy of ``mat`` with ``texture_index`` set."""
    return PmxMaterial(
        name_jp=mat.name_jp, name_en=mat.name_en,
        diffuse=mat.diffuse, specular=mat.specular,
        specular_factor=mat.specular_factor, ambient=mat.ambient,
        flags=mat.flags, edge_color=mat.edge_color, edge_size=mat.edge_size,
        texture_index=texture_index, sphere_texture_index=mat.sphere_texture_index,
        sphere_mode=mat.sphere_mode, toon_mode=mat.toon_mode,
        toon_reference=mat.toon_reference, memo=mat.memo,
        face_index_count=mat.face_index_count,
    )


def _read_bones(cursor: Cursor) -> tuple[PmxBone, ...]:
    count = cursor.read_uint16()
    if count > MAX_PMX_BONE_COUNT:
        raise MalformedAssetError(
            f"bone count out of range: {count} (cap {MAX_PMX_BONE_COUNT})"
        )
    out: list[PmxBone] = []
    for _ in range(count):
        out.append(_read_one_bone(cursor))
    return tuple(out)


def _read_one_bone(cursor: Cursor) -> PmxBone:
    name = read_pmd_text(cursor, _PMD_BONE_NAME_BYTES)
    parent = cursor.read_uint16()
    cursor.read_uint16()                    # tail bone — discarded for Phase 1
    cursor.read_uint8()                     # type — discarded for Phase 1
    cursor.read_uint16()                    # IK index — discarded for Phase 1
    position = cursor.read_vec3()
    parent_index = -1 if parent == _PMD_PARENT_NONE else int(parent)
    return PmxBone(
        name_jp=name, name_en=name, position=position,
        parent_index=parent_index, deformation_depth=0,
        flags=0x001E,                       # rotatable | translatable | visible | enabled
        tail_offset=(0.0, 0.0, 0.0), tail_bone_index=None,
        inherit_parent_index=None, inherit_weight=None,
        fixed_axis=None, local_x_axis=None, local_z_axis=None,
        external_parent_key=None, ik=None,
    )
