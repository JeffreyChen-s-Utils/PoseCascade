"""PMX 2.0 / 2.1 binary parser.

Walks a ``.pmx`` byte buffer into the format-internal :class:`PmxDocument`.
Every section gets its own helper so per-function complexity stays under the
SonarQube threshold and the spec mapping is obvious at a glance.

The reader does *not* touch disk for textures, materials, or any external
asset reference — it just records the texture file paths the model claims.
The importer adapter (``importer.py``) resolves those paths through
:func:`posecascade.assets.path_safety.resolve_safe`.
"""
from __future__ import annotations

from pmx.encoding import (
    Cursor,
    read_pmx_text,
    read_signed_index,
    read_unsigned_index,
)
from pmx.types import (
    PMX_BONE_FLAG_EXTERNAL_PARENT,
    PMX_BONE_FLAG_FIXED_AXIS,
    PMX_BONE_FLAG_IK,
    PMX_BONE_FLAG_INDEXED_TAIL,
    PMX_BONE_FLAG_INHERIT_ROTATION,
    PMX_BONE_FLAG_INHERIT_TRANSLATION,
    PMX_BONE_FLAG_LOCAL_AXIS,
    PmxBdef1,
    PmxBdef2,
    PmxBdef4,
    PmxBone,
    PmxBoneMorphOffset,
    PmxDeform,
    PmxDeformType,
    PmxDisplayElement,
    PmxDisplayFrame,
    PmxDocument,
    PmxFlipMorphOffset,
    PmxGroupMorphOffset,
    PmxHeader,
    PmxIk,
    PmxIkLink,
    PmxImpulseMorphOffset,
    PmxJoint,
    PmxMaterial,
    PmxMaterialMorphOffset,
    PmxMorph,
    PmxMorphOffset,
    PmxMorphPanel,
    PmxMorphType,
    PmxNames,
    PmxPhysicsMode,
    PmxQdef,
    PmxRigidBody,
    PmxRigidShape,
    PmxSdef,
    PmxSphereMode,
    PmxTextEncoding,
    PmxToonMode,
    PmxUvMorphOffset,
    PmxVertex,
    PmxVertexMorphOffset,
)
from posecascade.errors import MalformedAssetError
from posecascade.render.constants import (
    MAX_PMX_BONE_COUNT,
    MAX_PMX_DISPLAY_FRAME_COUNT,
    MAX_PMX_FACE_COUNT,
    MAX_PMX_JOINT_COUNT,
    MAX_PMX_MATERIAL_COUNT,
    MAX_PMX_MORPH_COUNT,
    MAX_PMX_RIGID_BODY_COUNT,
    MAX_PMX_TEXTURE_COUNT,
    MAX_PMX_VERTEX_COUNT,
)

# PMX magic — the first four bytes of every PMX file. Older toolchains
# wrote ``"Pmx "`` with mixed case, so we accept both spellings.
_PMX_MAGIC = (b"PMX ", b"Pmx ")
_PMX_HEADER_GLOBALS_MIN = 8
_VALID_INDEX_SIZES = (1, 2, 4)
_PMX_VERSION_MIN = 2.0
_PMX_VERSION_MAX = 2.1


def parse_pmx(data: bytes) -> PmxDocument:
    """Parse a complete PMX byte buffer into a :class:`PmxDocument`."""
    cursor = Cursor(data=data)
    header = _read_header(cursor)
    encoding = header.text_encoding
    names = _read_names(cursor, encoding)
    vertices = _read_vertices(cursor, header)
    indices = _read_indices(cursor, header, len(vertices))
    textures = _read_textures(cursor, encoding)
    materials = _read_materials(cursor, header, encoding)
    bones = _read_bones(cursor, header, encoding)
    morphs = _read_morphs(cursor, header, encoding)
    display_frames = _read_display_frames(cursor, header, encoding)
    rigid_bodies = _read_rigid_bodies(cursor, header, encoding)
    joints = _read_joints(cursor, header, encoding)
    return PmxDocument(
        header=header,
        names=names,
        vertices=vertices,
        indices=indices,
        textures=textures,
        materials=materials,
        bones=bones,
        morphs=morphs,
        display_frames=display_frames,
        rigid_bodies=rigid_bodies,
        joints=joints,
    )


# ----- header / names -----
def _read_header(cursor: Cursor) -> PmxHeader:
    magic = cursor.read_bytes(4)
    if magic not in _PMX_MAGIC:
        raise MalformedAssetError(f"not a PMX file (magic={magic!r})")
    version = cursor.read_float()
    if not (_PMX_VERSION_MIN <= version <= _PMX_VERSION_MAX):
        raise MalformedAssetError(f"unsupported PMX version {version}")
    globals_count = cursor.read_uint8()
    if globals_count < _PMX_HEADER_GLOBALS_MIN:
        raise MalformedAssetError(
            f"PMX globals count too small: {globals_count} "
            f"(need at least {_PMX_HEADER_GLOBALS_MIN})"
        )
    globals_bytes = cursor.read_bytes(globals_count)
    return _parse_globals(version, globals_bytes)


def _parse_globals(version: float, globals_bytes: bytes) -> PmxHeader:
    encoding_byte = globals_bytes[0]
    if encoding_byte not in (0, 1):
        raise MalformedAssetError(f"unknown PMX text encoding flag: {encoding_byte}")
    additional_uv_count = globals_bytes[1]
    additional_uv_max = 4
    if additional_uv_count > additional_uv_max:
        raise MalformedAssetError(
            f"additional UV count out of range: {additional_uv_count}"
        )
    sizes = (
        globals_bytes[2], globals_bytes[3], globals_bytes[4],
        globals_bytes[5], globals_bytes[6], globals_bytes[7],
    )
    for label, value in zip(
        ("vertex", "texture", "material", "bone", "morph", "rigid"),
        sizes,
        strict=True,
    ):
        if value not in _VALID_INDEX_SIZES:
            raise MalformedAssetError(f"invalid {label} index size: {value}")
    return PmxHeader(
        version=float(version),
        text_encoding=PmxTextEncoding(encoding_byte),
        additional_uv_count=int(additional_uv_count),
        vertex_index_size=int(sizes[0]),
        texture_index_size=int(sizes[1]),
        material_index_size=int(sizes[2]),
        bone_index_size=int(sizes[3]),
        morph_index_size=int(sizes[4]),
        rigid_index_size=int(sizes[5]),
    )


def _read_names(cursor: Cursor, encoding: PmxTextEncoding) -> PmxNames:
    return PmxNames(
        name_jp=read_pmx_text(cursor, encoding),
        name_en=read_pmx_text(cursor, encoding),
        comment_jp=read_pmx_text(cursor, encoding),
        comment_en=read_pmx_text(cursor, encoding),
    )


# ----- vertices -----
def _read_vertices(cursor: Cursor, header: PmxHeader) -> tuple[PmxVertex, ...]:
    count = cursor.read_int32()
    if count < 0 or count > MAX_PMX_VERTEX_COUNT:
        raise MalformedAssetError(
            f"vertex count out of range: {count} (cap {MAX_PMX_VERTEX_COUNT})"
        )
    vertices: list[PmxVertex] = []
    bone_size = header.bone_index_size
    extra_uv = header.additional_uv_count
    for _ in range(count):
        vertices.append(_read_one_vertex(cursor, bone_size, extra_uv))
    return tuple(vertices)


def _read_one_vertex(cursor: Cursor, bone_size: int, extra_uv: int) -> PmxVertex:
    position = cursor.read_vec3()
    normal = cursor.read_vec3()
    uv = cursor.read_vec2()
    additional_uvs = tuple(cursor.read_vec4() for _ in range(extra_uv))
    deform_type_byte = cursor.read_uint8()
    if deform_type_byte not in PmxDeformType._value2member_map_:
        raise MalformedAssetError(f"unknown deform type {deform_type_byte}")
    deform_type = PmxDeformType(deform_type_byte)
    deform = _read_deform(cursor, deform_type, bone_size)
    edge_ratio = cursor.read_float()
    return PmxVertex(
        position=position,
        normal=normal,
        uv=uv,
        additional_uvs=additional_uvs,
        deform_type=deform_type,
        deform=deform,
        edge_ratio=edge_ratio,
    )


def _read_deform(cursor: Cursor, deform_type: PmxDeformType, bone_size: int) -> PmxDeform:
    if deform_type == PmxDeformType.BDEF1:
        return PmxBdef1(bone=read_signed_index(cursor, bone_size))
    if deform_type == PmxDeformType.BDEF2:
        b1 = read_signed_index(cursor, bone_size)
        b2 = read_signed_index(cursor, bone_size)
        return PmxBdef2(bone1=b1, bone2=b2, weight1=cursor.read_float())
    if deform_type == PmxDeformType.BDEF4:
        bones = (
            read_signed_index(cursor, bone_size),
            read_signed_index(cursor, bone_size),
            read_signed_index(cursor, bone_size),
            read_signed_index(cursor, bone_size),
        )
        weights = (
            cursor.read_float(), cursor.read_float(),
            cursor.read_float(), cursor.read_float(),
        )
        return PmxBdef4(bones=bones, weights=weights)
    if deform_type == PmxDeformType.SDEF:
        b1 = read_signed_index(cursor, bone_size)
        b2 = read_signed_index(cursor, bone_size)
        weight1 = cursor.read_float()
        return PmxSdef(
            bone1=b1, bone2=b2, weight1=weight1,
            c=cursor.read_vec3(), r0=cursor.read_vec3(), r1=cursor.read_vec3(),
        )
    # QDEF (PMX 2.1)
    bones = (
        read_signed_index(cursor, bone_size),
        read_signed_index(cursor, bone_size),
        read_signed_index(cursor, bone_size),
        read_signed_index(cursor, bone_size),
    )
    weights = (
        cursor.read_float(), cursor.read_float(),
        cursor.read_float(), cursor.read_float(),
    )
    return PmxQdef(bones=bones, weights=weights)


# ----- faces -----
def _read_indices(cursor: Cursor, header: PmxHeader, vertex_count: int) -> tuple[int, ...]:
    count = cursor.read_int32()
    if count < 0 or count > MAX_PMX_FACE_COUNT:
        raise MalformedAssetError(
            f"face index count out of range: {count} (cap {MAX_PMX_FACE_COUNT})"
        )
    if count % 3 != 0:
        raise MalformedAssetError(f"face index count {count} is not a multiple of 3")
    size = header.vertex_index_size
    indices: list[int] = []
    for _ in range(count):
        idx = read_unsigned_index(cursor, size)
        if idx >= vertex_count:
            raise MalformedAssetError(
                f"face index {idx} out of range (vertex count {vertex_count})"
            )
        indices.append(idx)
    return tuple(indices)


# ----- textures -----
def _read_textures(cursor: Cursor, encoding: PmxTextEncoding) -> tuple[str, ...]:
    count = cursor.read_int32()
    if count < 0 or count > MAX_PMX_TEXTURE_COUNT:
        raise MalformedAssetError(
            f"texture count out of range: {count} (cap {MAX_PMX_TEXTURE_COUNT})"
        )
    return tuple(read_pmx_text(cursor, encoding) for _ in range(count))


# ----- materials -----
def _read_materials(
    cursor: Cursor, header: PmxHeader, encoding: PmxTextEncoding,
) -> tuple[PmxMaterial, ...]:
    count = cursor.read_int32()
    if count < 0 or count > MAX_PMX_MATERIAL_COUNT:
        raise MalformedAssetError(
            f"material count out of range: {count} (cap {MAX_PMX_MATERIAL_COUNT})"
        )
    return tuple(_read_one_material(cursor, header, encoding) for _ in range(count))


def _read_one_material(
    cursor: Cursor, header: PmxHeader, encoding: PmxTextEncoding,
) -> PmxMaterial:
    name_jp = read_pmx_text(cursor, encoding)
    name_en = read_pmx_text(cursor, encoding)
    diffuse = cursor.read_vec4()
    specular = cursor.read_vec3()
    specular_factor = cursor.read_float()
    ambient = cursor.read_vec3()
    flags = cursor.read_uint8()
    edge_color = cursor.read_vec4()
    edge_size = cursor.read_float()
    texture_index = read_signed_index(cursor, header.texture_index_size)
    sphere_texture_index = read_signed_index(cursor, header.texture_index_size)
    sphere_mode_byte = cursor.read_uint8()
    if sphere_mode_byte not in PmxSphereMode._value2member_map_:
        raise MalformedAssetError(f"unknown sphere mode {sphere_mode_byte}")
    toon_mode_byte = cursor.read_uint8()
    if toon_mode_byte not in PmxToonMode._value2member_map_:
        raise MalformedAssetError(f"unknown toon mode {toon_mode_byte}")
    toon_mode = PmxToonMode(toon_mode_byte)
    if toon_mode == PmxToonMode.EXTERNAL:
        toon_reference = read_signed_index(cursor, header.texture_index_size)
    else:
        toon_reference = cursor.read_uint8()
    memo = read_pmx_text(cursor, encoding)
    face_index_count = cursor.read_int32()
    if face_index_count < 0:
        raise MalformedAssetError(
            f"negative material face index count: {face_index_count}"
        )
    return PmxMaterial(
        name_jp=name_jp, name_en=name_en,
        diffuse=diffuse, specular=specular, specular_factor=specular_factor,
        ambient=ambient, flags=flags, edge_color=edge_color, edge_size=edge_size,
        texture_index=texture_index, sphere_texture_index=sphere_texture_index,
        sphere_mode=PmxSphereMode(sphere_mode_byte), toon_mode=toon_mode,
        toon_reference=toon_reference, memo=memo,
        face_index_count=face_index_count,
    )


# ----- bones -----
def _read_bones(
    cursor: Cursor, header: PmxHeader, encoding: PmxTextEncoding,
) -> tuple[PmxBone, ...]:
    count = cursor.read_int32()
    if count < 0 or count > MAX_PMX_BONE_COUNT:
        raise MalformedAssetError(
            f"bone count out of range: {count} (cap {MAX_PMX_BONE_COUNT})"
        )
    return tuple(_read_one_bone(cursor, header, encoding) for _ in range(count))


def _read_one_bone(
    cursor: Cursor, header: PmxHeader, encoding: PmxTextEncoding,
) -> PmxBone:
    name_jp = read_pmx_text(cursor, encoding)
    name_en = read_pmx_text(cursor, encoding)
    position = cursor.read_vec3()
    parent_index = read_signed_index(cursor, header.bone_index_size)
    deformation_depth = cursor.read_int32()
    flags = cursor.read_uint16()

    tail_offset, tail_bone_index = _read_bone_tail(cursor, header, flags)
    inherit_parent_index, inherit_weight = _read_bone_inherit(cursor, header, flags)
    fixed_axis = cursor.read_vec3() if flags & PMX_BONE_FLAG_FIXED_AXIS else None
    local_x_axis, local_z_axis = _read_bone_local_axes(cursor, flags)
    external_parent_key = (
        cursor.read_int32() if flags & PMX_BONE_FLAG_EXTERNAL_PARENT else None
    )
    ik = _read_bone_ik(cursor, header, flags)

    return PmxBone(
        name_jp=name_jp, name_en=name_en, position=position,
        parent_index=parent_index, deformation_depth=deformation_depth,
        flags=flags, tail_offset=tail_offset, tail_bone_index=tail_bone_index,
        inherit_parent_index=inherit_parent_index, inherit_weight=inherit_weight,
        fixed_axis=fixed_axis, local_x_axis=local_x_axis, local_z_axis=local_z_axis,
        external_parent_key=external_parent_key, ik=ik,
    )


def _read_bone_tail(
    cursor: Cursor, header: PmxHeader, flags: int,
) -> tuple[tuple[float, float, float] | None, int | None]:
    if flags & PMX_BONE_FLAG_INDEXED_TAIL:
        return None, read_signed_index(cursor, header.bone_index_size)
    return cursor.read_vec3(), None


def _read_bone_inherit(
    cursor: Cursor, header: PmxHeader, flags: int,
) -> tuple[int | None, float | None]:
    if flags & (PMX_BONE_FLAG_INHERIT_ROTATION | PMX_BONE_FLAG_INHERIT_TRANSLATION):
        parent = read_signed_index(cursor, header.bone_index_size)
        return parent, cursor.read_float()
    return None, None


def _read_bone_local_axes(
    cursor: Cursor, flags: int,
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    if flags & PMX_BONE_FLAG_LOCAL_AXIS:
        return cursor.read_vec3(), cursor.read_vec3()
    return None, None


def _read_bone_ik(cursor: Cursor, header: PmxHeader, flags: int) -> PmxIk | None:
    if not (flags & PMX_BONE_FLAG_IK):
        return None
    target = read_signed_index(cursor, header.bone_index_size)
    iterations = cursor.read_int32()
    limit_radian = cursor.read_float()
    link_count = cursor.read_int32()
    if link_count < 0:
        raise MalformedAssetError(f"negative IK link count: {link_count}")
    links: list[PmxIkLink] = []
    for _ in range(link_count):
        link_bone = read_signed_index(cursor, header.bone_index_size)
        has_limit = bool(cursor.read_uint8())
        limit_min = cursor.read_vec3() if has_limit else (0.0, 0.0, 0.0)
        limit_max = cursor.read_vec3() if has_limit else (0.0, 0.0, 0.0)
        links.append(
            PmxIkLink(
                bone_index=link_bone, has_limit=has_limit,
                limit_min=limit_min, limit_max=limit_max,
            )
        )
    return PmxIk(
        target_bone_index=target, iterations=iterations,
        limit_radian=limit_radian, links=tuple(links),
    )


# ----- morphs -----
def _read_morphs(
    cursor: Cursor, header: PmxHeader, encoding: PmxTextEncoding,
) -> tuple[PmxMorph, ...]:
    count = cursor.read_int32()
    if count < 0 or count > MAX_PMX_MORPH_COUNT:
        raise MalformedAssetError(
            f"morph count out of range: {count} (cap {MAX_PMX_MORPH_COUNT})"
        )
    return tuple(_read_one_morph(cursor, header, encoding) for _ in range(count))


def _read_one_morph(
    cursor: Cursor, header: PmxHeader, encoding: PmxTextEncoding,
) -> PmxMorph:
    name_jp = read_pmx_text(cursor, encoding)
    name_en = read_pmx_text(cursor, encoding)
    panel_byte = cursor.read_uint8()
    if panel_byte not in PmxMorphPanel._value2member_map_:
        raise MalformedAssetError(f"unknown morph panel {panel_byte}")
    morph_type_byte = cursor.read_uint8()
    if morph_type_byte not in PmxMorphType._value2member_map_:
        raise MalformedAssetError(f"unknown morph type {morph_type_byte}")
    morph_type = PmxMorphType(morph_type_byte)
    offset_count = cursor.read_int32()
    if offset_count < 0:
        raise MalformedAssetError(f"negative morph offset count: {offset_count}")
    offsets = tuple(
        _read_morph_offset(cursor, header, morph_type) for _ in range(offset_count)
    )
    return PmxMorph(
        name_jp=name_jp, name_en=name_en,
        panel=PmxMorphPanel(panel_byte), morph_type=morph_type, offsets=offsets,
    )


_UV_MORPH_TYPES = frozenset({
    PmxMorphType.UV, PmxMorphType.UV1, PmxMorphType.UV2,
    PmxMorphType.UV3, PmxMorphType.UV4,
})


def _read_morph_offset(
    cursor: Cursor, header: PmxHeader, morph_type: PmxMorphType,
) -> PmxMorphOffset:
    """Dispatch to the per-payload reader for ``morph_type``.

    Each branch is its own helper so the dispatcher itself stays linear
    (≤ 1 return) and we don't trip the SonarQube cyclomatic / return count
    rules with a giant if-elif chain.
    """
    if morph_type in _UV_MORPH_TYPES:
        return _read_uv_morph_offset(cursor, header)
    payload_reader = _MORPH_OFFSET_READERS.get(morph_type)
    if payload_reader is None:                    # IMPULSE / FLIP / unrecognised
        return _read_misc_morph_offset(cursor, header, morph_type)
    return payload_reader(cursor, header)


def _read_group_morph_offset(cursor: Cursor, header: PmxHeader) -> PmxGroupMorphOffset:
    return PmxGroupMorphOffset(
        morph_index=read_signed_index(cursor, header.morph_index_size),
        weight=cursor.read_float(),
    )


def _read_vertex_morph_offset(cursor: Cursor, header: PmxHeader) -> PmxVertexMorphOffset:
    return PmxVertexMorphOffset(
        vertex_index=read_unsigned_index(cursor, header.vertex_index_size),
        offset=cursor.read_vec3(),
    )


def _read_bone_morph_offset(cursor: Cursor, header: PmxHeader) -> PmxBoneMorphOffset:
    return PmxBoneMorphOffset(
        bone_index=read_signed_index(cursor, header.bone_index_size),
        translation=cursor.read_vec3(),
        rotation=cursor.read_vec4(),
    )


def _read_uv_morph_offset(cursor: Cursor, header: PmxHeader) -> PmxUvMorphOffset:
    return PmxUvMorphOffset(
        vertex_index=read_unsigned_index(cursor, header.vertex_index_size),
        offset=cursor.read_vec4(),
    )


def _read_misc_morph_offset(
    cursor: Cursor, header: PmxHeader, morph_type: PmxMorphType,
) -> PmxMorphOffset:
    """Handle FLIP and IMPULSE — the two PMX 2.1-only morph payloads."""
    if morph_type == PmxMorphType.FLIP:
        return PmxFlipMorphOffset(
            morph_index=read_signed_index(cursor, header.morph_index_size),
            weight=cursor.read_float(),
        )
    return PmxImpulseMorphOffset(
        rigid_body_index=read_signed_index(cursor, header.rigid_index_size),
        is_local=bool(cursor.read_uint8()),
        velocity=cursor.read_vec3(),
        torque=cursor.read_vec3(),
    )


def _read_material_morph_offset(cursor: Cursor, header: PmxHeader) -> PmxMaterialMorphOffset:
    return PmxMaterialMorphOffset(
        material_index=read_signed_index(cursor, header.material_index_size),
        op=cursor.read_uint8(),
        diffuse=cursor.read_vec4(),
        specular=cursor.read_vec3(),
        specular_factor=cursor.read_float(),
        ambient=cursor.read_vec3(),
        edge_color=cursor.read_vec4(),
        edge_size=cursor.read_float(),
        texture_coef=cursor.read_vec4(),
        sphere_coef=cursor.read_vec4(),
        toon_coef=cursor.read_vec4(),
    )


_MORPH_OFFSET_READERS = {
    PmxMorphType.GROUP: _read_group_morph_offset,
    PmxMorphType.VERTEX: _read_vertex_morph_offset,
    PmxMorphType.BONE: _read_bone_morph_offset,
    PmxMorphType.MATERIAL: _read_material_morph_offset,
}


# ----- display frames -----
def _read_display_frames(
    cursor: Cursor, header: PmxHeader, encoding: PmxTextEncoding,
) -> tuple[PmxDisplayFrame, ...]:
    count = cursor.read_int32()
    if count < 0 or count > MAX_PMX_DISPLAY_FRAME_COUNT:
        raise MalformedAssetError(
            f"display-frame count out of range: {count} "
            f"(cap {MAX_PMX_DISPLAY_FRAME_COUNT})"
        )
    return tuple(_read_one_display_frame(cursor, header, encoding) for _ in range(count))


def _read_one_display_frame(
    cursor: Cursor, header: PmxHeader, encoding: PmxTextEncoding,
) -> PmxDisplayFrame:
    name_jp = read_pmx_text(cursor, encoding)
    name_en = read_pmx_text(cursor, encoding)
    is_special = bool(cursor.read_uint8())
    element_count = cursor.read_int32()
    if element_count < 0:
        raise MalformedAssetError(f"negative display-frame element count: {element_count}")
    elements: list[PmxDisplayElement] = []
    for _ in range(element_count):
        kind = cursor.read_uint8()
        if kind == 0:
            index = read_signed_index(cursor, header.bone_index_size)
        elif kind == 1:
            index = read_signed_index(cursor, header.morph_index_size)
        else:
            raise MalformedAssetError(f"unknown display element kind {kind}")
        elements.append(PmxDisplayElement(kind=kind, index=index))
    return PmxDisplayFrame(
        name_jp=name_jp, name_en=name_en, is_special=is_special,
        elements=tuple(elements),
    )


# ----- rigid bodies -----
def _read_rigid_bodies(
    cursor: Cursor, header: PmxHeader, encoding: PmxTextEncoding,
) -> tuple[PmxRigidBody, ...]:
    if cursor.remaining() == 0:
        return ()
    count = cursor.read_int32()
    if count < 0 or count > MAX_PMX_RIGID_BODY_COUNT:
        raise MalformedAssetError(
            f"rigid body count out of range: {count} (cap {MAX_PMX_RIGID_BODY_COUNT})"
        )
    return tuple(_read_one_rigid_body(cursor, header, encoding) for _ in range(count))


def _read_one_rigid_body(
    cursor: Cursor, header: PmxHeader, encoding: PmxTextEncoding,
) -> PmxRigidBody:
    name_jp = read_pmx_text(cursor, encoding)
    name_en = read_pmx_text(cursor, encoding)
    related_bone = read_signed_index(cursor, header.bone_index_size)
    group = cursor.read_uint8()
    non_collision_mask = cursor.read_uint16()
    shape_byte = cursor.read_uint8()
    if shape_byte not in PmxRigidShape._value2member_map_:
        raise MalformedAssetError(f"unknown rigid shape {shape_byte}")
    size = cursor.read_vec3()
    position = cursor.read_vec3()
    rotation = cursor.read_vec3()
    mass = cursor.read_float()
    linear_damping = cursor.read_float()
    angular_damping = cursor.read_float()
    restitution = cursor.read_float()
    friction = cursor.read_float()
    physics_byte = cursor.read_uint8()
    if physics_byte not in PmxPhysicsMode._value2member_map_:
        raise MalformedAssetError(f"unknown physics mode {physics_byte}")
    return PmxRigidBody(
        name_jp=name_jp, name_en=name_en, related_bone_index=related_bone,
        group=group, non_collision_mask=non_collision_mask,
        shape=PmxRigidShape(shape_byte), size=size, position=position,
        rotation=rotation, mass=mass, linear_damping=linear_damping,
        angular_damping=angular_damping, restitution=restitution,
        friction=friction, physics_mode=PmxPhysicsMode(physics_byte),
    )


# ----- joints -----
def _read_joints(
    cursor: Cursor, header: PmxHeader, encoding: PmxTextEncoding,
) -> tuple[PmxJoint, ...]:
    if cursor.remaining() == 0:
        return ()
    count = cursor.read_int32()
    if count < 0 or count > MAX_PMX_JOINT_COUNT:
        raise MalformedAssetError(
            f"joint count out of range: {count} (cap {MAX_PMX_JOINT_COUNT})"
        )
    return tuple(_read_one_joint(cursor, header, encoding) for _ in range(count))


def _read_one_joint(
    cursor: Cursor, header: PmxHeader, encoding: PmxTextEncoding,
) -> PmxJoint:
    name_jp = read_pmx_text(cursor, encoding)
    name_en = read_pmx_text(cursor, encoding)
    joint_type = cursor.read_uint8()
    rigid_a = read_signed_index(cursor, header.rigid_index_size)
    rigid_b = read_signed_index(cursor, header.rigid_index_size)
    position = cursor.read_vec3()
    rotation = cursor.read_vec3()
    linear_lower = cursor.read_vec3()
    linear_upper = cursor.read_vec3()
    angular_lower = cursor.read_vec3()
    angular_upper = cursor.read_vec3()
    spring_linear = cursor.read_vec3()
    spring_angular = cursor.read_vec3()
    return PmxJoint(
        name_jp=name_jp, name_en=name_en, joint_type=joint_type,
        rigid_a_index=rigid_a, rigid_b_index=rigid_b,
        position=position, rotation=rotation,
        linear_lower=linear_lower, linear_upper=linear_upper,
        angular_lower=angular_lower, angular_upper=angular_upper,
        spring_linear=spring_linear, spring_angular=spring_angular,
    )
