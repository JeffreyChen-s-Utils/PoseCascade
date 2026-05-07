"""Build tiny PMX / PMD byte buffers for parser tests.

Pure-stdlib (``struct`` only) — deliberately does NOT import the importer's
own encoding helpers, so a parser bug cannot silently agree with a writer
bug. Anything outside of test code that wants a PMX should go through the
real importer; this module is a test helper, not a public API.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field, replace

# ----- PMX globals -----
# We pin all index sizes to 1 byte (signed, range -128..127) — sufficient for
# the tiny test models — and the unsigned vertex index width to 2 bytes so we
# also exercise that branch of the reader.
_VERTEX_INDEX_SIZE = 2
_TEXTURE_INDEX_SIZE = 1
_MATERIAL_INDEX_SIZE = 1
_BONE_INDEX_SIZE = 1
_MORPH_INDEX_SIZE = 1
_RIGID_INDEX_SIZE = 1


def _u8(value: int) -> bytes:
    return struct.pack("<B", value & 0xFF)


def _u16(value: int) -> bytes:
    return struct.pack("<H", value & 0xFFFF)


def _i32(value: int) -> bytes:
    return struct.pack("<i", value)


def _f(value: float) -> bytes:
    return struct.pack("<f", value)


def _v3(v: tuple[float, float, float]) -> bytes:
    return struct.pack("<fff", *v)


def _v4(v: tuple[float, float, float, float]) -> bytes:
    return struct.pack("<ffff", *v)


def _signed_index(value: int, size: int) -> bytes:
    if size == 1:
        return struct.pack("<b", value)
    if size == 2:
        return struct.pack("<h", value)
    return struct.pack("<i", value)


def _unsigned_index(value: int, size: int) -> bytes:
    if size == 1:
        return struct.pack("<B", value)
    if size == 2:
        return struct.pack("<H", value)
    return struct.pack("<I", value)


def _pmx_text_utf8(text: str) -> bytes:
    raw = text.encode("utf-8")
    return _i32(len(raw)) + raw


def _pmx_text_utf16(text: str) -> bytes:
    raw = text.encode("utf-16-le")
    return _i32(len(raw)) + raw


# ----- vertex skinning payloads -----
@dataclass(frozen=True)
class _Bdef1:
    bone: int


@dataclass(frozen=True)
class _Bdef2:
    bone1: int
    bone2: int
    weight1: float


@dataclass(frozen=True)
class _Bdef4:
    bones: tuple[int, int, int, int]
    weights: tuple[float, float, float, float]


@dataclass(frozen=True)
class _Sdef:
    bone1: int
    bone2: int
    weight1: float
    c: tuple[float, float, float]
    r0: tuple[float, float, float]
    r1: tuple[float, float, float]


_DeformPayload = _Bdef1 | _Bdef2 | _Bdef4 | _Sdef


def _deform_byte(deform: _DeformPayload) -> int:
    if isinstance(deform, _Bdef1):
        return 0
    if isinstance(deform, _Bdef2):
        return 1
    if isinstance(deform, _Bdef4):
        return 2
    return 3


def _encode_deform(deform: _DeformPayload, bone_size: int) -> bytes:
    if isinstance(deform, _Bdef1):
        return _signed_index(deform.bone, bone_size)
    if isinstance(deform, _Bdef2):
        return (
            _signed_index(deform.bone1, bone_size)
            + _signed_index(deform.bone2, bone_size)
            + _f(deform.weight1)
        )
    if isinstance(deform, _Bdef4):
        return (
            b"".join(_signed_index(b, bone_size) for b in deform.bones)
            + b"".join(_f(w) for w in deform.weights)
        )
    return (
        _signed_index(deform.bone1, bone_size)
        + _signed_index(deform.bone2, bone_size)
        + _f(deform.weight1)
        + _v3(deform.c) + _v3(deform.r0) + _v3(deform.r1)
    )


@dataclass(frozen=True)
class FixtureVertex:
    position: tuple[float, float, float]
    normal: tuple[float, float, float] = (0.0, 1.0, 0.0)
    uv: tuple[float, float] = (0.0, 0.0)
    deform: _DeformPayload = field(default_factory=lambda: _Bdef1(bone=0))
    edge_ratio: float = 1.0


@dataclass(frozen=True)
class FixtureIkLink:
    bone_index: int
    has_limit: bool = False
    limit_min: tuple[float, float, float] = (0.0, 0.0, 0.0)   # XYZ radians
    limit_max: tuple[float, float, float] = (0.0, 0.0, 0.0)   # XYZ radians


@dataclass(frozen=True)
class FixtureIk:
    """One PMX IK definition. ``target_bone_index`` is the *effector* — the
    bone whose world position should match the IK-owning bone after
    solving."""

    target_bone_index: int
    iterations: int = 10
    limit_radian: float = 0.0
    links: tuple[FixtureIkLink, ...] = ()


@dataclass(frozen=True)
class FixtureBone:
    name_jp: str
    position: tuple[float, float, float]
    parent_index: int = -1
    ik: FixtureIk | None = None
    inherit_parent_index: int | None = None
    inherit_weight: float = 1.0
    inherit_rotation: bool = False
    inherit_translation: bool = False
    fixed_axis: tuple[float, float, float] | None = None
    deformation_depth: int = 0


@dataclass(frozen=True)
class FixtureMaterial:
    name_jp: str = "mat"
    diffuse: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    texture_index: int = -1
    sphere_texture_index: int = -1
    sphere_mode: int = 0
    toon_mode: int = 1   # internal toon
    toon_reference: int = 0
    face_index_count: int = 0
    flags: int = 0
    edge_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    edge_size: float = 1.0


@dataclass(frozen=True)
class FixtureVertexMorphOffset:
    vertex_index: int
    offset: tuple[float, float, float]


@dataclass(frozen=True)
class FixtureBoneMorphOffset:
    bone_index: int
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True)
class FixtureUvMorphOffset:
    vertex_index: int
    offset: tuple[float, float, float, float]


@dataclass(frozen=True)
class FixtureMaterialMorphTarget:
    material_index: int
    op: int   # 0=multiply, 1=add
    diffuse: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    specular: tuple[float, float, float] = (0.0, 0.0, 0.0)
    specular_power: float = 0.0
    ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)
    edge_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    edge_size: float = 0.0
    texture_coef: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    sphere_coef: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    toon_coef: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class FixtureGroupChild:
    morph_index: int
    weight: float


@dataclass(frozen=True)
class FixtureMorph:
    """One morph entry — ``morph_type`` selects which of the offset tuples
    is encoded. Tests that only care about one type leave the others empty.
    """

    name: str
    morph_type: int
    panel: int = 4
    vertex_offsets: tuple[FixtureVertexMorphOffset, ...] = ()
    bone_offsets: tuple[FixtureBoneMorphOffset, ...] = ()
    uv_offsets: tuple[FixtureUvMorphOffset, ...] = ()
    material_targets: tuple[FixtureMaterialMorphTarget, ...] = ()
    group_children: tuple[FixtureGroupChild, ...] = ()
    flip_children: tuple[FixtureGroupChild, ...] = ()


@dataclass(frozen=True)
class FixtureDisplayElement:
    kind: int     # 0 = bone, 1 = morph
    index: int


@dataclass(frozen=True)
class FixtureDisplayFrame:
    name: str
    is_special: bool = False
    elements: tuple[FixtureDisplayElement, ...] = ()


@dataclass(frozen=True)
class FixtureRigidBody:
    """One PMX rigid body. Defaults match a small, lightly-damped sphere
    that the simulator can drop reasonably in unit tests."""

    name: str
    bone_index: int = -1
    group: int = 0
    non_collision_mask: int = 0
    shape: int = 0                                  # 0=sphere, 1=box, 2=capsule
    size: tuple[float, float, float] = (0.5, 0.0, 0.0)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 1.0
    linear_damping: float = 0.05
    angular_damping: float = 0.05
    restitution: float = 0.0
    friction: float = 0.5
    physics_mode: int = 1                           # 1 = dynamic


@dataclass(frozen=True)
class FixtureJoint:
    """One PMX 6DOF spring joint."""

    name: str
    rigid_a: int
    rigid_b: int
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_lower: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_upper: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_lower: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_upper: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spring_linear: tuple[float, float, float] = (50.0, 50.0, 50.0)
    spring_angular: tuple[float, float, float] = (5.0, 5.0, 5.0)


@dataclass(frozen=True)
class FixtureBuild:
    """All knobs for a tiny synthetic PMX. Fields default to the canonical
    ``tiny.pmx`` shape used as a happy-path fixture."""

    encoding_byte: int = 1   # 0=UTF-16-LE, 1=UTF-8
    version: float = 2.0
    name_jp: str = "tiny"
    name_en: str = "tiny"
    comment_jp: str = ""
    comment_en: str = ""
    vertices: tuple[FixtureVertex, ...] = ()
    indices: tuple[int, ...] = ()
    textures: tuple[str, ...] = ()
    materials: tuple[FixtureMaterial, ...] = ()
    bones: tuple[FixtureBone, ...] = ()
    morphs: tuple[FixtureMorph, ...] = ()
    rigid_bodies: tuple[FixtureRigidBody, ...] = ()
    joints: tuple[FixtureJoint, ...] = ()
    display_frames: tuple[FixtureDisplayFrame, ...] = ()


def _encode_text(text: str, encoding_byte: int) -> bytes:
    return _pmx_text_utf8(text) if encoding_byte == 1 else _pmx_text_utf16(text)


def _encode_header(spec: FixtureBuild) -> bytes:
    magic = b"PMX "
    return (
        magic
        + _f(spec.version)
        + _u8(8)
        + _u8(spec.encoding_byte)
        + _u8(0)                         # additional UV count
        + _u8(_VERTEX_INDEX_SIZE)
        + _u8(_TEXTURE_INDEX_SIZE)
        + _u8(_MATERIAL_INDEX_SIZE)
        + _u8(_BONE_INDEX_SIZE)
        + _u8(_MORPH_INDEX_SIZE)
        + _u8(_RIGID_INDEX_SIZE)
    )


def _encode_names(spec: FixtureBuild) -> bytes:
    return (
        _encode_text(spec.name_jp, spec.encoding_byte)
        + _encode_text(spec.name_en, spec.encoding_byte)
        + _encode_text(spec.comment_jp, spec.encoding_byte)
        + _encode_text(spec.comment_en, spec.encoding_byte)
    )


def _encode_vertices(spec: FixtureBuild) -> bytes:
    out = [_i32(len(spec.vertices))]
    for v in spec.vertices:
        out.append(_v3(v.position))
        out.append(_v3(v.normal))
        out.append(struct.pack("<ff", *v.uv))
        out.append(_u8(_deform_byte(v.deform)))
        out.append(_encode_deform(v.deform, _BONE_INDEX_SIZE))
        out.append(_f(v.edge_ratio))
    return b"".join(out)


def _encode_indices(spec: FixtureBuild) -> bytes:
    out = [_i32(len(spec.indices))]
    for i in spec.indices:
        out.append(_unsigned_index(i, _VERTEX_INDEX_SIZE))
    return b"".join(out)


def _encode_textures(spec: FixtureBuild) -> bytes:
    out = [_i32(len(spec.textures))]
    for tex in spec.textures:
        out.append(_encode_text(tex, spec.encoding_byte))
    return b"".join(out)


def _encode_materials(spec: FixtureBuild) -> bytes:
    out = [_i32(len(spec.materials))]
    for mat in spec.materials:
        out.append(_encode_one_material(mat, spec.encoding_byte))
    return b"".join(out)


def _encode_one_material(mat: FixtureMaterial, encoding_byte: int) -> bytes:
    chunks = [
        _encode_text(mat.name_jp, encoding_byte),
        _encode_text(mat.name_jp, encoding_byte),         # name_en — reuse JP for brevity
        _v4(mat.diffuse),
        _v3((0.0, 0.0, 0.0)),                             # specular
        _f(0.0),                                          # specular factor
        _v3((0.5, 0.5, 0.5)),                             # ambient
        _u8(mat.flags),
        _v4(mat.edge_color),
        _f(mat.edge_size),
        _signed_index(mat.texture_index, _TEXTURE_INDEX_SIZE),
        _signed_index(mat.sphere_texture_index, _TEXTURE_INDEX_SIZE),
        _u8(mat.sphere_mode),
        _u8(mat.toon_mode),
    ]
    if mat.toon_mode == 0:
        chunks.append(_signed_index(mat.toon_reference, _TEXTURE_INDEX_SIZE))
    else:
        chunks.append(_u8(mat.toon_reference))
    chunks.append(_encode_text("", encoding_byte))        # memo
    chunks.append(_i32(mat.face_index_count))
    return b"".join(chunks)


_BONE_FLAG_BASE = 0x001E   # rotatable | translatable | visible | enabled
_BONE_FLAG_IK = 0x0020
_BONE_FLAG_INHERIT_ROTATION = 0x0100
_BONE_FLAG_INHERIT_TRANSLATION = 0x0200
_BONE_FLAG_FIXED_AXIS = 0x0400


def _encode_bones(spec: FixtureBuild) -> bytes:
    """Encode the bone section, including optional IK / append / fixed-axis blocks."""
    out = [_i32(len(spec.bones))]
    for bone in spec.bones:
        out.append(_encode_text(bone.name_jp, spec.encoding_byte))
        out.append(_encode_text(bone.name_jp, spec.encoding_byte))   # name_en
        out.append(_v3(bone.position))
        out.append(_signed_index(bone.parent_index, _BONE_INDEX_SIZE))
        out.append(_i32(int(bone.deformation_depth)))
        out.append(_u16(_compute_bone_flags(bone)))
        out.append(_v3((0.0, 1.0, 0.0)))                             # tail offset
        if bone.inherit_rotation or bone.inherit_translation:
            parent = -1 if bone.inherit_parent_index is None else int(bone.inherit_parent_index)
            out.append(_signed_index(parent, _BONE_INDEX_SIZE))
            out.append(_f(bone.inherit_weight))
        if bone.fixed_axis is not None:
            out.append(_v3(bone.fixed_axis))
        if bone.ik is not None:
            out.append(_encode_ik_block(bone.ik))
    return b"".join(out)


def _compute_bone_flags(bone: FixtureBone) -> int:
    flags = _BONE_FLAG_BASE
    if bone.ik is not None:
        flags |= _BONE_FLAG_IK
    if bone.inherit_rotation:
        flags |= _BONE_FLAG_INHERIT_ROTATION
    if bone.inherit_translation:
        flags |= _BONE_FLAG_INHERIT_TRANSLATION
    if bone.fixed_axis is not None:
        flags |= _BONE_FLAG_FIXED_AXIS
    return flags


def _encode_ik_block(ik: FixtureIk) -> bytes:
    chunks = [
        _signed_index(ik.target_bone_index, _BONE_INDEX_SIZE),
        _i32(ik.iterations),
        _f(ik.limit_radian),
        _i32(len(ik.links)),
    ]
    for link in ik.links:
        chunks.append(_signed_index(link.bone_index, _BONE_INDEX_SIZE))
        chunks.append(_u8(1 if link.has_limit else 0))
        if link.has_limit:
            chunks.append(_v3(link.limit_min))
            chunks.append(_v3(link.limit_max))
    return b"".join(chunks)


def _encode_empty_section() -> bytes:
    return _i32(0)


def _encode_morphs(spec: FixtureBuild) -> bytes:
    """Encode the morph section with one record per :class:`FixtureMorph`."""
    out = [_i32(len(spec.morphs))]
    for morph in spec.morphs:
        out.append(_encode_one_morph(morph, spec.encoding_byte))
    return b"".join(out)


def _encode_one_morph(morph: FixtureMorph, encoding_byte: int) -> bytes:
    chunks = [
        _encode_text(morph.name, encoding_byte),
        _encode_text(morph.name, encoding_byte),    # name_en — reuse JP for brevity
        _u8(morph.panel),
        _u8(morph.morph_type),
    ]
    payload, count = _encode_morph_offsets(morph)
    chunks.append(_i32(count))
    chunks.append(payload)
    return b"".join(chunks)


_UV_MORPH_TYPES_FIXTURE = (3, 4, 5, 6, 7)


def _encode_morph_offsets(morph: FixtureMorph) -> tuple[bytes, int]:
    """Dispatch on morph type to encode the right per-offset payload + count.

    UV channels (3..7) share one encoder; everything else flows through
    :data:`_FIXTURE_MORPH_ENCODERS` so this dispatcher stays linear.
    """
    if morph.morph_type in _UV_MORPH_TYPES_FIXTURE:
        return _encode_uv_offsets(morph.uv_offsets)
    encoder = _FIXTURE_MORPH_ENCODERS.get(morph.morph_type)
    if encoder is None:
        return b"", 0          # IMPULSE / unsupported in fixtures
    return encoder(morph)


_FIXTURE_MORPH_ENCODERS = {
    0: lambda m: _encode_group_children(m.group_children),
    1: lambda m: _encode_vertex_offsets(m.vertex_offsets),
    2: lambda m: _encode_bone_offsets(m.bone_offsets),
    8: lambda m: _encode_material_targets(m.material_targets),
    9: lambda m: _encode_group_children(m.flip_children),
}


def _encode_group_children(children: tuple[FixtureGroupChild, ...]) -> tuple[bytes, int]:
    return b"".join(
        _signed_index(child.morph_index, _MORPH_INDEX_SIZE) + _f(child.weight)
        for child in children
    ), len(children)


def _encode_vertex_offsets(
    offsets: tuple[FixtureVertexMorphOffset, ...],
) -> tuple[bytes, int]:
    return b"".join(
        _unsigned_index(o.vertex_index, _VERTEX_INDEX_SIZE) + _v3(o.offset)
        for o in offsets
    ), len(offsets)


def _encode_bone_offsets(
    offsets: tuple[FixtureBoneMorphOffset, ...],
) -> tuple[bytes, int]:
    return b"".join(
        _signed_index(o.bone_index, _BONE_INDEX_SIZE) + _v3(o.translation) + _v4(o.rotation)
        for o in offsets
    ), len(offsets)


def _encode_uv_offsets(offsets: tuple[FixtureUvMorphOffset, ...]) -> tuple[bytes, int]:
    return b"".join(
        _unsigned_index(o.vertex_index, _VERTEX_INDEX_SIZE) + _v4(o.offset)
        for o in offsets
    ), len(offsets)


def _encode_material_targets(
    targets: tuple[FixtureMaterialMorphTarget, ...],
) -> tuple[bytes, int]:
    out: list[bytes] = []
    for t in targets:
        out.append(_signed_index(t.material_index, _MATERIAL_INDEX_SIZE))
        out.append(_u8(t.op))
        out.append(_v4(t.diffuse))
        out.append(_v3(t.specular))
        out.append(_f(t.specular_power))
        out.append(_v3(t.ambient))
        out.append(_v4(t.edge_color))
        out.append(_f(t.edge_size))
        out.append(_v4(t.texture_coef))
        out.append(_v4(t.sphere_coef))
        out.append(_v4(t.toon_coef))
    return b"".join(out), len(targets)


def _encode_rigid_bodies(spec: FixtureBuild) -> bytes:
    out = [_i32(len(spec.rigid_bodies))]
    encoding_byte = spec.encoding_byte
    for body in spec.rigid_bodies:
        out.append(_encode_text(body.name, encoding_byte))
        out.append(_encode_text(body.name, encoding_byte))
        out.append(_signed_index(body.bone_index, _BONE_INDEX_SIZE))
        out.append(_u8(body.group))
        out.append(_u16(body.non_collision_mask))
        out.append(_u8(body.shape))
        out.append(_v3(body.size))
        out.append(_v3(body.position))
        out.append(_v3(body.rotation))
        out.append(_f(body.mass))
        out.append(_f(body.linear_damping))
        out.append(_f(body.angular_damping))
        out.append(_f(body.restitution))
        out.append(_f(body.friction))
        out.append(_u8(body.physics_mode))
    return b"".join(out)


def _encode_joints(spec: FixtureBuild) -> bytes:
    out = [_i32(len(spec.joints))]
    encoding_byte = spec.encoding_byte
    for joint in spec.joints:
        out.append(_encode_text(joint.name, encoding_byte))
        out.append(_encode_text(joint.name, encoding_byte))
        out.append(_u8(0))                                              # joint type — spring 6DOF
        out.append(_signed_index(joint.rigid_a, _RIGID_INDEX_SIZE))
        out.append(_signed_index(joint.rigid_b, _RIGID_INDEX_SIZE))
        out.append(_v3(joint.position))
        out.append(_v3(joint.rotation))
        out.append(_v3(joint.linear_lower))
        out.append(_v3(joint.linear_upper))
        out.append(_v3(joint.angular_lower))
        out.append(_v3(joint.angular_upper))
        out.append(_v3(joint.spring_linear))
        out.append(_v3(joint.spring_angular))
    return b"".join(out)


def _encode_display_frames(spec: FixtureBuild) -> bytes:
    out = [_i32(len(spec.display_frames))]
    encoding_byte = spec.encoding_byte
    for frame in spec.display_frames:
        out.append(_encode_text(frame.name, encoding_byte))
        out.append(_encode_text(frame.name, encoding_byte))   # name_en — reuse JP
        out.append(_u8(1 if frame.is_special else 0))
        out.append(_i32(len(frame.elements)))
        for element in frame.elements:
            out.append(_u8(element.kind))
            if element.kind == 0:
                out.append(_signed_index(element.index, _BONE_INDEX_SIZE))
            else:
                out.append(_signed_index(element.index, _MORPH_INDEX_SIZE))
    return b"".join(out)


def build_pmx(spec: FixtureBuild) -> bytes:
    """Serialise ``spec`` into a complete PMX byte buffer."""
    return (
        _encode_header(spec)
        + _encode_names(spec)
        + _encode_vertices(spec)
        + _encode_indices(spec)
        + _encode_textures(spec)
        + _encode_materials(spec)
        + _encode_bones(spec)
        + _encode_morphs(spec)
        + _encode_display_frames(spec)
        + _encode_rigid_bodies(spec)
        + _encode_joints(spec)
    )


# ----- canonical "tiny" cube fixture --------------------------------------
def tiny_cube_spec() -> FixtureBuild:
    """Eight-vertex unit cube exercising every PMX skinning weight type.

    Two vertices use BDEF1, two use BDEF2, two use BDEF4, two use SDEF —
    so the parser has to dispatch through every branch of the deform reader
    on a single fixture. One internal-toon material covers all 36 indices.
    """
    vertices = (
        FixtureVertex(position=(-1.0, -1.0, -1.0), deform=_Bdef1(bone=0)),
        FixtureVertex(position=(+1.0, -1.0, -1.0), deform=_Bdef1(bone=0)),
        FixtureVertex(position=(+1.0, +1.0, -1.0), deform=_Bdef2(0, 1, 0.5)),
        FixtureVertex(position=(-1.0, +1.0, -1.0), deform=_Bdef2(0, 1, 0.5)),
        FixtureVertex(
            position=(-1.0, -1.0, +1.0),
            deform=_Bdef4(bones=(0, 1, -1, -1), weights=(0.7, 0.3, 0.0, 0.0)),
        ),
        FixtureVertex(
            position=(+1.0, -1.0, +1.0),
            deform=_Bdef4(bones=(1, 0, -1, -1), weights=(0.6, 0.4, 0.0, 0.0)),
        ),
        FixtureVertex(
            position=(+1.0, +1.0, +1.0),
            deform=_Sdef(
                bone1=0, bone2=1, weight1=0.5,
                c=(0.0, 0.5, 0.0), r0=(0.0, 0.0, 0.0), r1=(0.0, 0.0, 0.0),
            ),
        ),
        FixtureVertex(
            position=(-1.0, +1.0, +1.0),
            deform=_Sdef(
                bone1=0, bone2=1, weight1=0.5,
                c=(0.0, 0.5, 0.0), r0=(0.0, 0.0, 0.0), r1=(0.0, 0.0, 0.0),
            ),
        ),
    )
    # Indices are wound CCW from outside the cube so back-face culling keeps
    # the camera-facing faces visible (the alternative inward winding makes
    # the cube vanish under any pipeline that culls back faces — e.g. the
    # toon pass).
    indices = (
        0, 2, 1, 0, 3, 2,        # -Z
        4, 5, 6, 4, 6, 7,        # +Z
        0, 5, 4, 0, 1, 5,        # -Y
        2, 7, 6, 2, 3, 7,        # +Y
        1, 6, 5, 1, 2, 6,        # +X
        0, 7, 3, 0, 4, 7,        # -X
    )
    materials = (
        FixtureMaterial(
            name_jp="mat0",
            diffuse=(1.0, 0.85, 0.85, 1.0),
            texture_index=0,
            face_index_count=len(indices),
        ),
    )
    bones = (
        FixtureBone(name_jp="root", position=(0.0, 0.0, 0.0), parent_index=-1),
        FixtureBone(name_jp="child", position=(0.0, 1.0, 0.0), parent_index=0),
    )
    return FixtureBuild(
        encoding_byte=1,
        name_jp="tiny",
        name_en="tiny",
        vertices=vertices,
        indices=indices,
        textures=("tex/diffuse.png",),
        materials=materials,
        bones=bones,
    )


def tiny_leg_with_ik_spec() -> FixtureBuild:
    """Minimal 4-bone leg fixture for the IK CCD solver.

    Layout (rest pose, all in the model's Y-up frame):

    - bone 0 ``hip`` at ``(0, 2, 0)`` — root of the chain
    - bone 1 ``knee`` at ``(0, 1, 0)`` — child of hip
    - bone 2 ``ankle`` at ``(0, 0, 0)`` — child of knee, the *effector*
    - bone 3 ``ik_driver`` at ``(0.5, 0.5, 0.0)`` — the IK *driver* bone

    The driver carries an IK chain pointing at the ankle with two links
    (knee, hip) where the knee is constrained to bend only around the X
    axis. A CCD solve must pull the ankle's world position toward the
    driver's ``(0.5, 0.5, 0.0)`` while keeping the knee's rotation
    inside its limit box.
    """
    # Reuse the cube vertices / material / index buffer.
    base = tiny_cube_spec()
    knee_link = FixtureIkLink(
        bone_index=1,
        has_limit=True,
        # MMD knee convention: bend only forward on X (negative range).
        limit_min=(-3.1415, 0.0, 0.0),
        limit_max=(0.0, 0.0, 0.0),
    )
    hip_link = FixtureIkLink(bone_index=0)
    bones = (
        FixtureBone(name_jp="hip", position=(0.0, 2.0, 0.0), parent_index=-1),
        FixtureBone(name_jp="knee", position=(0.0, 1.0, 0.0), parent_index=0),
        FixtureBone(name_jp="ankle", position=(0.0, 0.0, 0.0), parent_index=1),
        FixtureBone(
            name_jp="ik_driver", position=(0.5, 0.5, 0.0), parent_index=-1,
            ik=FixtureIk(
                target_bone_index=2,
                iterations=20,
                limit_radian=1.0,
                links=(knee_link, hip_link),
            ),
        ),
    )
    return replace(base, bones=bones)


def tiny_cube_with_morphs_spec() -> FixtureBuild:
    """Tiny cube + four morphs covering every Phase-4-driven type.

    Indices in :class:`FixtureGroupChild` reference morphs in the same
    tuple, so the last entry (the group morph) recursively pulls the
    earlier vertex + bone morphs.
    """
    base = tiny_cube_spec()
    morphs = (
        FixtureMorph(
            name="vert_pull",
            morph_type=1,           # VERTEX
            vertex_offsets=(
                FixtureVertexMorphOffset(vertex_index=0, offset=(0.5, 0.0, 0.0)),
                FixtureVertexMorphOffset(vertex_index=1, offset=(0.0, 0.5, 0.0)),
            ),
        ),
        FixtureMorph(
            name="bone_tilt",
            morph_type=2,           # BONE
            bone_offsets=(
                FixtureBoneMorphOffset(
                    bone_index=1,
                    translation=(0.1, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.2588190, 0.9659258),  # 30° around Z
                ),
            ),
        ),
        FixtureMorph(
            name="redder",
            morph_type=8,           # MATERIAL
            material_targets=(
                FixtureMaterialMorphTarget(
                    material_index=0, op=0,                # multiply
                    diffuse=(1.5, 0.5, 0.5, 1.0),
                ),
            ),
        ),
        FixtureMorph(
            name="all",
            morph_type=0,           # GROUP
            group_children=(
                FixtureGroupChild(morph_index=0, weight=1.0),
                FixtureGroupChild(morph_index=1, weight=1.0),
            ),
        ),
    )
    return replace(base, morphs=morphs)


# ----- PMD bytes ---------------------------------------------------------
def _pmd_text(text: str, byte_count: int) -> bytes:
    raw = text.encode("cp932", errors="replace")
    if len(raw) >= byte_count:
        return raw[:byte_count]
    return raw + b"\x00" * (byte_count - len(raw))


def _pmd_vertex(
    position: tuple[float, float, float],
    bone1: int,
    bone2: int,
    weight_percent: int,
) -> bytes:
    """Encode one PMD vertex (38 bytes): pos, normal, uv, b1, b2, weight%, edge."""
    return (
        _v3(position)
        + _v3((0.0, 1.0, 0.0))            # normal
        + struct.pack("<ff", 0.0, 0.0)    # uv
        + _u16(bone1) + _u16(bone2)
        + _u8(weight_percent) + _u8(0)
    )


def _pmd_material(diffuse_texture: str, face_index_count: int) -> bytes:
    return (
        _v4((1.0, 0.85, 0.85, 1.0))       # diffuse
        + _f(0.0)                          # specular factor
        + _v3((0.0, 0.0, 0.0))             # specular
        + _v3((0.5, 0.5, 0.5))             # ambient
        + _u8(0)                           # toon index
        + _u8(0)                           # edge flag
        + struct.pack("<I", face_index_count)
        + _pmd_text(diffuse_texture, 20)
    )


def _pmd_bone(name: str, parent: int, position: tuple[float, float, float]) -> bytes:
    parent_word = parent & 0xFFFF if parent >= 0 else 0xFFFF
    return (
        _pmd_text(name, 20)
        + _u16(parent_word)
        + _u16(0xFFFF)            # tail bone — none
        + _u8(0)                  # type — generic
        + _u16(0xFFFF)            # IK index — none
        + _v3(position)
    )


def _vmd_text(text: str, byte_count: int) -> bytes:
    raw = text.encode("cp932", errors="replace")
    if len(raw) >= byte_count:
        return raw[:byte_count]
    return raw + b"\x00" * (byte_count - len(raw))


def _vmd_bone_record(
    bone_name: str,
    frame: int,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
    handles: tuple[tuple[int, int, int, int], ...] = (
        (20, 20, 107, 107),     # X
        (20, 20, 107, 107),     # Y
        (20, 20, 107, 107),     # Z
        (20, 20, 107, 107),     # rotation
    ),
) -> bytes:
    """Encode one 111-byte VMD bone keyframe.

    Default handles are the standard "linear" bezier (control points on the
    diagonal) — same setting MMD writes when the user picks "linear" in the
    interpolation panel. Tests that need a specific easing override this.
    """
    bezier = bytearray(64)
    for channel in range(4):
        x1, y1, x2, y2 = handles[channel]
        bezier[channel + 0] = x1
        bezier[channel + 4] = y1
        bezier[channel + 8] = x2
        bezier[channel + 12] = y2
    return (
        _vmd_text(bone_name, 15)
        + struct.pack("<I", frame)
        + struct.pack("<fff", *position)
        + struct.pack("<ffff", *rotation)
        + bytes(bezier)
    )


def _vmd_morph_record(morph_name: str, frame: int, weight: float) -> bytes:
    """Encode one 23-byte VMD morph keyframe."""
    return _vmd_text(morph_name, 15) + struct.pack("<I", frame) + struct.pack("<f", weight)


def build_vmd_morphs(
    target_morph: str = "all",
    target_model: str = "tiny",
    *,
    weights_at_frames: tuple[tuple[int, float], ...] = ((0, 0.0), (10, 1.0), (20, 0.0)),
) -> bytes:
    """VMD: one morph track ramping ``target_morph`` from 0 → 1 → 0 over 20 frames."""
    signature = _vmd_text("Vocaloid Motion Data 0002", 30)
    model_name = _vmd_text(target_model, 20)
    bone_section = struct.pack("<I", 0)
    morph_records = b"".join(
        _vmd_morph_record(target_morph, frame, weight)
        for frame, weight in weights_at_frames
    )
    morph_section = struct.pack("<I", len(weights_at_frames)) + morph_records
    camera_section = struct.pack("<I", 0)
    light_section = struct.pack("<I", 0)
    self_shadow_section = struct.pack("<I", 0)
    ik_section = struct.pack("<I", 0)
    return (
        signature + model_name
        + bone_section + morph_section + camera_section
        + light_section + self_shadow_section + ik_section
    )


def _vmd_camera_record(
    frame: int,
    distance: float,
    target: tuple[float, float, float],
    rotation: tuple[float, float, float],
    fov_degrees: int,
    perspective_off: bool,
    *,
    handles_per_channel: tuple[int, int, int, int] = (20, 20, 107, 107),
) -> bytes:
    """Encode one 61-byte VMD camera keyframe.

    The 6 bezier channels each carry the same default control points
    (``20, 20, 107, 107`` ≈ near-linear); tests that need explicit
    eases override the bytes by hand.
    """
    bezier = bytearray(24)
    for channel in range(6):
        bezier[channel * 4 + 0] = handles_per_channel[0]
        bezier[channel * 4 + 1] = handles_per_channel[1]
        bezier[channel * 4 + 2] = handles_per_channel[2]
        bezier[channel * 4 + 3] = handles_per_channel[3]
    return (
        struct.pack("<I", frame)
        + struct.pack("<f", distance)
        + struct.pack("<fff", *target)
        + struct.pack("<fff", *rotation)
        + bytes(bezier)
        + struct.pack("<I", fov_degrees)
        + bytes([1 if perspective_off else 0])
    )


def _vmd_light_record(
    frame: int,
    color: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> bytes:
    """Encode one 28-byte VMD light keyframe."""
    return (
        struct.pack("<I", frame)
        + struct.pack("<fff", *color)
        + struct.pack("<fff", *direction)
    )


def _vmd_self_shadow_record(frame: int, mode: int, distance: float) -> bytes:
    return struct.pack("<I", frame) + bytes([mode]) + struct.pack("<f", distance)


def build_vmd_camera_motion(
    target_model: str = "",
    *,
    camera_keyframes: tuple[
        tuple[int, float, tuple[float, float, float], tuple[float, float, float], int, bool],
        ...,
    ] = (),
    light_keyframes: tuple[
        tuple[int, tuple[float, float, float], tuple[float, float, float]], ...,
    ] = (),
    self_shadow_keyframes: tuple[tuple[int, int, float], ...] = (),
) -> bytes:
    """VMD with no bone / morph data — just camera / light / self-shadow.

    Used to exercise the camera + light + self-shadow tracks without
    spinning up a full bone player. Each keyframe tuple lists positional
    fields in the order expected by their corresponding ``_vmd_*_record``
    encoder.
    """
    signature = _vmd_text("Vocaloid Motion Data 0002", 30)
    model_name = _vmd_text(target_model, 20)
    bone_section = struct.pack("<I", 0)
    morph_section = struct.pack("<I", 0)
    camera_records = b"".join(_vmd_camera_record(*kf) for kf in camera_keyframes)
    camera_section = struct.pack("<I", len(camera_keyframes)) + camera_records
    light_records = b"".join(_vmd_light_record(*kf) for kf in light_keyframes)
    light_section = struct.pack("<I", len(light_keyframes)) + light_records
    self_shadow_records = b"".join(
        _vmd_self_shadow_record(*kf) for kf in self_shadow_keyframes
    )
    self_shadow_section = (
        struct.pack("<I", len(self_shadow_keyframes)) + self_shadow_records
    )
    ik_section = struct.pack("<I", 0)
    return (
        signature + model_name
        + bone_section + morph_section + camera_section
        + light_section + self_shadow_section + ik_section
    )


def build_vpd_text(
    *,
    model_name: str = "tiny.osm",
    bones: tuple[
        tuple[
            str,
            tuple[float, float, float],
            tuple[float, float, float, float],
        ],
        ...,
    ] = (),
    morphs: tuple[tuple[str, float], ...] = (),
) -> str:
    """Build a VPD source string in MMD's canonical layout.

    Each bone tuple is ``(name, translation, rotation_quat)``; each morph
    tuple is ``(name, weight)``. The output uses LF newlines and
    semicolon-terminated value lines so it round-trips cleanly through
    :func:`vpd.reader.parse_vpd`.
    """
    lines: list[str] = ["Vocaloid Pose Data file", "", f"{model_name};", f"{len(bones)};", ""]
    for index, (name, translation, rotation) in enumerate(bones):
        lines.extend(
            [
                f"Bone{index}{{{name}",
                f"  {translation[0]:.6f},{translation[1]:.6f},{translation[2]:.6f};",
                f"  {rotation[0]:.6f},{rotation[1]:.6f},{rotation[2]:.6f},{rotation[3]:.6f};",
                "}",
                "",
            ]
        )
    for index, (name, weight) in enumerate(morphs):
        lines.extend(
            [f"Morph{index}{{{name}", f"  {weight:.6f};", "}", ""],
        )
    return "\n".join(lines).rstrip("\n") + "\n"


def build_vpd_bytes(**kwargs) -> bytes:                # noqa: ANN003 — passthrough
    """SJIS-encoded VPD bytes for tests that need an on-disk fixture."""
    return build_vpd_text(**kwargs).encode("cp932", errors="replace")


def build_vmd_wave(target_bone: str = "child", target_model: str = "tiny") -> bytes:
    """Tiny VMD: one bone (``target_bone``) waving on the X axis over 5 keys.

    Frames 0, 5, 10, 15, 20 alternate the rotation between identity and a
    quarter turn around X — enough motion that a render smoke can spot the
    movement without slipping below the noise floor.
    """
    signature = _vmd_text("Vocaloid Motion Data 0002", 30)
    model_name = _vmd_text(target_model, 20)
    quarter_turn = (0.3826834323650898, 0.0, 0.0, 0.9238795325112867)   # 45° around X
    identity = (0.0, 0.0, 0.0, 1.0)
    keyframes = [
        _vmd_bone_record(target_bone, 0,  (0.0, 0.0, 0.0), identity),
        _vmd_bone_record(target_bone, 5,  (0.0, 0.0, 0.0), quarter_turn),
        _vmd_bone_record(target_bone, 10, (0.0, 0.0, 0.0), identity),
        _vmd_bone_record(target_bone, 15, (0.0, 0.0, 0.0), quarter_turn),
        _vmd_bone_record(target_bone, 20, (0.0, 0.0, 0.0), identity),
    ]
    bone_section = struct.pack("<I", len(keyframes)) + b"".join(keyframes)
    morph_section = struct.pack("<I", 0)
    camera_section = struct.pack("<I", 0)
    light_section = struct.pack("<I", 0)
    self_shadow_section = struct.pack("<I", 0)
    ik_section = struct.pack("<I", 0)
    return (
        signature + model_name
        + bone_section + morph_section + camera_section
        + light_section + self_shadow_section + ik_section
    )


def build_pmd_tiny() -> bytes:
    """Tiny 8-vertex 12-tri 2-bone PMD — analogue of :func:`tiny_cube_spec`."""
    header = b"Pmd" + _f(1.0) + _pmd_text("tiny", 20) + _pmd_text("comment", 256)
    vertex_records = [
        _pmd_vertex((-1.0, -1.0, -1.0), 0, 0, 100),
        _pmd_vertex((+1.0, -1.0, -1.0), 0, 0, 100),
        _pmd_vertex((+1.0, +1.0, -1.0), 0, 1, 50),
        _pmd_vertex((-1.0, +1.0, -1.0), 0, 1, 50),
        _pmd_vertex((-1.0, -1.0, +1.0), 0, 1, 70),
        _pmd_vertex((+1.0, -1.0, +1.0), 1, 0, 60),
        _pmd_vertex((+1.0, +1.0, +1.0), 0, 1, 50),
        _pmd_vertex((-1.0, +1.0, +1.0), 0, 1, 50),
    ]
    vertex_section = struct.pack("<I", len(vertex_records)) + b"".join(vertex_records)
    indices = (
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        2, 6, 7, 2, 7, 3,
        1, 5, 6, 1, 6, 2,
        0, 3, 7, 0, 7, 4,
    )
    index_section = struct.pack("<I", len(indices)) + b"".join(_u16(i) for i in indices)
    material_section = (
        struct.pack("<I", 1)
        + _pmd_material("diffuse.png", len(indices))
    )
    bone_records = [
        _pmd_bone("root", -1, (0.0, 0.0, 0.0)),
        _pmd_bone("child", 0, (0.0, 1.0, 0.0)),
    ]
    bone_section = _u16(len(bone_records)) + b"".join(bone_records)
    return header + vertex_section + index_section + material_section + bone_section
