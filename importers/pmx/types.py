"""Format-internal intermediate types for PMX / PMD documents.

These dataclasses are the parser output. The PMX/PMD importer adapter
(``importer.py``) consumes them and produces the engine-facing
:class:`~posecascade.assets.types.ImportedScene`. Sections that Phase 1 does
not yet render (morphs, rigid bodies, joints, soft bodies) are still parsed
and held here so later phases can pick them up without re-reading the file.

Field naming follows the PMX 2.0/2.1 spec terminology (``deform_type``,
``edge_color`` …) so anyone cross-referencing the spec can navigate quickly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class PmxTextEncoding(IntEnum):
    """PMX header text-encoding flag."""

    UTF16_LE = 0
    UTF8 = 1


class PmxDeformType(IntEnum):
    """Vertex skinning weight type recorded per vertex."""

    BDEF1 = 0
    BDEF2 = 1
    BDEF4 = 2
    SDEF = 3
    QDEF = 4   # PMX 2.1 only


class PmxSphereMode(IntEnum):
    """Sphere-texture blend mode."""

    DISABLED = 0
    MULTIPLY = 1
    ADD = 2
    SUB_TEXTURE = 3


class PmxToonMode(IntEnum):
    """Whether the material's toon ramp is an external texture or one of the
    built-in ``toon01.bmp`` … ``toon10.bmp`` ramps."""

    EXTERNAL = 0
    INTERNAL = 1


class PmxMorphPanel(IntEnum):
    """MMD UI panel a morph appears in."""

    HIDDEN = 0
    EYE = 1
    LIP = 2
    BROW = 3
    OTHER = 4


class PmxMorphType(IntEnum):
    """Morph offset payload shape."""

    GROUP = 0
    VERTEX = 1
    BONE = 2
    UV = 3
    UV1 = 4
    UV2 = 5
    UV3 = 6
    UV4 = 7
    MATERIAL = 8
    FLIP = 9        # PMX 2.1 only
    IMPULSE = 10    # PMX 2.1 only


class PmxRigidShape(IntEnum):
    SPHERE = 0
    BOX = 1
    CAPSULE = 2


class PmxPhysicsMode(IntEnum):
    KINEMATIC = 0
    DYNAMIC = 1
    DYNAMIC_BONE = 2


class PmxJointType(IntEnum):
    """Joint type byte. PMX 2.1 defines 0..5; we support 0 (spring 6DOF) on
    the rendering / physics side, and parse the rest as raw values."""

    SPRING_6DOF = 0
    SIX_DOF = 1
    P2P = 2
    CONE_TWIST = 3
    SLIDER = 4
    HINGE = 5


@dataclass(frozen=True)
class PmxHeader:
    """PMX header — version + encoding flags + per-index byte sizes.

    ``vertex_index_size`` is unsigned (faces use unsigned indices); all other
    index-size fields refer to *signed* indices where ``-1`` is the spec's
    "no reference" sentinel.
    """

    version: float
    text_encoding: PmxTextEncoding
    additional_uv_count: int
    vertex_index_size: int
    texture_index_size: int
    material_index_size: int
    bone_index_size: int
    morph_index_size: int
    rigid_index_size: int


@dataclass(frozen=True)
class PmxNames:
    """Model name + comment, both Japanese + English."""

    name_jp: str = ""
    name_en: str = ""
    comment_jp: str = ""
    comment_en: str = ""


@dataclass(frozen=True)
class PmxBdef1:
    bone: int


@dataclass(frozen=True)
class PmxBdef2:
    bone1: int
    bone2: int
    weight1: float


@dataclass(frozen=True)
class PmxBdef4:
    bones: tuple[int, int, int, int]
    weights: tuple[float, float, float, float]


@dataclass(frozen=True)
class PmxSdef:
    """Spherical-defined skinning. ``c`` / ``r0`` / ``r1`` are 3-vectors."""

    bone1: int
    bone2: int
    weight1: float
    c: tuple[float, float, float]
    r0: tuple[float, float, float]
    r1: tuple[float, float, float]


@dataclass(frozen=True)
class PmxQdef:
    bones: tuple[int, int, int, int]
    weights: tuple[float, float, float, float]


PmxDeform = PmxBdef1 | PmxBdef2 | PmxBdef4 | PmxSdef | PmxQdef


@dataclass(frozen=True)
class PmxVertex:
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    uv: tuple[float, float]
    additional_uvs: tuple[tuple[float, float, float, float], ...]
    deform_type: PmxDeformType
    deform: PmxDeform
    edge_ratio: float


@dataclass(frozen=True)
class PmxMaterial:
    """One material section. ``face_index_count`` is the number of *indices*
    (not triangles) this material consumes from the flat face buffer."""

    name_jp: str
    name_en: str
    diffuse: tuple[float, float, float, float]
    specular: tuple[float, float, float]
    specular_factor: float
    ambient: tuple[float, float, float]
    flags: int
    edge_color: tuple[float, float, float, float]
    edge_size: float
    texture_index: int                # -1 = none
    sphere_texture_index: int         # -1 = none
    sphere_mode: PmxSphereMode
    toon_mode: PmxToonMode
    toon_reference: int               # texture index OR internal toon idx (0..9)
    memo: str
    face_index_count: int


@dataclass(frozen=True)
class PmxIkLink:
    bone_index: int
    has_limit: bool
    limit_min: tuple[float, float, float]   # radians
    limit_max: tuple[float, float, float]   # radians


@dataclass(frozen=True)
class PmxIk:
    """IK chain hung off a bone (the IK *driver* bone holds this struct)."""

    target_bone_index: int
    iterations: int
    limit_radian: float
    links: tuple[PmxIkLink, ...]


@dataclass(frozen=True)
class PmxBone:
    """One bone. Tail is either an offset vec3 OR a bone index — exactly one
    of ``tail_offset`` / ``tail_bone_index`` is populated (the other is
    ``None``), keyed off ``flag_indexed_tail``.
    """

    name_jp: str
    name_en: str
    position: tuple[float, float, float]
    parent_index: int                   # -1 = root
    deformation_depth: int
    flags: int
    tail_offset: tuple[float, float, float] | None
    tail_bone_index: int | None
    inherit_parent_index: int | None    # bit 8 / bit 9
    inherit_weight: float | None
    fixed_axis: tuple[float, float, float] | None
    local_x_axis: tuple[float, float, float] | None
    local_z_axis: tuple[float, float, float] | None
    external_parent_key: int | None
    ik: PmxIk | None


# ----- bone flag bits (PMX 2.0 spec) -----
PMX_BONE_FLAG_INDEXED_TAIL = 1 << 0
PMX_BONE_FLAG_ROTATABLE = 1 << 1
PMX_BONE_FLAG_TRANSLATABLE = 1 << 2
PMX_BONE_FLAG_VISIBLE = 1 << 3
PMX_BONE_FLAG_ENABLED = 1 << 4
PMX_BONE_FLAG_IK = 1 << 5
PMX_BONE_FLAG_INHERIT_ROTATION = 1 << 8
PMX_BONE_FLAG_INHERIT_TRANSLATION = 1 << 9
PMX_BONE_FLAG_FIXED_AXIS = 1 << 10
PMX_BONE_FLAG_LOCAL_AXIS = 1 << 11
PMX_BONE_FLAG_PHYSICS_AFTER = 1 << 12
PMX_BONE_FLAG_EXTERNAL_PARENT = 1 << 13


# ----- material flag bits (PMX 2.0 spec) -----
PMX_MAT_FLAG_DOUBLE_SIDED = 1 << 0
PMX_MAT_FLAG_GROUND_SHADOW = 1 << 1
PMX_MAT_FLAG_CAST_SHADOW = 1 << 2
PMX_MAT_FLAG_RECEIVE_SHADOW = 1 << 3
PMX_MAT_FLAG_HAS_EDGE = 1 << 4
PMX_MAT_FLAG_VERTEX_COLOR = 1 << 5    # PMX 2.1
PMX_MAT_FLAG_POINT_DRAW = 1 << 6      # PMX 2.1
PMX_MAT_FLAG_LINE_DRAW = 1 << 7       # PMX 2.1


# ----- morph offset payloads -----
@dataclass(frozen=True)
class PmxGroupMorphOffset:
    morph_index: int
    weight: float


@dataclass(frozen=True)
class PmxVertexMorphOffset:
    vertex_index: int
    offset: tuple[float, float, float]


@dataclass(frozen=True)
class PmxBoneMorphOffset:
    bone_index: int
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]   # quat (x, y, z, w)


@dataclass(frozen=True)
class PmxUvMorphOffset:
    vertex_index: int
    offset: tuple[float, float, float, float]


@dataclass(frozen=True)
class PmxMaterialMorphOffset:
    material_index: int                # -1 = applies to all materials
    op: int                            # 0=multiply, 1=add
    diffuse: tuple[float, float, float, float]
    specular: tuple[float, float, float]
    specular_factor: float
    ambient: tuple[float, float, float]
    edge_color: tuple[float, float, float, float]
    edge_size: float
    texture_coef: tuple[float, float, float, float]
    sphere_coef: tuple[float, float, float, float]
    toon_coef: tuple[float, float, float, float]


@dataclass(frozen=True)
class PmxFlipMorphOffset:
    morph_index: int
    weight: float


@dataclass(frozen=True)
class PmxImpulseMorphOffset:
    rigid_body_index: int
    is_local: bool
    velocity: tuple[float, float, float]
    torque: tuple[float, float, float]


PmxMorphOffset = (
    PmxGroupMorphOffset
    | PmxVertexMorphOffset
    | PmxBoneMorphOffset
    | PmxUvMorphOffset
    | PmxMaterialMorphOffset
    | PmxFlipMorphOffset
    | PmxImpulseMorphOffset
)


@dataclass(frozen=True)
class PmxMorph:
    name_jp: str
    name_en: str
    panel: PmxMorphPanel
    morph_type: PmxMorphType
    offsets: tuple[PmxMorphOffset, ...]


@dataclass(frozen=True)
class PmxDisplayElement:
    """Display-frame element: ``kind`` is 0 for bone, 1 for morph; ``index``
    points into the bones / morphs list accordingly."""

    kind: int
    index: int


@dataclass(frozen=True)
class PmxDisplayFrame:
    name_jp: str
    name_en: str
    is_special: bool
    elements: tuple[PmxDisplayElement, ...]


@dataclass(frozen=True)
class PmxRigidBody:
    name_jp: str
    name_en: str
    related_bone_index: int            # -1 = none
    group: int                         # 0..15
    non_collision_mask: int            # 16-bit mask
    shape: PmxRigidShape
    size: tuple[float, float, float]
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]   # euler radians
    mass: float
    linear_damping: float
    angular_damping: float
    restitution: float
    friction: float
    physics_mode: PmxPhysicsMode


@dataclass(frozen=True)
class PmxJoint:
    name_jp: str
    name_en: str
    joint_type: int                                # raw byte (see PmxJointType)
    rigid_a_index: int
    rigid_b_index: int
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    linear_lower: tuple[float, float, float]
    linear_upper: tuple[float, float, float]
    angular_lower: tuple[float, float, float]
    angular_upper: tuple[float, float, float]
    spring_linear: tuple[float, float, float]
    spring_angular: tuple[float, float, float]


@dataclass(frozen=True)
class PmxDocument:
    """Root container — the full parsed PMX/PMD document."""

    header: PmxHeader
    names: PmxNames
    vertices: tuple[PmxVertex, ...] = field(default_factory=tuple)
    indices: tuple[int, ...] = field(default_factory=tuple)
    textures: tuple[str, ...] = field(default_factory=tuple)
    materials: tuple[PmxMaterial, ...] = field(default_factory=tuple)
    bones: tuple[PmxBone, ...] = field(default_factory=tuple)
    morphs: tuple[PmxMorph, ...] = field(default_factory=tuple)
    display_frames: tuple[PmxDisplayFrame, ...] = field(default_factory=tuple)
    rigid_bodies: tuple[PmxRigidBody, ...] = field(default_factory=tuple)
    joints: tuple[PmxJoint, ...] = field(default_factory=tuple)
