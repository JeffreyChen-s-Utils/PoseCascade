"""PMX 2.0 / 2.1 importer adapter.

Walks a parsed :class:`~pmx.types.PmxDocument` into the engine-facing
:class:`~posecascade.assets.types.ImportedScene`. Phase 1 emits:

- one :class:`Mesh` per PMX material (sharing the full vertex buffer; each
  Mesh holds only its slice of the flat index array)
- a :class:`Skin` whose joints are the bone scene-graph nodes and whose
  inverse-bind matrices reduce to a translation by ``-bone.position``
  because PMX bones rest at identity rotation/scale
- a :class:`Scene` with the bone hierarchy + a "model" node carrying a
  :class:`MeshRefComponent` (every material) and a :class:`SkinRefComponent`

Sphere textures, toon ramps, morphs, rigid bodies, and joints are still
parsed by the reader but not exposed on :class:`ImportedScene` yet — those
move into the engine in later phases (Phase 2 toon, Phase 4 morph,
Phase 7 physics).
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pmx.reader import parse_pmx
from pmx.types import (
    PMX_BONE_FLAG_FIXED_AXIS,
    PMX_BONE_FLAG_INHERIT_ROTATION,
    PMX_BONE_FLAG_INHERIT_TRANSLATION,
    PmxBdef1,
    PmxBdef2,
    PmxBdef4,
    PmxBone,
    PmxBoneMorphOffset,
    PmxDisplayFrame,
    PmxDocument,
    PmxFlipMorphOffset,
    PmxGroupMorphOffset,
    PmxImpulseMorphOffset,
    PmxJoint,
    PmxMaterial,
    PmxMaterialMorphOffset,
    PmxMorph,
    PmxMorphType,
    PmxQdef,
    PmxRigidBody,
    PmxSdef,
    PmxToonMode,
    PmxUvMorphOffset,
    PmxVertex,
    PmxVertexMorphOffset,
)
from posecascade.animation.bone_resolver import (
    BoneAppendRule,
    BoneResolverRules,
    FixedAxisRule,
)
from posecascade.animation.display_frames import (
    DisplayFrameElement,
    DisplayFrameElementKind,
    DisplayFrameGroup,
)
from posecascade.animation.ik import IkChain, IkLink
from posecascade.animation.morph import (
    BoneMorph,
    BoneMorphOffset,
    FlipMorph,
    GroupMorph,
    GroupMorphChild,
    ImpulseMorph,
    ImpulseMorphOffset,
    MaterialMorph,
    MaterialMorphOp,
    MaterialMorphTarget,
    Morph,
    MorphAsset,
    MorphPanel,
    UvMorph,
    UvMorphOffset,
    VertexMorph,
    VertexMorphOffset,
    build_morph_asset,
)
from posecascade.assets.path_safety import resolve_safe
from posecascade.assets.types import ImportedScene, Mesh, Skin, Texture
from posecascade.errors import MalformedAssetError, UnsafePathError
from posecascade.physics.types import (
    Joint6DofSpring,
    PhysicsMode,
    PhysicsScene,
    RigidBody,
    RigidShape,
)
from posecascade.render.constants import MAX_TEXTURE_DIMENSION
from posecascade.render.internal_toon import INTERNAL_TOON_COUNT, load_internal_toon
from posecascade.render.material import MMDMaterial, SphereMode
from posecascade.scene.component import MeshRefComponent, SkinRefComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.scene.transform import Transform
from posecascade.utils.math3d import quat_identity, vec3

# White 1×1 placeholder so the renderer always has a valid base-colour bind.
_PLACEHOLDER_PIXELS = np.full((1, 1, 4), 255, dtype=np.uint8)


@dataclass(frozen=True)
class _VertexBuffers:
    """All per-vertex arrays the engine needs, packed once for the whole model."""

    positions: NDArray[np.float32]
    normals: NDArray[np.float32]
    texcoords: NDArray[np.float32]
    joints: NDArray[np.uint16]
    weights: NDArray[np.float32]


class PmxImporter:
    """Loads ``.pmx`` files into :class:`ImportedScene`."""

    supported_extensions: tuple[str, ...] = (".pmx",)

    def load(self, path: Path) -> ImportedScene:
        path = path.resolve()
        if not path.is_file():
            raise MalformedAssetError(f"PMX file not found: {path}")
        return _build_imported_scene(parse_pmx(path.read_bytes()), path.parent)


def _build_imported_scene(doc: PmxDocument, asset_root: Path) -> ImportedScene:
    pmx_textures = _load_textures(doc, asset_root)
    internal_toons = _load_internal_toons()
    textures = pmx_textures + internal_toons
    toon_offset = len(pmx_textures)
    buffers = _pack_vertex_buffers(doc)
    meshes = _build_meshes(doc, buffers, toon_offset)
    bone_nodes = _build_bone_nodes(doc.bones)
    skin = _build_skin(doc.bones, bone_nodes)
    scene = _build_scene(doc, meshes, skin, bone_nodes)
    morphs = _build_morph_asset(doc.morphs)
    ik_chains = _build_ik_chains(doc.bones)
    bone_resolver_rules = _build_bone_resolver_rules(doc.bones)
    physics_scene = _build_physics_scene(doc.rigid_bodies, doc.joints)
    display_frame_groups = _build_display_frame_groups(doc.display_frames)
    return ImportedScene(
        meshes=tuple(meshes),
        textures=textures,
        skins=(skin,) if skin is not None else (),
        scene=scene,
        morphs=morphs,
        ik_chains=ik_chains,
        bone_resolver_rules=bone_resolver_rules,
        physics_scene=physics_scene,
        display_frame_groups=display_frame_groups,
    )


def _load_internal_toons() -> tuple[Texture, ...]:
    """Load the ten built-in toon ramps so PMX materials with
    ``toon_mode == INTERNAL`` can address them via a normal texture index."""
    return tuple(load_internal_toon(i) for i in range(INTERNAL_TOON_COUNT))


# ----- textures -----------------------------------------------------------
def _load_textures(doc: PmxDocument, asset_root: Path) -> tuple[Texture, ...]:
    """Decode every PMX-listed texture from disk into an RGBA :class:`Texture`.

    Failures (missing file, unsupported format, escape attempt) collapse to
    a 1×1 white placeholder so the renderer still has a valid bind for that
    slot. The slot's name keeps the PMX-claimed path so debug output is
    legible.
    """
    if not doc.textures:
        return ()
    from PIL import Image, UnidentifiedImageError  # noqa: PLC0415 — lazy heavy import

    textures: list[Texture] = []
    for index, ref in enumerate(doc.textures):
        name = ref or f"texture_{index}"
        textures.append(_load_one_texture(ref, asset_root, name, Image, UnidentifiedImageError))
    return tuple(textures)


def _load_one_texture(
    reference: str,
    asset_root: Path,
    name: str,
    image_module,                        # noqa: ANN001 — late-bound PIL.Image
    unidentified_error: type[Exception],
) -> Texture:
    try:
        resolved = resolve_safe(asset_root, reference)
    except UnsafePathError:
        return Texture(name=name, pixels=_PLACEHOLDER_PIXELS.copy(), srgb=True)
    try:
        with image_module.open(BytesIO(resolved.read_bytes())) as opened:
            pil = opened.convert("RGBA")
            if pil.width > MAX_TEXTURE_DIMENSION or pil.height > MAX_TEXTURE_DIMENSION:
                raise MalformedAssetError(
                    f"texture {name!r} exceeds MAX_TEXTURE_DIMENSION "
                    f"({pil.width}x{pil.height})"
                )
            pixels = np.array(pil, dtype=np.uint8)
            return Texture(name=name, pixels=pixels, srgb=True)
    except (OSError, unidentified_error, MalformedAssetError, ValueError):
        return Texture(name=name, pixels=_PLACEHOLDER_PIXELS.copy(), srgb=True)


# ----- vertex buffers -----------------------------------------------------
def _pack_vertex_buffers(doc: PmxDocument) -> _VertexBuffers:
    n = len(doc.vertices)
    positions = np.empty((n, 3), dtype=np.float32)
    normals = np.empty((n, 3), dtype=np.float32)
    texcoords = np.empty((n, 2), dtype=np.float32)
    joints = np.zeros((n, 4), dtype=np.uint16)
    weights = np.zeros((n, 4), dtype=np.float32)
    for i, vert in enumerate(doc.vertices):
        positions[i] = vert.position
        normals[i] = vert.normal
        texcoords[i] = vert.uv
        bones, vertex_weights = _flatten_skin(vert)
        joints[i] = bones
        weights[i] = vertex_weights
    return _VertexBuffers(
        positions=positions, normals=normals, texcoords=texcoords,
        joints=joints, weights=weights,
    )


_FlatJoints = tuple[int, int, int, int]
_FlatWeights = tuple[float, float, float, float]


def _flatten_skin(vert: PmxVertex) -> tuple[_FlatJoints, _FlatWeights]:
    """Map a PMX deform record onto the engine's 4-bone joints/weights layout.

    SDEF retains the BDEF2-equivalent bone pair + linear weight; the SDEF
    correction terms ``c`` / ``r0`` / ``r1`` are dropped here and revisited
    when SDEF skinning lands on the GPU side.
    """
    deform = vert.deform
    if isinstance(deform, PmxBdef1):
        return (max(deform.bone, 0), 0, 0, 0), (1.0, 0.0, 0.0, 0.0)
    if isinstance(deform, PmxBdef2 | PmxSdef):
        return (
            (max(deform.bone1, 0), max(deform.bone2, 0), 0, 0),
            (deform.weight1, 1.0 - deform.weight1, 0.0, 0.0),
        )
    if isinstance(deform, PmxBdef4 | PmxQdef):
        return (
            tuple(max(b, 0) for b in deform.bones),       # type: ignore[return-value]
            tuple(deform.weights),                        # type: ignore[return-value]
        )
    raise MalformedAssetError(f"unknown deform payload {type(deform).__name__}")


# ----- meshes -------------------------------------------------------------
def _build_meshes(doc: PmxDocument, buffers: _VertexBuffers, toon_offset: int) -> list[Mesh]:
    """Split the flat face buffer into one :class:`Mesh` per material.

    Every output mesh aliases the same vertex arrays via numpy views — the
    renderer still gets distinct draw calls per material (different indices,
    different uniforms) without paying for vertex-data duplication. Each
    Mesh also carries an :class:`MMDMaterial` derived from the PMX material
    record so the renderer can route it through the toon pass.
    """
    indices_array = np.asarray(doc.indices, dtype=np.uint32)
    meshes: list[Mesh] = []
    cursor = 0
    for mat_index, material in enumerate(doc.materials):
        end = cursor + material.face_index_count
        if end > indices_array.size:
            raise MalformedAssetError(
                f"material {mat_index} face_index_count exceeds total indices "
                f"({end} > {indices_array.size})"
            )
        slice_indices = indices_array[cursor:end].copy()
        cursor = end
        meshes.append(
            Mesh(
                name=material.name_jp or f"mat_{mat_index}",
                positions=buffers.positions,
                indices=slice_indices,
                normals=buffers.normals,
                texcoords_0=buffers.texcoords,
                joints_0=buffers.joints,
                weights_0=buffers.weights,
                base_color=material.diffuse,
                base_color_texture_index=(
                    material.texture_index if material.texture_index >= 0 else None
                ),
                mmd_material=_to_mmd_material(material, toon_offset),
            )
        )
    if cursor != indices_array.size:
        raise MalformedAssetError(
            f"materials covered {cursor} indices, file has {indices_array.size}"
        )
    return meshes


def _to_mmd_material(material: PmxMaterial, toon_offset: int) -> MMDMaterial:
    """Translate a PMX material record into the engine's :class:`MMDMaterial`."""
    sphere_index: int | None = (
        material.sphere_texture_index if material.sphere_texture_index >= 0 else None
    )
    return MMDMaterial(
        diffuse=material.diffuse,
        specular=material.specular,
        specular_power=material.specular_factor,
        ambient=material.ambient,
        edge_color=material.edge_color,
        edge_size=material.edge_size,
        sphere_texture_index=sphere_index,
        sphere_mode=SphereMode(int(material.sphere_mode)),
        toon_texture_index=_resolve_toon_index(material, toon_offset),
        flags=int(material.flags),
    )


def _resolve_toon_index(material: PmxMaterial, toon_offset: int) -> int | None:
    """Map a PMX material's toon reference to a global texture index.

    Internal toons (``toon_mode == INTERNAL``) live at ``toon_offset + 0..9``
    in :attr:`ImportedScene.textures` because the importer appends the ten
    built-in ramps after the PMX-listed textures. External toons reuse the
    existing PMX texture table.
    """
    if material.toon_mode == PmxToonMode.INTERNAL:
        if 0 <= material.toon_reference < INTERNAL_TOON_COUNT:
            return toon_offset + material.toon_reference
        return None
    if material.toon_reference >= 0:
        return material.toon_reference
    return None


# ----- bones / skin / scene ----------------------------------------------
def _build_bone_nodes(bones: tuple[PmxBone, ...]) -> list[Node]:
    """Materialise one :class:`Node` per PMX bone, with local TRS = local
    translation only (PMX rest pose has identity rotation and unit scale)."""
    nodes: list[Node] = []
    for index, bone in enumerate(bones):
        if bone.parent_index >= 0 and bone.parent_index < len(bones):
            parent_position = np.asarray(bones[bone.parent_index].position, dtype=np.float32)
        else:
            parent_position = np.zeros(3, dtype=np.float32)
        local = np.asarray(bone.position, dtype=np.float32) - parent_position
        nodes.append(
            Node(
                name=bone.name_jp or f"bone_{index}",
                transform=Transform(
                    translation=local.astype(np.float32),
                    rotation=quat_identity(),
                    scale=vec3(1.0, 1.0, 1.0),
                ),
            )
        )
    return nodes


def _attach_bone_hierarchy(bones: tuple[PmxBone, ...], nodes: list[Node]) -> list[Node]:
    """Wire each bone-node under its parent. Returns the root bones (those
    whose ``parent_index`` is negative or out of range)."""
    roots: list[Node] = []
    for index, bone in enumerate(bones):
        node = nodes[index]
        if 0 <= bone.parent_index < len(nodes):
            nodes[bone.parent_index].add_child(node)
        else:
            roots.append(node)
    return roots


def _build_skin(bones: tuple[PmxBone, ...], nodes: list[Node]) -> Skin | None:
    """Build a :class:`Skin` whose IBMs translate each bone back to the
    origin (PMX rest pose has identity rotation, so the inverse-bind world
    matrix is just ``T(-position)``)."""
    if not bones:
        return None
    ibms = np.tile(np.eye(4, dtype=np.float32), (len(bones), 1, 1))
    for i, bone in enumerate(bones):
        ibms[i, 0, 3] = -float(bone.position[0])
        ibms[i, 1, 3] = -float(bone.position[1])
        ibms[i, 2, 3] = -float(bone.position[2])
    return Skin(
        name="skin",
        joints=tuple(nodes),
        inverse_bind_matrices=ibms,
    )


def _build_scene(
    doc: PmxDocument,
    meshes: list[Mesh],
    skin: Skin | None,
    bone_nodes: list[Node],
) -> Scene:
    """Build the scene graph: a root with one model node + the bone tree."""
    name = doc.names.name_jp or doc.names.name_en or "pmx"
    scene = Scene(name=name)
    model_node = Node(name=f"{name}_mesh")
    if meshes:
        model_node.add_component(
            MeshRefComponent(mesh_indices=tuple(range(len(meshes))))
        )
    if skin is not None:
        model_node.add_component(SkinRefComponent(skin=skin))
    scene.root.add_child(model_node)
    for root in _attach_bone_hierarchy(doc.bones, bone_nodes):
        scene.root.add_child(root)
    return scene


# ----- morph asset --------------------------------------------------------
_UV_MORPH_TYPES = {
    PmxMorphType.UV: 0,
    PmxMorphType.UV1: 1,
    PmxMorphType.UV2: 2,
    PmxMorphType.UV3: 3,
    PmxMorphType.UV4: 4,
}


def _build_morph_asset(pmx_morphs: tuple[PmxMorph, ...]) -> MorphAsset:
    """Translate every PMX morph into the engine schema + wrap in a lookup."""
    return build_morph_asset(tuple(_to_engine_morph(m) for m in pmx_morphs))


def _to_engine_morph(pmx_morph: PmxMorph) -> Morph:
    """Translate one PMX morph record into its engine-side dataclass.

    UV morphs share a single builder across the five UV channel variants
    (``UV``, ``UV1``..``UV4``); the rest dispatch through
    :data:`_MORPH_BUILDERS` so this function stays a single linear path.
    """
    panel = MorphPanel(int(pmx_morph.panel))
    morph_type = pmx_morph.morph_type
    if morph_type in _UV_MORPH_TYPES:
        return UvMorph(
            name=pmx_morph.name_jp, panel=panel,
            channel=_UV_MORPH_TYPES[morph_type],
            offsets=tuple(_to_uv_offset(o) for o in pmx_morph.offsets),
        )
    builder = _MORPH_BUILDERS.get(morph_type, _build_impulse_morph)
    return builder(pmx_morph, panel)


def _build_group_morph(pmx_morph: PmxMorph, panel: MorphPanel) -> GroupMorph:
    return GroupMorph(
        name=pmx_morph.name_jp, panel=panel,
        children=tuple(_to_group_child(o) for o in pmx_morph.offsets),
    )


def _build_vertex_morph(pmx_morph: PmxMorph, panel: MorphPanel) -> VertexMorph:
    return VertexMorph(
        name=pmx_morph.name_jp, panel=panel,
        offsets=tuple(_to_vertex_offset(o) for o in pmx_morph.offsets),
    )


def _build_bone_morph(pmx_morph: PmxMorph, panel: MorphPanel) -> BoneMorph:
    return BoneMorph(
        name=pmx_morph.name_jp, panel=panel,
        offsets=tuple(_to_bone_offset(o) for o in pmx_morph.offsets),
    )


def _build_material_morph(pmx_morph: PmxMorph, panel: MorphPanel) -> MaterialMorph:
    return MaterialMorph(
        name=pmx_morph.name_jp, panel=panel,
        targets=tuple(_to_material_target(o) for o in pmx_morph.offsets),
    )


def _build_flip_morph(pmx_morph: PmxMorph, panel: MorphPanel) -> FlipMorph:
    return FlipMorph(
        name=pmx_morph.name_jp, panel=panel,
        children=tuple(_to_group_child(o) for o in pmx_morph.offsets),
    )


def _build_impulse_morph(pmx_morph: PmxMorph, panel: MorphPanel) -> ImpulseMorph:
    return ImpulseMorph(
        name=pmx_morph.name_jp, panel=panel,
        offsets=tuple(_to_impulse_offset(o) for o in pmx_morph.offsets),
    )


_MORPH_BUILDERS = {
    PmxMorphType.GROUP: _build_group_morph,
    PmxMorphType.VERTEX: _build_vertex_morph,
    PmxMorphType.BONE: _build_bone_morph,
    PmxMorphType.MATERIAL: _build_material_morph,
    PmxMorphType.FLIP: _build_flip_morph,
    PmxMorphType.IMPULSE: _build_impulse_morph,
}


def _to_group_child(offset: PmxGroupMorphOffset | PmxFlipMorphOffset) -> GroupMorphChild:
    return GroupMorphChild(morph_index=int(offset.morph_index), weight=float(offset.weight))


def _to_vertex_offset(offset: PmxVertexMorphOffset) -> VertexMorphOffset:
    return VertexMorphOffset(vertex_index=int(offset.vertex_index), offset=offset.offset)


def _to_bone_offset(offset: PmxBoneMorphOffset) -> BoneMorphOffset:
    return BoneMorphOffset(
        bone_index=int(offset.bone_index),
        translation=offset.translation,
        rotation=offset.rotation,
    )


def _to_uv_offset(offset: PmxUvMorphOffset) -> UvMorphOffset:
    return UvMorphOffset(vertex_index=int(offset.vertex_index), offset=offset.offset)


def _to_material_target(offset: PmxMaterialMorphOffset) -> MaterialMorphTarget:
    return MaterialMorphTarget(
        material_index=int(offset.material_index),
        op=MaterialMorphOp(int(offset.op)),
        diffuse=offset.diffuse,
        specular=offset.specular,
        specular_power=float(offset.specular_factor),
        ambient=offset.ambient,
        edge_color=offset.edge_color,
        edge_size=float(offset.edge_size),
        texture_coef=offset.texture_coef,
        sphere_coef=offset.sphere_coef,
        toon_coef=offset.toon_coef,
    )


def _to_impulse_offset(offset: PmxImpulseMorphOffset) -> ImpulseMorphOffset:
    return ImpulseMorphOffset(
        rigid_body_index=int(offset.rigid_body_index),
        is_local=bool(offset.is_local),
        velocity=offset.velocity,
        torque=offset.torque,
    )


# ----- display-frame groups ---------------------------------------------
def _build_display_frame_groups(
    pmx_frames: tuple[PmxDisplayFrame, ...],
) -> tuple[DisplayFrameGroup, ...]:
    """Translate PMX display-frame panels to engine-side groups.

    PMX stores element ``kind`` as ``0`` for bone, ``1`` for morph;
    we mirror that into :class:`DisplayFrameElementKind`. Groups whose
    members reference unknown indices are still kept — the timeline UI
    skips invalid entries, but a PMX-edited model that lost a bone
    after publishing its panel layout shouldn't break the importer.
    """
    return tuple(
        DisplayFrameGroup(
            name=frame.name_jp,
            is_special=bool(frame.is_special),
            elements=tuple(
                DisplayFrameElement(
                    kind=DisplayFrameElementKind(int(element.kind)),
                    index=int(element.index),
                )
                for element in frame.elements
            ),
        )
        for frame in pmx_frames
    )


# ----- physics scene -----------------------------------------------------
def _build_physics_scene(
    pmx_bodies: tuple[PmxRigidBody, ...],
    pmx_joints: tuple[PmxJoint, ...],
) -> PhysicsScene:
    """Translate every PMX rigid body / joint to the engine schema."""
    bodies = tuple(_to_engine_rigid_body(body) for body in pmx_bodies)
    joints = tuple(_to_engine_joint(joint) for joint in pmx_joints)
    return PhysicsScene(bodies=bodies, joints=joints)


def _to_engine_rigid_body(body: PmxRigidBody) -> RigidBody:
    return RigidBody(
        name=body.name_jp,
        bone_index=int(body.related_bone_index),
        group=int(body.group),
        non_collision_mask=int(body.non_collision_mask),
        shape=RigidShape(int(body.shape)),
        size=body.size,
        position=body.position,
        rotation=body.rotation,
        mass=float(body.mass),
        linear_damping=float(body.linear_damping),
        angular_damping=float(body.angular_damping),
        restitution=float(body.restitution),
        friction=float(body.friction),
        physics_mode=PhysicsMode(int(body.physics_mode)),
    )


def _to_engine_joint(joint: PmxJoint) -> Joint6DofSpring:
    return Joint6DofSpring(
        name=joint.name_jp,
        rigid_a_index=int(joint.rigid_a_index),
        rigid_b_index=int(joint.rigid_b_index),
        position=joint.position,
        rotation=joint.rotation,
        linear_lower=joint.linear_lower,
        linear_upper=joint.linear_upper,
        angular_lower=joint.angular_lower,
        angular_upper=joint.angular_upper,
        spring_linear=joint.spring_linear,
        spring_angular=joint.spring_angular,
    )


# ----- bone resolver rules ----------------------------------------------
def _build_bone_resolver_rules(
    pmx_bones: tuple[PmxBone, ...],
) -> BoneResolverRules:
    """Extract append + fixed-axis rules + the deformation-order list.

    PMX bones store an explicit ``deformation_depth`` integer; the
    resolver walks bones in ascending depth order, with stable ties
    broken by index so the same model always resolves identically.
    """
    indexed = sorted(
        range(len(pmx_bones)),
        key=lambda i: (int(pmx_bones[i].deformation_depth), i),
    )
    appends: list[BoneAppendRule] = []
    fixed_axes: list[FixedAxisRule] = []
    for index, bone in enumerate(pmx_bones):
        rule = _bone_append_rule(index, bone)
        if rule is not None:
            appends.append(rule)
        if bone.flags & PMX_BONE_FLAG_FIXED_AXIS and bone.fixed_axis is not None:
            fixed_axes.append(
                FixedAxisRule(bone_index=index, axis=tuple(bone.fixed_axis)),
            )
    return BoneResolverRules(
        deformation_order=tuple(indexed),
        appends=tuple(appends),
        fixed_axes=tuple(fixed_axes),
    )


def _bone_append_rule(index: int, bone: PmxBone) -> BoneAppendRule | None:
    """Return an append rule for a PMX bone with inherit flags set."""
    inherit_rotation = bool(bone.flags & PMX_BONE_FLAG_INHERIT_ROTATION)
    inherit_translation = bool(bone.flags & PMX_BONE_FLAG_INHERIT_TRANSLATION)
    if not (inherit_rotation or inherit_translation):
        return None
    if bone.inherit_parent_index is None or bone.inherit_weight is None:
        return None
    return BoneAppendRule(
        bone_index=index,
        parent_index=int(bone.inherit_parent_index),
        weight=float(bone.inherit_weight),
        inherit_rotation=inherit_rotation,
        inherit_translation=inherit_translation,
    )


# ----- IK chains ---------------------------------------------------------
def _build_ik_chains(pmx_bones: tuple[PmxBone, ...]) -> tuple[IkChain, ...]:
    """Walk every PMX bone whose ``ik`` is set and return the chains.

    The bone that owns the IK definition becomes the chain's *driver*; the
    PMX-stored ``target_bone_index`` becomes the *effector* (the bone
    whose world position should match the driver after solving).
    """
    chains: list[IkChain] = []
    for driver_index, bone in enumerate(pmx_bones):
        ik = bone.ik
        if ik is None:
            continue
        chains.append(
            IkChain(
                driver_bone_index=driver_index,
                effector_bone_index=int(ik.target_bone_index),
                iterations=int(ik.iterations),
                limit_radian=float(ik.limit_radian),
                links=tuple(
                    IkLink(
                        bone_index=int(link.bone_index),
                        has_limit=bool(link.has_limit),
                        limit_min=link.limit_min,
                        limit_max=link.limit_max,
                    )
                    for link in ik.links
                ),
            )
        )
    return tuple(chains)
