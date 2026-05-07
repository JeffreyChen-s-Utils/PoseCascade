"""Apply leaf-morph weights to per-frame mesh / bone / material state.

Construction snapshots the imported model's base vertex buffer (positions
and primary UV); :meth:`MorphApplier.apply` resets those buffers each
call, then accumulates every active vertex / UV / bone / material morph's
contribution. Bone and material results are returned as a
:class:`MorphSnapshot` for the player + renderer to consume; vertex / UV
results live on the applier itself and are streamed to the renderer via
:meth:`MorphApplier.stream_to_renderer`.

Phase 4 supports the five leaf morph kinds the engine actually consumes:
Vertex, Bone, UV (channel 0 only), Material, and — through the
accumulator — Group / Flip recursion. Impulse morphs are stored on the
asset but skipped here; Phase 7 picks them up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from posecascade.animation.morph import (
    BoneMorph,
    MaterialMorph,
    MaterialMorphOp,
    MaterialMorphTarget,
    MorphAsset,
    UvMorph,
    VertexMorph,
)
from posecascade.animation.morph_accumulator import LeafWeights
from posecascade.assets.types import ImportedScene
from posecascade.render.material import MMDMaterial
from posecascade.utils.math3d import (
    Quat,
    Vec3,
    quat_identity,
    quat_mul,
    quat_slerp,
    vec3,
)

_PRIMARY_UV_CHANNEL = 0
_WILDCARD_MATERIAL = -1
_DEFAULT_MUL_VEC4 = (1.0, 1.0, 1.0, 1.0)
_DEFAULT_MUL_VEC3 = (1.0, 1.0, 1.0)
_DEFAULT_ADD_VEC4 = (0.0, 0.0, 0.0, 0.0)
_DEFAULT_ADD_VEC3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MorphSnapshot:
    """One frame's computed bone offsets + material overrides.

    ``bone_offsets[i] = (translation_offset, rotation_offset)`` describes a
    delta the player composes onto the VMD-driven bone TRS. The rotation
    is a quaternion already scaled by morph weight (slerp from identity).

    ``material_overrides[material_index] = MMDMaterial`` is the effective
    material after every active material morph; absent indices keep their
    base material as defined by the importer.
    """

    bone_offsets: dict[int, tuple[Vec3, Quat]] = field(default_factory=dict)
    material_overrides: dict[int, MMDMaterial] = field(default_factory=dict)


@dataclass
class _MaterialDelta:
    """Accumulator for one material's mul / add chain across active morphs."""

    diffuse_mul: tuple[float, float, float, float] = _DEFAULT_MUL_VEC4
    diffuse_add: tuple[float, float, float, float] = _DEFAULT_ADD_VEC4
    specular_mul: tuple[float, float, float] = _DEFAULT_MUL_VEC3
    specular_add: tuple[float, float, float] = _DEFAULT_ADD_VEC3
    specular_power_mul: float = 1.0
    specular_power_add: float = 0.0
    ambient_mul: tuple[float, float, float] = _DEFAULT_MUL_VEC3
    ambient_add: tuple[float, float, float] = _DEFAULT_ADD_VEC3
    edge_color_mul: tuple[float, float, float, float] = _DEFAULT_MUL_VEC4
    edge_color_add: tuple[float, float, float, float] = _DEFAULT_ADD_VEC4
    edge_size_mul: float = 1.0
    edge_size_add: float = 0.0


class MorphApplier:
    """Holds base vertex buffers and computes per-frame morph state.

    Bound to one :class:`ImportedScene` — PMX models share their position
    / texcoord arrays across every material-Mesh, so a single base
    snapshot here covers every drawn primitive.
    """

    def __init__(self, scene: ImportedScene) -> None:
        self.scene = scene
        self.asset: MorphAsset = scene.morphs
        first_mesh = scene.meshes[0] if scene.meshes else None
        if first_mesh is None:
            self._base_positions = np.zeros((0, 3), dtype=np.float32)
            self._base_texcoords = np.zeros((0, 2), dtype=np.float32)
        else:
            self._base_positions = np.asarray(
                first_mesh.positions, dtype=np.float32,
            ).copy()
            self._base_texcoords = (
                np.asarray(first_mesh.texcoords_0, dtype=np.float32).copy()
                if first_mesh.texcoords_0 is not None
                else np.zeros((self._base_positions.shape[0], 2), dtype=np.float32)
            )
        self._current_positions = self._base_positions.copy()
        self._current_texcoords = self._base_texcoords.copy()
        self._dirty_positions = False
        self._dirty_texcoords = False

    @property
    def current_positions(self) -> NDArray[np.float32]:
        return self._current_positions

    @property
    def current_texcoords(self) -> NDArray[np.float32]:
        return self._current_texcoords

    @property
    def positions_dirty(self) -> bool:
        """``True`` until :meth:`mark_uploaded` runs after a mutating apply."""
        return self._dirty_positions

    @property
    def texcoords_dirty(self) -> bool:
        return self._dirty_texcoords

    def mark_uploaded(self) -> None:
        """Renderer calls this after streaming the dirty buffers to the GPU."""
        self._dirty_positions = False
        self._dirty_texcoords = False

    def apply(self, weights: LeafWeights) -> MorphSnapshot:
        """Reset, accumulate every active morph, and return the snapshot."""
        np.copyto(self._current_positions, self._base_positions)
        np.copyto(self._current_texcoords, self._base_texcoords)
        # Even when no morphs fire we still mark dirty so the renderer's
        # first uploads after a re-bind catch the rest pose.
        self._dirty_positions = True
        self._dirty_texcoords = True

        bone_offsets: dict[int, tuple[Vec3, Quat]] = {}
        material_deltas: dict[int, _MaterialDelta] = {}

        for index, weight in weights.weights.items():
            if index < 0 or index >= len(self.asset.by_index):
                continue
            morph = self.asset.by_index[index]
            self._dispatch(morph, float(weight), bone_offsets, material_deltas)

        material_overrides = self._materialise_overrides(material_deltas)
        return MorphSnapshot(
            bone_offsets=bone_offsets,
            material_overrides=material_overrides,
        )

    # ----- per-type appliers ------------------------------------------
    def _dispatch(
        self,
        morph,                                         # noqa: ANN001 — runtime type-dispatched
        weight: float,
        bone_offsets: dict[int, tuple[Vec3, Quat]],
        material_deltas: dict[int, _MaterialDelta],
    ) -> None:
        if isinstance(morph, VertexMorph):
            self._apply_vertex(morph, weight)
        elif isinstance(morph, UvMorph) and morph.channel == _PRIMARY_UV_CHANNEL:
            self._apply_uv(morph, weight)
        elif isinstance(morph, BoneMorph):
            self._apply_bone(morph, weight, bone_offsets)
        elif isinstance(morph, MaterialMorph):
            self._apply_material(morph, weight, material_deltas)

    def _apply_vertex(self, morph: VertexMorph, weight: float) -> None:
        for offset in morph.offsets:
            if 0 <= offset.vertex_index < self._current_positions.shape[0]:
                self._current_positions[offset.vertex_index] += np.asarray(
                    offset.offset, dtype=np.float32,
                ) * weight

    def _apply_uv(self, morph: UvMorph, weight: float) -> None:
        for offset in morph.offsets:
            if 0 <= offset.vertex_index < self._current_texcoords.shape[0]:
                self._current_texcoords[offset.vertex_index, 0] += offset.offset[0] * weight
                self._current_texcoords[offset.vertex_index, 1] += offset.offset[1] * weight

    def _apply_bone(
        self,
        morph: BoneMorph,
        weight: float,
        bone_offsets: dict[int, tuple[Vec3, Quat]],
    ) -> None:
        for offset in morph.offsets:
            current = bone_offsets.get(offset.bone_index)
            scaled_t = vec3(*(c * weight for c in offset.translation))
            scaled_r = quat_slerp(
                quat_identity(),
                np.asarray(offset.rotation, dtype=np.float32),
                weight,
            )
            if current is None:
                bone_offsets[offset.bone_index] = (scaled_t, scaled_r)
                continue
            combined_t = (current[0] + scaled_t).astype(np.float32, copy=False)
            combined_r = quat_mul(scaled_r, current[1])
            bone_offsets[offset.bone_index] = (combined_t, combined_r)

    def _apply_material(
        self,
        morph: MaterialMorph,
        weight: float,
        material_deltas: dict[int, _MaterialDelta],
    ) -> None:
        for target in morph.targets:
            if target.material_index == _WILDCARD_MATERIAL:
                for mat_index in range(len(self.scene.meshes)):
                    self._merge_target_into(material_deltas, mat_index, target, weight)
                continue
            self._merge_target_into(
                material_deltas, target.material_index, target, weight,
            )

    def _merge_target_into(
        self,
        material_deltas: dict[int, _MaterialDelta],
        material_index: int,
        target: MaterialMorphTarget,
        weight: float,
    ) -> None:
        delta = material_deltas.get(material_index) or _MaterialDelta()
        if target.op == MaterialMorphOp.MULTIPLY:
            delta = _multiply_into(delta, target, weight)
        else:
            delta = _add_into(delta, target, weight)
        material_deltas[material_index] = delta

    # ----- conversion -------------------------------------------------
    def _materialise_overrides(
        self,
        material_deltas: dict[int, _MaterialDelta],
    ) -> dict[int, MMDMaterial]:
        out: dict[int, MMDMaterial] = {}
        for material_index, delta in material_deltas.items():
            if material_index < 0 or material_index >= len(self.scene.meshes):
                continue
            mesh = self.scene.meshes[material_index]
            base = mesh.mmd_material
            if base is None:
                continue
            out[material_index] = _compose_material(base, delta)
        return out


# ----- helper combiners --------------------------------------------------
def _lerp_one(weight: float, target: float) -> float:
    """Multiplicative-morph factor: ``1`` when ``weight = 0``, ``target``
    when ``weight = 1`` — chained component-wise across morphs."""
    return 1.0 + (target - 1.0) * weight


def _multiply_into(
    delta: _MaterialDelta, target: MaterialMorphTarget, weight: float,
) -> _MaterialDelta:
    return _MaterialDelta(
        diffuse_mul=_compose_mul_vec(delta.diffuse_mul, target.diffuse, weight),
        diffuse_add=delta.diffuse_add,
        specular_mul=_compose_mul_vec(delta.specular_mul, target.specular, weight),
        specular_add=delta.specular_add,
        specular_power_mul=delta.specular_power_mul * _lerp_one(weight, target.specular_power),
        specular_power_add=delta.specular_power_add,
        ambient_mul=_compose_mul_vec(delta.ambient_mul, target.ambient, weight),
        ambient_add=delta.ambient_add,
        edge_color_mul=_compose_mul_vec(delta.edge_color_mul, target.edge_color, weight),
        edge_color_add=delta.edge_color_add,
        edge_size_mul=delta.edge_size_mul * _lerp_one(weight, target.edge_size),
        edge_size_add=delta.edge_size_add,
    )


def _add_into(
    delta: _MaterialDelta, target: MaterialMorphTarget, weight: float,
) -> _MaterialDelta:
    return _MaterialDelta(
        diffuse_mul=delta.diffuse_mul,
        diffuse_add=_compose_add_vec(delta.diffuse_add, target.diffuse, weight),
        specular_mul=delta.specular_mul,
        specular_add=_compose_add_vec(delta.specular_add, target.specular, weight),
        specular_power_mul=delta.specular_power_mul,
        specular_power_add=delta.specular_power_add + target.specular_power * weight,
        ambient_mul=delta.ambient_mul,
        ambient_add=_compose_add_vec(delta.ambient_add, target.ambient, weight),
        edge_color_mul=delta.edge_color_mul,
        edge_color_add=_compose_add_vec(delta.edge_color_add, target.edge_color, weight),
        edge_size_mul=delta.edge_size_mul,
        edge_size_add=delta.edge_size_add + target.edge_size * weight,
    )


def _compose_mul_vec(current: tuple, target: tuple, weight: float) -> tuple:
    return tuple(c * _lerp_one(weight, t) for c, t in zip(current, target, strict=True))


def _compose_add_vec(current: tuple, target: tuple, weight: float) -> tuple:
    return tuple(c + t * weight for c, t in zip(current, target, strict=True))


def _compose_material(base: MMDMaterial, delta: _MaterialDelta) -> MMDMaterial:
    """Apply ``base * mul + add`` to every channel of ``base``."""
    return MMDMaterial(
        diffuse=_apply_mul_add_vec(base.diffuse, delta.diffuse_mul, delta.diffuse_add),
        specular=_apply_mul_add_vec(base.specular, delta.specular_mul, delta.specular_add),
        specular_power=base.specular_power * delta.specular_power_mul + delta.specular_power_add,
        ambient=_apply_mul_add_vec(base.ambient, delta.ambient_mul, delta.ambient_add),
        edge_color=_apply_mul_add_vec(
            base.edge_color, delta.edge_color_mul, delta.edge_color_add,
        ),
        edge_size=base.edge_size * delta.edge_size_mul + delta.edge_size_add,
        sphere_texture_index=base.sphere_texture_index,
        sphere_mode=base.sphere_mode,
        toon_texture_index=base.toon_texture_index,
        flags=base.flags,
    )


def _apply_mul_add_vec(base: tuple, mul: tuple, add: tuple) -> tuple:
    return tuple(b * m + a for b, m, a in zip(base, mul, add, strict=True))


__all__ = [
    "MorphApplier",
    "MorphSnapshot",
]
