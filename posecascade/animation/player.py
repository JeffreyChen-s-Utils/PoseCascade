"""Animation player — drives a scene's bone Nodes + morphs from a VMD motion.

Construction snapshots each bone Node's *rest* TRS so the VMD's stored
position can be re-applied as an offset rather than a replacement (VMD
records OFFSET-from-rest values, not absolute world positions).
Per-frame :meth:`apply` writes the eased translation/rotation back onto
the Node's :class:`~posecascade.scene.transform.Transform`, then composes
bone-morph deltas on top and streams vertex / UV / material morph state
through the renderer when one is attached.

The player is deliberately not Qt-aware. UI code — see
:mod:`posecascade.ui.timeline_basic` — owns its own QTimer and calls
:meth:`apply` on each tick.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from vmd.types import VMD_FRAMES_PER_SECOND

from posecascade.animation.bone_resolver import BoneResolver, BoneResolverRules
from posecascade.animation.ik import IkChain, solve_chain
from posecascade.animation.morph_accumulator import accumulate_indexed_weights
from posecascade.animation.morph_apply import MorphApplier
from posecascade.animation.skeleton import Skeleton
from posecascade.animation.vmd_track import VmdMotionAsset, vmd_bone_key
from posecascade.assets.types import ImportedScene, Skin
from posecascade.physics.world import PhysicsWorld
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.utils.math3d import Quat, Vec3, quat_identity, quat_mul, vec3


@dataclass(frozen=True)
class _RestPose:
    """A bone's rest TRS captured at player-construction time."""

    translation: Vec3
    rotation: Quat


class VmdAnimationPlayer:
    """Maps a :class:`VmdMotionAsset` onto a set of bone :class:`Node`s.

    The lookup table is built once at construction; each call to
    :meth:`apply` walks the motion's tracks, samples the eased
    ``(translation, rotation)`` per bone, and writes them onto the Node
    transform as ``rest_translation + offset`` / ``vmd_rotation``.

    Optional ``scene`` enables morph driving: vertex / UV morphs stream
    new buffers through ``renderer`` (when one is attached), bone morphs
    compose on top of the VMD bone TRS, and material morphs land in the
    renderer's override map. Construct via :meth:`for_imported_scene` to
    enable both bone and morph paths in one call.
    """

    def __init__(
        self,
        motion: VmdMotionAsset,
        joint_lookup: dict[str, Node],
        scene: ImportedScene | None = None,
        renderer: object | None = None,
        bone_index_to_node: dict[int, Node] | None = None,
        ik_chains: tuple[IkChain, ...] = (),
        resolver_rules: BoneResolverRules | None = None,
    ) -> None:
        self._motion = motion
        self._lookup = joint_lookup
        self._rest = {
            key: _RestPose(
                translation=node.transform.translation.copy(),
                rotation=node.transform.rotation.copy(),
            )
            for key, node in joint_lookup.items()
        }
        self._morph_applier: MorphApplier | None = None
        self._morph_resolver: dict[str, int] = {}
        self._bone_index_to_node = dict(bone_index_to_node or {})
        self._renderer = renderer
        if scene is not None and scene.morphs.by_index:
            self._morph_applier = MorphApplier(scene)
            self._morph_resolver = {
                vmd_bone_key(morph.name): index
                for index, morph in enumerate(scene.morphs.by_index)
            }
        # IK plumbing: a VMD ``IK`` keyframe addresses the chain by its
        # *driver* bone's name (truncated to 15 SJIS bytes). We pre-build
        # the driver-index → name-key map so per-frame application is just
        # a lookup + boolean toggle.
        self._ik_chains: tuple[IkChain, ...] = ik_chains
        self._ik_resolver: dict[str, IkChain] = {}
        if ik_chains:
            self._ik_resolver = {
                vmd_bone_key(self._bone_index_to_node[chain.driver_bone_index].name): chain
                for chain in ik_chains
                if chain.driver_bone_index in self._bone_index_to_node
            }
        # Bone post-IK resolver (append + fixed-axis). The resolver itself
        # snapshots rest TRS at construction time; we only build it when
        # the model has at least one rule, so unrelated imports incur no
        # per-frame walk.
        self._bone_resolver: BoneResolver | None = None
        if resolver_rules is not None and (
            resolver_rules.appends or resolver_rules.fixed_axes
        ):
            self._bone_resolver = BoneResolver.from_rules(
                resolver_rules, self._bone_index_to_node,
            )
        # Physics: instantiated when the imported scene has at least one
        # rigid body. We track the previous apply-time so :meth:`apply`
        # can compute ``dt`` for the simulator and trigger a reset on
        # large gaps (timeline scrub).
        self._physics_world: PhysicsWorld | None = None
        self._last_apply_time: float | None = None
        if scene is not None and scene.physics_scene.bodies:
            self._physics_world = PhysicsWorld(
                scene=scene.physics_scene,
                bone_index_to_node=self._bone_index_to_node,
            )

    @classmethod
    def for_skin(cls, motion: VmdMotionAsset, skin: Skin) -> VmdAnimationPlayer:
        """Build a player whose lookup table is keyed on every bone of ``skin``.

        Bone-only — morphs require a full :class:`ImportedScene`; use
        :meth:`for_imported_scene` for that.
        """
        return cls(motion, build_skin_lookup(skin))

    @classmethod
    def for_skeleton(
        cls, motion: VmdMotionAsset, skeleton: Skeleton, scene: Scene,
    ) -> VmdAnimationPlayer:
        """Build a player from a flat :class:`Skeleton` resolved against the scene's nodes."""
        return cls(motion, build_skeleton_lookup(skeleton, scene))

    @classmethod
    def for_imported_scene(
        cls,
        motion: VmdMotionAsset,
        imported: ImportedScene,
        renderer: object | None = None,
    ) -> VmdAnimationPlayer:
        """Build a fully-wired player: bone TRS + morph apply + IK + renderer streaming.

        ``renderer`` may be ``None`` for unit tests; bone, material morph,
        and IK state still compute correctly and the bone Node TRS still
        composes — only the GPU-side stream calls go silent.
        """
        skin = imported.skins[0] if imported.skins else None
        joint_lookup = build_skin_lookup(skin) if skin is not None else {}
        bone_index_to_node = (
            {i: joint for i, joint in enumerate(skin.joints) if isinstance(joint, Node)}
            if skin is not None else {}
        )
        return cls(
            motion=motion,
            joint_lookup=joint_lookup,
            scene=imported,
            renderer=renderer,
            bone_index_to_node=bone_index_to_node,
            ik_chains=imported.ik_chains,
            resolver_rules=imported.bone_resolver_rules,
        )

    @property
    def duration_seconds(self) -> float:
        return self._motion.duration_frames / VMD_FRAMES_PER_SECOND

    def apply(self, time_seconds: float) -> None:
        """Apply the motion at ``time_seconds`` to bones, morphs, IK, materials, physics."""
        frame = max(0.0, time_seconds * VMD_FRAMES_PER_SECOND)
        self._apply_bone_tracks(frame)
        if self._morph_applier is not None:
            self._apply_morph_tracks(frame)
        if self._ik_chains:
            self._apply_ik(frame)
        if self._bone_resolver is not None:
            self._bone_resolver.resolve()
        if self._physics_world is not None:
            self._step_physics(time_seconds)

    def _step_physics(self, time_seconds: float) -> None:
        """Advance the physics world by ``time - last_apply_time``.

        First call after construction (or after :meth:`reset_physics`)
        skips integration: the bones are mid-resolution at construction
        time and we want the world to capture them as the rest pose,
        not integrate from a phantom dt.
        """
        if self._physics_world is None:
            return
        if self._last_apply_time is None:
            self._physics_world.reset()
            self._last_apply_time = time_seconds
            return
        dt = time_seconds - self._last_apply_time
        self._physics_world.step(dt)
        self._last_apply_time = time_seconds

    def reset_physics(self, *, warmup_steps: int = 0) -> None:
        """Snap every dynamic body back to its kinematic baseline."""
        if self._physics_world is None:
            return
        self._physics_world.reset(warmup_steps=warmup_steps)
        self._last_apply_time = None

    def _apply_ik(self, frame: float) -> None:
        """Sample the VMD IK enable tracks, then run CCD on every active chain."""
        for track in self._motion.ik_tracks:
            chain = self._ik_resolver.get(track.name_key)
            if chain is None:
                continue
            chain.enabled = track.sample(frame)
        for chain in self._ik_chains:
            if chain.enabled:
                solve_chain(chain, self._bone_index_to_node)

    def _apply_bone_tracks(self, frame: float) -> None:
        for track in self._motion.bone_tracks:
            node = self._lookup.get(track.name_key)
            if node is None:
                continue
            offset, rotation = track.sample(frame)
            rest = self._rest[track.name_key]
            node.transform.set_translation(
                (rest.translation + offset).astype(np.float32, copy=False)
            )
            node.transform.set_rotation(rotation.astype(np.float32, copy=False))

    def _apply_morph_tracks(self, frame: float) -> None:
        applier = self._morph_applier
        if applier is None:
            return
        indexed_weights: dict[int, float] = {}
        for track in self._motion.morph_tracks:
            asset_index = self._morph_resolver.get(track.name_key)
            if asset_index is None:
                continue
            weight = track.sample(frame)
            if weight != 0.0:
                indexed_weights[asset_index] = weight
        leaf = accumulate_indexed_weights(applier.asset, indexed_weights)
        snapshot = applier.apply(leaf)
        self._compose_bone_morph_offsets(snapshot.bone_offsets)
        self._stream_to_renderer(applier, snapshot.material_overrides)

    def _compose_bone_morph_offsets(
        self, bone_offsets: dict[int, tuple[Vec3, Quat]],
    ) -> None:
        for bone_index, (translation_offset, rotation_offset) in bone_offsets.items():
            node = self._bone_index_to_node.get(bone_index)
            if node is None:
                continue
            current_t = node.transform.translation
            current_r = node.transform.rotation
            node.transform.set_translation(
                (current_t + translation_offset).astype(np.float32, copy=False)
            )
            node.transform.set_rotation(quat_mul(rotation_offset, current_r))

    def _stream_to_renderer(
        self,
        applier: MorphApplier,
        material_overrides: dict[int, object],
    ) -> None:
        if self._renderer is None:
            return
        stream = getattr(self._renderer, "stream_morphed_buffers", None)
        if stream is not None:
            stream(applier.current_positions, applier.current_texcoords)
        set_overrides = getattr(self._renderer, "set_material_overrides", None)
        if set_overrides is not None:
            set_overrides(material_overrides)
        applier.mark_uploaded()

    def reset_to_rest(self) -> None:
        """Restore every mapped bone Node to its captured rest TRS."""
        for key, node in self._lookup.items():
            rest = self._rest[key]
            node.transform.set_translation(rest.translation.astype(np.float32, copy=False))
            node.transform.set_rotation(rest.rotation.astype(np.float32, copy=False))


def build_skin_lookup(skin: Skin) -> dict[str, Node]:
    """Build ``{vmd_bone_key: Node}`` for every joint in ``skin``."""
    out: dict[str, Node] = {}
    for joint in skin.joints:
        if not isinstance(joint, Node):
            continue
        key = vmd_bone_key(joint.name)
        out.setdefault(key, joint)
    return out


def build_skeleton_lookup(skeleton: Skeleton, scene: Scene) -> dict[str, Node]:
    """Resolve each :class:`Bone` in ``skeleton`` to a Node in ``scene``."""
    out: dict[str, Node] = {}
    for bone in skeleton.bones:
        node = scene.find(bone.name)
        if node is None:
            continue
        out[vmd_bone_key(bone.name)] = node
    return out


# Re-export so callers don't need a second import for these utility quats.
__all__ = [
    "VmdAnimationPlayer",
    "build_skeleton_lookup",
    "build_skin_lookup",
    "quat_identity",
    "vec3",
]
