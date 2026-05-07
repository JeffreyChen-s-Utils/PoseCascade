"""Bridge between :class:`ProjectFile` and the live engine state.

Two directions:

- :func:`project_from_state` — reads the in-memory
  :class:`SceneSlots` / :class:`EffectChain` / playback knobs and
  builds an immutable :class:`ProjectFile` ready for serialisation.
- :func:`load_state_from_project` — reverse: given a project + the
  importer manager, resolves every relative path through
  :func:`resolve_safe` and rebuilds :class:`SceneSlots` plus the
  effect chain by re-loading models / motions from disk.

Path safety is the responsibility of *this* module — the schema /
reader / writer don't touch the filesystem at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from posecascade.assets.path_safety import resolve_safe
from posecascade.assets.types import ImportedScene
from posecascade.project.schema import (
    ProjectAudio,
    ProjectExternalParent,
    ProjectFile,
    ProjectPlayback,
    ProjectSlot,
)
from posecascade.render.effects.chain import EffectChain, EffectLibrary
from posecascade.render.effects.loader import (
    load_chain_from_toml,
    serialize_chain_to_toml,
)
from posecascade.scene.external_parent import ExternalParentBinding
from posecascade.scene.model_slot import ModelSlot, SceneSlots


@dataclass(frozen=True)
class LoadedProjectState:
    """What :func:`load_state_from_project` hands back."""

    slots: SceneSlots
    playback: ProjectPlayback
    audio: ProjectAudio | None
    effect_chain: EffectChain


def project_from_state(
    *,
    slots: SceneSlots,
    slot_source_paths: dict[str, tuple[Path, Path | None]],
    playback: ProjectPlayback,
    audio: ProjectAudio | None,
    effect_chain: EffectChain,
    project_root: Path,
    name: str = "",
) -> ProjectFile:
    """Snapshot every live engine container into a serialisable project.

    ``slot_source_paths`` maps a slot's name to ``(model_path, motion_path)``
    — the absolute paths the integrator knows about. We re-express each
    as project-root-relative; if a path falls outside the root, the
    serialiser raises rather than silently writing an unsafe value.
    """
    project_root = project_root.resolve()
    return ProjectFile(
        name=name,
        slots=tuple(
            _slot_to_project(slot, slot_source_paths, project_root)
            for slot in slots
        ),
        playback=playback,
        audio=audio,
        effect_chain_toml=serialize_chain_to_toml(effect_chain),
    )


def load_state_from_project(
    project: ProjectFile,
    *,
    project_root: Path,
    pmx_loader,                                              # noqa: ANN001
    vmd_loader,                                              # noqa: ANN001
    effect_library: EffectLibrary,
) -> LoadedProjectState:
    """Resolve + re-load every reference in ``project``.

    ``pmx_loader(path) -> ImportedScene`` and
    ``vmd_loader(path) -> VmdMotionAsset`` are the per-format hooks the
    caller threads in (typically the engine's
    :class:`~posecascade.assets.importer_manager.ImporterManager` or
    direct importer instances). Keeping them parameters lets the
    project layer stay decoupled from the importer plugin chain.
    """
    project_root = project_root.resolve(strict=True)
    slots = SceneSlots()
    for project_slot in project.slots:
        slots.add(_slot_from_project(project_slot, project_root, pmx_loader, vmd_loader))
    chain = load_chain_from_toml(project.effect_chain_toml, effect_library) if (
        project.effect_chain_toml.strip()
    ) else EffectChain()
    return LoadedProjectState(
        slots=slots,
        playback=project.playback,
        audio=project.audio,
        effect_chain=chain,
    )


# ----- conversion helpers --------------------------------------------
def _slot_to_project(
    slot: ModelSlot,
    slot_source_paths: dict[str, tuple[Path, Path | None]],
    project_root: Path,
) -> ProjectSlot:
    """Convert one live :class:`ModelSlot` to its project-file shape."""
    paths = slot_source_paths.get(slot.name, (None, None))
    model_path, motion_path = paths
    if model_path is None:
        raise ValueError(
            f"slot {slot.name!r} has no source model path; pass it via "
            f"``slot_source_paths`` so the project file can record a "
            f"reproducible reference",
        )
    translation = tuple(float(v) for v in slot.transform.translation)
    rotation = tuple(float(v) for v in slot.transform.rotation)
    return ProjectSlot(
        name=slot.name,
        model_path=_relative_to_root(model_path, project_root),
        motion_path=_relative_to_root(motion_path, project_root) if motion_path else "",
        visible=bool(slot.visible),
        translation=translation,                         # type: ignore[arg-type]
        rotation=rotation,                               # type: ignore[arg-type]
        external_parents=tuple(
            ProjectExternalParent(
                self_bone_name=binding.self_bone_name,
                target_slot_name=binding.target_slot_name,
                target_bone_name=binding.target_bone_name,
            )
            for binding in slot.external_parents
        ),
    )


def _relative_to_root(path: Path, project_root: Path) -> str:
    """Express ``path`` as a forward-slash string relative to ``project_root``.

    Raises :class:`ValueError` when ``path`` lies outside the root —
    a project file that points at ``../etc/passwd`` is exactly what
    :func:`resolve_safe` rejects on load, so we refuse to write one.
    """
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(project_root)
    except ValueError as err:
        raise ValueError(
            f"asset path {path!r} is not under project root {project_root!r}",
        ) from err
    return relative.as_posix()


def _slot_from_project(
    project_slot: ProjectSlot,
    project_root: Path,
    pmx_loader,                                              # noqa: ANN001
    vmd_loader,                                              # noqa: ANN001
) -> ModelSlot:
    """Re-load a single slot off disk through the resolved importer hooks."""
    model_path = resolve_safe(project_root, project_slot.model_path)
    imported: ImportedScene = pmx_loader(model_path)
    motion = None
    if project_slot.motion_path:
        motion_path = resolve_safe(project_root, project_slot.motion_path)
        motion = vmd_loader(motion_path)
    bindings = tuple(
        ExternalParentBinding(
            self_bone_name=ep.self_bone_name,
            target_slot_name=ep.target_slot_name,
            target_bone_name=ep.target_bone_name,
        )
        for ep in project_slot.external_parents
    )
    slot = ModelSlot(
        name=project_slot.name,
        imported=imported,
        motion=motion,
        visible=project_slot.visible,
        external_parents=bindings,
    )
    slot.transform.set_translation(_vec3(project_slot.translation))
    slot.transform.set_rotation(_vec4(project_slot.rotation))
    return slot


def _vec3(value):                                              # noqa: ANN001, ANN201
    import numpy as np  # noqa: PLC0415
    return np.asarray(value, dtype=np.float32)


def _vec4(value):                                              # noqa: ANN001, ANN201
    import numpy as np  # noqa: PLC0415
    return np.asarray(value, dtype=np.float32)
