"""Tests for the project schema, JSON round-trip, and sync helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pmx.importer import PmxImporter
from vmd.importer import VmdImporter

from posecascade.errors import MalformedAssetError
from posecascade.project import (
    CURRENT_SCHEMA_VERSION,
    ProjectAudio,
    ProjectExternalParent,
    ProjectFile,
    ProjectPlayback,
    ProjectSlot,
    ProjectVersionError,
    load_project,
    parse_project,
    save_project,
    serialize_project,
)
from posecascade.project.sync import (
    LoadedProjectState,
    load_state_from_project,
    project_from_state,
)
from posecascade.render.effects.builtins import register_builtins
from posecascade.render.effects.chain import EffectChain, EffectLibrary
from posecascade.scene.external_parent import ExternalParentBinding
from posecascade.scene.model_slot import ModelSlot, SceneSlots
from posecascade.utils.math3d import vec3

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mmd"


def _sample_project() -> ProjectFile:
    return ProjectFile(
        name="demo",
        slots=(
            ProjectSlot(
                name="character",
                model_path="models/character.pmx",
                motion_path="motions/dance.vmd",
                translation=(1.0, 0.5, -2.0),
                rotation=(0.0, 0.0, 0.7071068, 0.7071068),
                external_parents=(
                    ProjectExternalParent(
                        self_bone_name="hand",
                        target_slot_name="microphone",
                        target_bone_name="anchor",
                    ),
                ),
            ),
            ProjectSlot(
                name="microphone",
                model_path="models/mic.pmx",
                visible=False,
            ),
        ),
        audio=ProjectAudio(path="audio/track.wav", offset_seconds=0.05),
        playback=ProjectPlayback(
            fps=30, start_frame=0, end_frame=900, loop=True, current_frame=120,
        ),
    )


# ----- round-trip ----------------------------------------------------
def test_serialize_and_parse_round_trip() -> None:
    project = _sample_project()
    text = serialize_project(project)
    again = parse_project(text)
    assert again == project


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    project = _sample_project()
    path = tmp_path / "demo.posecascade"
    save_project(project, path)
    again = load_project(path)
    assert again == project


def test_serialize_emits_human_readable_json() -> None:
    project = _sample_project()
    text = serialize_project(project)
    payload = json.loads(text)
    assert payload["version"] == CURRENT_SCHEMA_VERSION
    assert isinstance(payload["slots"], list)


def test_parse_accepts_dict_input() -> None:
    project = _sample_project()
    payload = json.loads(serialize_project(project))
    again = parse_project(payload)
    assert again == project


# ----- schema validation ---------------------------------------------
def test_parse_missing_version_raises() -> None:
    with pytest.raises(MalformedAssetError, match="missing integer"):
        parse_project('{"name": "no version"}')


def test_parse_unknown_version_raises() -> None:
    payload = {"version": CURRENT_SCHEMA_VERSION + 5, "name": "future"}
    with pytest.raises(ProjectVersionError, match="newer than the engine"):
        parse_project(payload)


def test_parse_older_version_without_migration_raises() -> None:
    payload = {"version": 0, "name": "ancient"}
    with pytest.raises(ProjectVersionError, match="no migration registered"):
        parse_project(payload)


def test_parse_invalid_json_raises() -> None:
    with pytest.raises(MalformedAssetError, match="invalid project JSON"):
        parse_project("not_json")


def test_parse_non_object_top_level_raises() -> None:
    with pytest.raises(MalformedAssetError, match="must be an object"):
        parse_project("[1, 2, 3]")


def test_parse_slot_missing_required_field_raises() -> None:
    payload = {
        "version": CURRENT_SCHEMA_VERSION,
        "slots": [{"name": "incomplete"}],   # missing model_path
    }
    with pytest.raises(MalformedAssetError, match="model_path"):
        parse_project(payload)


def test_parse_slot_invalid_translation_shape_raises() -> None:
    payload = {
        "version": CURRENT_SCHEMA_VERSION,
        "slots": [{
            "name": "x", "model_path": "x.pmx",
            "translation": [1.0, 2.0],    # only two components
        }],
    }
    with pytest.raises(MalformedAssetError, match="vec3"):
        parse_project(payload)


def test_parse_slot_invalid_rotation_shape_raises() -> None:
    payload = {
        "version": CURRENT_SCHEMA_VERSION,
        "slots": [{
            "name": "x", "model_path": "x.pmx",
            "rotation": [1.0, 2.0, 3.0],    # missing w
        }],
    }
    with pytest.raises(MalformedAssetError, match="vec4"):
        parse_project(payload)


# ----- sync: project_from_state --------------------------------------
def _scene_with_one_slot(tmp_path: Path) -> tuple[SceneSlots, dict[str, tuple[Path, Path | None]]]:
    """Build a SceneSlots backed by tmp-path-located fixture files."""
    models_dir = tmp_path / "models"
    motions_dir = tmp_path / "motions"
    models_dir.mkdir()
    motions_dir.mkdir()
    pmx_target = models_dir / "tiny.pmx"
    pmx_target.write_bytes((_FIXTURES / "tiny.pmx").read_bytes())
    vmd_target = motions_dir / "wave.vmd"
    vmd_target.write_bytes((_FIXTURES / "wave.vmd").read_bytes())
    slot = ModelSlot(
        name="character",
        imported=PmxImporter().load(pmx_target),
        motion=VmdImporter().load(vmd_target),
    )
    slot.transform.set_translation(vec3(2.0, 0.0, 0.0))
    slots = SceneSlots()
    slots.add(slot)
    paths = {"character": (pmx_target, vmd_target)}
    return slots, paths


def test_project_from_state_writes_relative_paths(tmp_path: Path) -> None:
    slots, source_paths = _scene_with_one_slot(tmp_path)
    project = project_from_state(
        slots=slots,
        slot_source_paths=source_paths,
        playback=ProjectPlayback(),
        audio=None,
        effect_chain=EffectChain(),
        project_root=tmp_path,
    )
    assert project.slots[0].model_path == "models/tiny.pmx"
    assert project.slots[0].motion_path == "motions/wave.vmd"


def test_project_from_state_preserves_translation(tmp_path: Path) -> None:
    slots, source_paths = _scene_with_one_slot(tmp_path)
    project = project_from_state(
        slots=slots, slot_source_paths=source_paths,
        playback=ProjectPlayback(), audio=None, effect_chain=EffectChain(),
        project_root=tmp_path,
    )
    np.testing.assert_allclose(project.slots[0].translation, [2.0, 0.0, 0.0])


def test_project_from_state_rejects_path_outside_root(tmp_path: Path) -> None:
    slots, _ = _scene_with_one_slot(tmp_path)
    other_root = tmp_path.parent / "other_project"
    other_root.mkdir(exist_ok=True)
    bad_paths = {"character": (other_root / "ghost.pmx", None)}
    with pytest.raises(ValueError, match="not under project root"):
        project_from_state(
            slots=slots,
            slot_source_paths=bad_paths,
            playback=ProjectPlayback(), audio=None, effect_chain=EffectChain(),
            project_root=tmp_path,
        )


def test_project_from_state_rejects_slot_without_paths(tmp_path: Path) -> None:
    slot = ModelSlot(name="orphan", imported=PmxImporter().load(_FIXTURES / "tiny.pmx"))
    slots = SceneSlots()
    slots.add(slot)
    with pytest.raises(ValueError, match="no source model path"):
        project_from_state(
            slots=slots,
            slot_source_paths={},                # nothing supplied
            playback=ProjectPlayback(), audio=None, effect_chain=EffectChain(),
            project_root=tmp_path,
        )


# ----- sync: load_state_from_project ---------------------------------
def test_load_state_from_project_reloads_slots(tmp_path: Path) -> None:
    slots, source_paths = _scene_with_one_slot(tmp_path)
    project = project_from_state(
        slots=slots, slot_source_paths=source_paths,
        playback=ProjectPlayback(end_frame=120, current_frame=42),
        audio=ProjectAudio(path="audio/song.wav"),
        effect_chain=EffectChain(),
        project_root=tmp_path,
    )
    state = load_state_from_project(
        project,
        project_root=tmp_path,
        pmx_loader=PmxImporter().load,
        vmd_loader=VmdImporter().load,
        effect_library=EffectLibrary(),
    )
    assert isinstance(state, LoadedProjectState)
    assert len(state.slots) == 1
    reloaded = state.slots.find("character")
    assert reloaded is not None
    np.testing.assert_allclose(reloaded.transform.translation, [2.0, 0.0, 0.0])
    assert state.audio is not None
    assert state.audio.path == "audio/song.wav"
    assert state.playback.current_frame == 42


def test_load_state_rejects_path_traversal_in_slot(tmp_path: Path) -> None:
    """A project that points at ``../../etc/passwd`` must fail to load."""
    project = ProjectFile(
        slots=(ProjectSlot(name="evil", model_path="../../etc/passwd"),),
    )
    from posecascade.errors import UnsafePathError  # noqa: PLC0415 — exception only here
    (tmp_path / "models").mkdir(exist_ok=True)
    with pytest.raises(UnsafePathError):
        load_state_from_project(
            project, project_root=tmp_path,
            pmx_loader=PmxImporter().load, vmd_loader=VmdImporter().load,
            effect_library=EffectLibrary(),
        )


def test_load_state_round_trips_effect_chain(tmp_path: Path) -> None:
    slots, source_paths = _scene_with_one_slot(tmp_path)
    library = register_builtins(EffectLibrary())
    chain = EffectChain()
    chain.append(library.find("autoluminous"))
    chain.set_uniform(0, "intensity", 1.7)
    project = project_from_state(
        slots=slots, slot_source_paths=source_paths,
        playback=ProjectPlayback(), audio=None, effect_chain=chain,
        project_root=tmp_path,
    )
    state = load_state_from_project(
        project, project_root=tmp_path,
        pmx_loader=PmxImporter().load, vmd_loader=VmdImporter().load,
        effect_library=library,
    )
    assert len(state.effect_chain) == 1
    assert state.effect_chain.entries[0].effective_value("intensity") == pytest.approx(1.7)


def test_load_project_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MalformedAssetError, match="not found"):
        load_project(tmp_path / "nope.posecascade")


# ----- external parents ---------------------------------------------
def test_external_parent_round_trip_through_project(tmp_path: Path) -> None:
    """A binding survives state → project → state without losing field values."""
    pmx_target = tmp_path / "tiny.pmx"
    pmx_target.write_bytes((_FIXTURES / "tiny.pmx").read_bytes())
    binding = ExternalParentBinding(
        self_bone_name="hand", target_slot_name="ally", target_bone_name="anchor",
    )
    slot = ModelSlot(
        name="character",
        imported=PmxImporter().load(pmx_target),
        external_parents=(binding,),
    )
    slots = SceneSlots()
    slots.add(slot)
    project = project_from_state(
        slots=slots,
        slot_source_paths={"character": (pmx_target, None)},
        playback=ProjectPlayback(), audio=None, effect_chain=EffectChain(),
        project_root=tmp_path,
    )
    text = serialize_project(project)
    again = parse_project(text)
    assert again.slots[0].external_parents == (
        ProjectExternalParent(
            self_bone_name="hand",
            target_slot_name="ally",
            target_bone_name="anchor",
        ),
    )


# Keep the unused symbols load-bearing for IDE jumps.
__all__ = ["pytest"]
