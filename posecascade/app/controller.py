"""High-level orchestration glue between Phase-1–15 layers.

The :class:`AppController` is the place that knows about *all* the
moving parts at once:

- the live :class:`SceneSlots` (Phase 12) backed by a per-slot
  :class:`SlotsPlayer` (animation),
- the global :class:`EffectChain` (Phase 13) and its registry,
- the :class:`AudioPlayer` (Phase 10),
- a per-slot :class:`VmdSceneDriver` for camera / light state when the
  active slot's motion ships those tracks,
- save / load through :mod:`posecascade.project`.

The MainWindow wires its docks + menu actions into this controller's
slots; the controller in turn updates the renderer + viewport. Keeping
all the cross-layer wiring here lets unit tests stand the controller
up against mock viewports and headless renderer doubles.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from posecascade.animation.scene_driver import VmdSceneDriver
from posecascade.animation.slots_player import SlotsPlayer
from posecascade.animation.vmd_track import VMD_FRAMES_PER_SECOND
from posecascade.audio.player import AudioPlayer
from posecascade.export.image_sequence import export_image_sequence
from posecascade.export.video import export_video_from_image_sequence
from posecascade.export.vmd import export_animation_to_vmd
from posecascade.project.reader import load_project
from posecascade.project.schema import ProjectFile
from posecascade.project.sync import (
    LoadedProjectState,
    load_state_from_project,
    project_from_state,
)
from posecascade.project.writer import save_project
from posecascade.render.effects.builtins import load_builtin, register_builtins
from posecascade.render.effects.chain import EffectChain, EffectLibrary
from posecascade.scene.model_slot import ModelSlot, SceneSlots
from posecascade.ui.export_dialog import ExportSpec, ExportTarget


@dataclass
class AppController:
    """Coordinates animation + rendering + project I/O for the editor.

    Construct with the integrator-supplied loader / renderer hooks; the
    controller never touches Qt directly so it stays testable in
    headless contexts.
    """

    project_root: Path
    pmx_loader: Callable                                    # noqa: ANN001 — late-bound callable
    vmd_loader: Callable                                    # noqa: ANN001
    on_slots_changed: Callable[[SceneSlots], None] | None = None
    on_camera_changed: Callable[[object], None] | None = None
    on_light_changed: Callable[[object], None] | None = None
    on_chain_changed: Callable[[EffectChain], None] | None = None
    slots: SceneSlots = field(default_factory=SceneSlots)
    effect_library: EffectLibrary = field(default_factory=EffectLibrary)
    effect_chain: EffectChain = field(default_factory=EffectChain)
    audio: AudioPlayer = field(default_factory=AudioPlayer)
    _slots_player: SlotsPlayer | None = field(default=None, init=False)
    _scene_drivers: dict[str, VmdSceneDriver] = field(default_factory=dict, init=False)
    _slot_source_paths: dict[str, tuple[Path, Path | None]] = field(
        default_factory=dict, init=False,
    )

    def __post_init__(self) -> None:
        # Pre-register the four built-in effect descriptors so the chain
        # UI's "Add" menu has something to offer out of the gate.
        register_builtins(self.effect_library)
        self._seed_default_effect_chain()

    def _seed_default_effect_chain(self) -> None:
        """Append AutoLuminous to a freshly-constructed empty chain.

        MMD ships its bloom equivalent on by default — the "glow" you
        see at every dance opening is AutoLuminous-style emission. We
        match that out of the box: an empty chain becomes a one-entry
        chain with AutoLuminous at descriptor defaults. Projects that
        explicitly load a saved chain hit ``_apply_loaded_state`` which
        replaces ``effect_chain`` wholesale, so user-saved configs are
        never clobbered.
        """
        if len(self.effect_chain) > 0:
            return
        try:
            descriptor = load_builtin("autoluminous")
        except Exception:                                       # noqa: BLE001
            # Effect file missing / malformed — fall back to no default
            # rather than crashing the editor on launch.
            return
        self.effect_chain.append(descriptor)

    # ----- frame ticking ----------------------------------------------
    def on_frame_changed(self, frame: int) -> None:
        """Drive every layer with the new playhead frame.

        Animation player advances; the scene driver hands us the
        camera + light at this frame; the integrator's hooks fire so
        the viewport / renderer can refresh.
        """
        time_seconds = float(frame) / float(VMD_FRAMES_PER_SECOND)
        if self._slots_player is not None:
            self._slots_player.apply(time_seconds)
        camera = self._sample_first_camera(time_seconds)
        if camera is not None and self.on_camera_changed is not None:
            self.on_camera_changed(camera)
        light = self._sample_first_light(time_seconds)
        if light is not None and self.on_light_changed is not None:
            self.on_light_changed(light)

    # ----- project I/O ------------------------------------------------
    def open_project(self, path: Path) -> ProjectFile:
        """Load a ``.posecascade`` file + replace the live state.

        After this returns, ``self.slots`` reflects the project's slot
        list, the effect chain is rebuilt from the project's TOML, and
        the integrator hooks fire so the UI re-binds.
        """
        path = Path(path)
        project = load_project(path)
        state = load_state_from_project(
            project,
            project_root=self.project_root,
            pmx_loader=self.pmx_loader,
            vmd_loader=self.vmd_loader,
            effect_library=self.effect_library,
        )
        self._apply_loaded_state(project, state)
        return project

    def save_project_to(self, path: Path, *, name: str = "") -> ProjectFile:
        """Snapshot live state into a project file + write it to ``path``."""
        project = project_from_state(
            slots=self.slots,
            slot_source_paths=self._slot_source_paths,
            playback=self._build_playback(),
            audio=None,
            effect_chain=self.effect_chain,
            project_root=self.project_root,
            name=name,
        )
        save_project(project, Path(path))
        return project

    # ----- export ------------------------------------------------------
    def run_export(
        self,
        spec: ExportSpec,
        *,
        render_frame_fn: Callable | None = None,    # noqa: ANN001
        document=None,                              # noqa: ANN001 — late-bound AnimationDocument
    ) -> Path:
        """Dispatch ``spec`` to the matching exporter.

        ``render_frame_fn`` is required for image-sequence and video
        exports (the renderer side wires this up); ``document`` is
        required for VMD exports.
        """
        if spec.target == ExportTarget.VMD:
            if document is None:
                raise ValueError(
                    "VMD export requires an AnimationDocument; "
                    "pass ``document=`` to run_export",
                )
            return export_animation_to_vmd(document, spec.output_path)
        if spec.target == ExportTarget.IMAGE_SEQUENCE:
            if render_frame_fn is None:
                raise ValueError("image-sequence export requires a render_frame_fn")
            export_image_sequence(
                render_frame_fn=render_frame_fn,
                start_frame=spec.start_frame, end_frame=spec.end_frame,
                output_dir=spec.output_path,
                padding=spec.padding,
                overwrite=True,
            )
            return spec.output_path
        if spec.target == ExportTarget.VIDEO:
            if render_frame_fn is None:
                raise ValueError("video export requires a render_frame_fn")
            tmp_dir = spec.output_path.with_suffix("")
            export_image_sequence(
                render_frame_fn=render_frame_fn,
                start_frame=spec.start_frame, end_frame=spec.end_frame,
                output_dir=tmp_dir,
                overwrite=True,
            )
            return export_video_from_image_sequence(
                input_dir=tmp_dir,
                output_path=spec.output_path,
                fps=spec.fps,
                codec=spec.codec,
            )
        raise ValueError(f"unhandled export target: {spec.target}")

    # ----- slot management --------------------------------------------
    def add_slot(
        self, slot: ModelSlot, *,
        model_path: Path,
        motion_path: Path | None = None,
    ) -> ModelSlot:
        """Append ``slot`` + remember the source paths for later save.

        We don't try to back-derive the path from the imported scene —
        the integrator that loaded the file knows it best. Pass the
        absolute path here.
        """
        self.slots.add(slot)
        self._slot_source_paths[slot.name] = (model_path, motion_path)
        self._rebuild_players()
        if self.on_slots_changed is not None:
            self.on_slots_changed(self.slots)
        return slot

    def clear_slots(self) -> None:
        """Drop every slot + tear down per-slot players."""
        self.slots = SceneSlots()
        self._slot_source_paths.clear()
        self._slots_player = None
        self._scene_drivers.clear()
        if self.on_slots_changed is not None:
            self.on_slots_changed(self.slots)

    # ----- effect chain ----------------------------------------------
    def set_effect_chain(self, chain: EffectChain) -> None:
        self.effect_chain = chain
        if self.on_chain_changed is not None:
            self.on_chain_changed(chain)

    # ----- internal ---------------------------------------------------
    def _rebuild_players(self) -> None:
        self._slots_player = SlotsPlayer(slots=self.slots)
        self._scene_drivers = {
            slot.name: VmdSceneDriver(motion=slot.motion)
            for slot in self.slots
            if slot.motion is not None
        }

    def _apply_loaded_state(
        self, project: ProjectFile, state: LoadedProjectState,
    ) -> None:
        self.slots = state.slots
        self._slot_source_paths.clear()
        for slot, project_slot in zip(state.slots, project.slots, strict=False):
            model = self.project_root / project_slot.model_path
            motion = (
                (self.project_root / project_slot.motion_path)
                if project_slot.motion_path else None
            )
            self._slot_source_paths[slot.name] = (model, motion)
        self._rebuild_players()
        self.set_effect_chain(state.effect_chain)
        if self.on_slots_changed is not None:
            self.on_slots_changed(self.slots)

    def _build_playback(self):                   # noqa: ANN201 — late-bound ProjectPlayback
        from posecascade.project.schema import ProjectPlayback  # noqa: PLC0415
        return ProjectPlayback()

    def _sample_first_camera(self, time_seconds: float):     # noqa: ANN201
        for driver in self._scene_drivers.values():
            camera = driver.camera_at(time_seconds)
            if camera is not None:
                return camera
        return None

    def _sample_first_light(self, time_seconds: float):      # noqa: ANN201
        for driver in self._scene_drivers.values():
            light = driver.light_at(time_seconds)
            if light is not None:
                return light
        return None
