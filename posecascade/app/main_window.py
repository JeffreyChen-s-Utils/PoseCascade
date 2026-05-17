"""Top-level MainWindow — viewport + outliner + inspector + menus + status bar.

Layout (Blender / Godot style):

* **Central widget**: the live :class:`Viewport` (3D scene render).
* **Left dock**: :class:`OutlinerDock` — scene tree, click to select.
* **Right dock**: :class:`InspectorDock` — edit selected node's transform and
  inline-tune any spring chain or cloth piece attached to it.
* **Menu bar**: File (Open Scene / Open Script / Quit), View (Reset Camera).
* **Toolbar**: ⏵ Play / ⏸ Pause toggle, ↻ Reset physics.
* **Status bar**: scene path · script · FPS · selected node.

The window owns the per-frame ``QTimer`` that pumps the physics, cloth, and
script hosts. The pause flag short-circuits those ticks but still calls
``viewport.update()`` so the user can keep orbiting the camera.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QToolBar,
)

from posecascade.animation.document import AnimationDocument
from posecascade.app.controller import AppController
from posecascade.app.registry import Services
from posecascade.i18n import (
    available_languages,
    current_language,
    language_display_name,
    set_language,
    t,
)
from posecascade.render.effects.chain import EffectChain
from posecascade.scene.model_slot import SceneSlots
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.ui.animation_command_stack import AnimationCommandStack
from posecascade.ui.animation_json_dock import AnimationJsonDock
from posecascade.ui.animation_json_document import AnimationJsonDocument
from posecascade.ui.effect_chain_dock import EffectChainDock
from posecascade.ui.export_dialog import ExportDialog, ExportSpec
from posecascade.ui.inspector import InspectorDock
from posecascade.ui.multi_track_timeline import MultiTrackTimelineDock
from posecascade.ui.outliner import OutlinerDock
from posecascade.ui.phase_blocks_dock import PhaseBlocksDock
from posecascade.ui.slot_list_dock import SlotListDock
from posecascade.ui.timeline_basic import TimelineDock
from posecascade.ui.viewport import Viewport
from posecascade.utils.clock import FrameClock
from posecascade.utils.logging import get_logger

_log = get_logger(__name__)
_TARGET_FRAME_INTERVAL_MS = 16  # ~60Hz; the timer is best-effort, the clamp does the rest
_FPS_SMOOTHING_WINDOW = 30      # average FPS over this many frames so the readout doesn't flicker

# Font-metric multipliers for widget sizing — replace the old pixel literals so
# the layout scales with the user's font size / DPI. ``QFontMetrics.height()``
# is roughly one text row, so ``height * N`` reads as "N rows tall".
_VIEWPORT_MIN_ROWS_W = 30   # ≈ 640 px at default 21 px row height
_VIEWPORT_MIN_ROWS_H = 22   # ≈ 480 px at default 21 px row height
_FPS_LABEL_MIN_WIDTH_CHARS = 8


def make_default_scene() -> Scene:
    """Return a fresh empty :class:`Scene` — just a root Node, no children.

    Used as the editor's startup scene so the viewport, outliner, and inspector
    are all valid out of the gate (no ``None`` checks scattered everywhere).
    Loading a file replaces this scene wholesale.
    """
    return Scene(root=Node(name="Scene"))


class MainWindow(QMainWindow):
    """Editor shell: viewport + outliner + inspector + menus + status bar."""

    def __init__(self, services: Services) -> None:
        super().__init__()
        self.setWindowTitle(t("app.window_title"))
        self._services = services
        self._paused = False
        self._dt_history: deque[float] = deque(maxlen=_FPS_SMOOTHING_WINDOW)
        self._scene_path: Path | None = None
        self._script_path: Path | None = None

        self._viewport = Viewport(self)
        self.setCentralWidget(self._viewport)
        row_h = self.fontMetrics().height()
        self._viewport.setMinimumSize(
            row_h * _VIEWPORT_MIN_ROWS_W, row_h * _VIEWPORT_MIN_ROWS_H,
        )

        self._outliner = OutlinerDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._outliner)
        self._slot_list = SlotListDock(SceneSlots(), self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._slot_list)
        self._inspector = InspectorDock(services=services, parent=self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._inspector)
        self._effect_dock = EffectChainDock(EffectChain(), self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._effect_dock)
        self.tabifyDockWidget(self._inspector, self._effect_dock)
        self._build_animation_docks()
        self._timeline = TimelineDock(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._timeline)
        self._multi_track = MultiTrackTimelineDock(
            document=AnimationDocument(), scene=None, parent=self,
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._multi_track)
        self.tabifyDockWidget(self._timeline, self._multi_track)

        # AppController owns the cross-layer state. We pass simple
        # importer hooks; bootstrap can swap them out for the live
        # ImporterManager if richer dispatch is needed.
        self._controller = AppController(
            project_root=Path.cwd(),
            pmx_loader=self._fallback_loader,
            vmd_loader=self._fallback_loader,
            on_slots_changed=self._on_controller_slots_changed,
            on_chain_changed=self._on_controller_chain_changed,
        )
        self._timeline.current_frame_changed.connect(self._on_timeline_frame_changed)
        self._multi_track.document_changed.connect(self._viewport.update)
        self._effect_dock.chain_changed.connect(
            lambda: self._controller.set_effect_chain(self._effect_dock.chain),
        )

        self._build_menu_bar()
        self._build_toolbar()
        self._build_status_bar()

        # Install an empty scene so the viewport draws (clears to background) and
        # the outliner / inspector are usable before the user loads anything.
        self._viewport.set_scene(make_default_scene())
        self._outliner.set_scene(self._viewport.scene)
        self._update_status_text()

        self._outliner.node_selected.connect(self._on_outliner_selection)
        self._outliner.node_deleted.connect(self._on_outliner_delete)

        self._clock = FrameClock()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(_TARGET_FRAME_INTERVAL_MS)

    # --- public API used by bootstrap ----------------------------------------

    @property
    def viewport(self) -> Viewport:
        return self._viewport

    @property
    def services(self) -> Services:
        return self._services

    @property
    def clock(self) -> FrameClock:
        return self._clock

    def set_scene_path(self, path: Path | None) -> None:
        """Bootstrap calls this after a successful scene import; refreshes UI labels."""
        self._scene_path = path
        if self._viewport.scene is not None:
            self._outliner.set_scene(self._viewport.scene)
        self._update_status_text()

    def set_script_path(self, path: Path | None) -> None:
        self._script_path = path
        # Mirror the just-loaded script into the JSON dock when it's a
        # declarative animation. Python scripts go through the sandbox
        # loader and don't have an in-editor view yet, so we leave the
        # dock alone for them — its placeholder text already tells the
        # user to use ``File → Open Script`` with a ``.json``.
        if path is not None and path.suffix.lower() == ".json":
            self._animation_json_dock.load_file(path)
        self._update_status_text()

    def _on_animation_json_reload(self, text: str) -> None:
        """Re-attach the dock's current text as a declarative animation.

        Writes the buffer to a temp file is unnecessary — ``attach_script``
        only needs the path for ``extends`` resolution + the script-host
        registry name. We mirror the in-memory text to the on-disk file
        through a transient save (the dock's Save button does this on
        demand) and then re-run the attach flow.
        """
        if self._script_path is None or self._script_path.suffix.lower() != ".json":
            _log.warning("animation reload requested but no .json path bound")
            return
        try:
            self._script_path.write_text(text, encoding="utf-8")
        except OSError as err:
            _log.error("failed to persist edited animation: %s", err)
            return
        from posecascade.app.bootstrap import attach_script  # noqa: PLC0415

        attach_script(self, self._services, self._script_path)

    # --- Qt overrides --------------------------------------------------------

    def closeEvent(self, event: object) -> None:  # noqa: N802 — Qt override
        self._timer.stop()
        super().closeEvent(event)

    # --- dock construction helpers ------------------------------------------

    def _build_animation_docks(self) -> None:
        """Build the JSON + phase-blocks docks sharing one document.

        Pulled out of ``__init__`` to keep that method under the
        cyclomatic-complexity bound — the editor's right-column dock
        wiring adds enough statements that the constructor crosses the
        50-statement linter cap when this is inline.
        """
        self._animation_document = AnimationJsonDocument(self)
        self._animation_stack = AnimationCommandStack(
            self._animation_document, self,
        )
        self._animation_json_dock = AnimationJsonDock(
            document=self._animation_document,
            command_stack=self._animation_stack,
            parent=self,
        )
        self._phase_blocks_dock = PhaseBlocksDock(
            document=self._animation_document,
            command_stack=self._animation_stack,
            parent=self,
        )
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self._animation_json_dock,
        )
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self._phase_blocks_dock,
        )
        self.tabifyDockWidget(self._inspector, self._animation_json_dock)
        self.tabifyDockWidget(self._animation_json_dock, self._phase_blocks_dock)
        self._animation_json_dock.reload_requested.connect(
            self._on_animation_json_reload,
        )
        self._phase_blocks_dock.reload_requested.connect(
            self._on_animation_json_reload,
        )

    # --- menu / toolbar / status bar -----------------------------------------

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu(t("menu.file"))

        open_scene = QAction(t("menu.file.open_scene"), self)
        open_scene.setShortcut(QKeySequence.StandardKey.Open)
        open_scene.triggered.connect(self._on_open_scene)
        file_menu.addAction(open_scene)

        open_script = QAction(t("menu.file.open_script"), self)
        open_script.triggered.connect(self._on_open_script)
        file_menu.addAction(open_script)

        file_menu.addSeparator()
        open_project = QAction(t("menu.file.open_project"), self)
        open_project.triggered.connect(self._on_open_project)
        file_menu.addAction(open_project)
        save_project = QAction(t("menu.file.save_project_as"), self)
        save_project.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_project.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_project)
        file_menu.addSeparator()
        export_action = QAction(t("menu.file.export"), self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        quit_action = QAction(t("menu.file.quit"), self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = menu_bar.addMenu(t("menu.view"))
        reset_camera = QAction(t("menu.view.reset_camera"), self)
        reset_camera.triggered.connect(self._on_reset_camera)
        view_menu.addAction(reset_camera)
        view_menu.addSeparator()
        view_menu.addAction(self._outliner.toggleViewAction())
        view_menu.addAction(self._slot_list.toggleViewAction())
        view_menu.addAction(self._inspector.toggleViewAction())
        view_menu.addAction(self._effect_dock.toggleViewAction())
        view_menu.addAction(self._animation_json_dock.toggleViewAction())
        view_menu.addAction(self._phase_blocks_dock.toggleViewAction())
        view_menu.addAction(self._timeline.toggleViewAction())
        view_menu.addAction(self._multi_track.toggleViewAction())

        self._build_settings_menu(menu_bar)

        help_menu = menu_bar.addMenu(t("menu.help"))
        about_action = QAction(t("menu.help.about"), self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _build_settings_menu(self, menu_bar: object) -> None:
        """Add ``Settings → Language`` with one entry per available locale.

        The currently active locale is marked with a checkable / checked
        action so the user can see at a glance which language is in use.
        Switching writes the choice to ``QSettings`` and shows a
        restart-to-apply message — a full in-place retranslate would need
        ``changeEvent(QEvent.LanguageChange)`` plumbing on every widget,
        which is a future PR.
        """
        settings_menu = menu_bar.addMenu(t("menu.settings"))  # type: ignore[attr-defined]
        language_menu = settings_menu.addMenu(t("menu.settings.language"))
        active = current_language()
        for code in available_languages():
            action = QAction(language_display_name(code), self)
            action.setCheckable(True)
            action.setChecked(code == active)
            action.triggered.connect(
                # Bind ``code`` at definition time — without ``code=code``
                # every lambda would close over the loop's final value.
                lambda _checked=False, code=code: self._on_language_chosen(code),
            )
            language_menu.addAction(action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar(t("toolbar.playback"), self)
        toolbar.setObjectName("PlaybackToolbar")
        self.addToolBar(toolbar)

        self._play_pause = QAction(t("toolbar.pause"), self)
        self._play_pause.setCheckable(False)
        self._play_pause.triggered.connect(self._on_toggle_pause)
        toolbar.addAction(self._play_pause)

        reset_physics = QAction(t("toolbar.reset_physics"), self)
        reset_physics.triggered.connect(self._on_reset_physics)
        toolbar.addAction(reset_physics)

    def _build_status_bar(self) -> None:
        self._status_text = QLabel(t("status.ready"))
        self._fps_label = QLabel(t("status.fps_placeholder"))
        # Pad the FPS label to a stable width so the readout doesn't
        # twitch as the digit count changes — derived from the font's
        # "0" advance so it scales with DPI / user font size.
        digit_w = self.fontMetrics().horizontalAdvance("0")
        self._fps_label.setMinimumWidth(digit_w * _FPS_LABEL_MIN_WIDTH_CHARS)
        self.statusBar().addWidget(self._status_text, 1)
        self.statusBar().addPermanentWidget(self._fps_label)

    # --- per-frame tick ------------------------------------------------------

    def _on_tick(self) -> None:
        dt = self._clock.tick()
        if not self._paused:
            self._tick_simulations(dt)
        self._update_fps(dt)
        self._viewport.update()

    def _tick_simulations(self, dt: float) -> None:
        try:
            self._services.physics_host.tick(dt)
        except Exception as err:  # noqa: BLE001 — boundary; log and continue
            _log.error("physics tick raised: %s", err)
        try:
            self._services.cloth_host.tick(dt)
        except Exception as err:  # noqa: BLE001 — boundary
            _log.error("cloth tick raised: %s", err)
        try:
            self._services.script_host.tick(dt)
        except Exception as err:  # noqa: BLE001 — boundary; offending script already disabled
            _log.error("script tick raised: %s", err)
        # Foot planting runs LAST so it sees the final per-frame pose
        # the script + IK chose, and lifts any foot that's clipped
        # below its registered ground surface. Without this hook a
        # script that doesn't manually pin feet to stair surfaces
        # ends up with the foot sliding into the stair geometry.
        try:
            self._services.foot_planter.apply()
        except Exception as err:  # noqa: BLE001 — boundary; one bad chain shouldn't freeze the timeline
            _log.error("foot planter raised: %s", err)

    def _update_fps(self, dt: float) -> None:
        if dt > 0.0:
            self._dt_history.append(dt)
        if not self._dt_history:
            return
        avg_dt = sum(self._dt_history) / len(self._dt_history)
        if avg_dt > 0.0:
            self._fps_label.setText(t("status.fps", value=1.0 / avg_dt))

    # --- menu / toolbar handlers ---------------------------------------------

    def _on_open_scene(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, t("dialog.open_scene.title"), "", t("dialog.scene_filter"),
        )
        if not path_str:
            return
        # Lazy import to avoid a top-level cycle (bootstrap ↔ main_window).
        from posecascade.app.bootstrap import attach_scene  # noqa: PLC0415

        attach_scene(self, self._services, Path(path_str))

    def _on_open_script(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, t("dialog.open_script.title"), "", t("dialog.script_filter"),
        )
        if not path_str:
            return
        from posecascade.app.bootstrap import attach_script  # noqa: PLC0415

        attach_script(self, self._services, Path(path_str))

    def _on_reset_camera(self) -> None:
        self._viewport.camera.reset() if hasattr(self._viewport.camera, "reset") else None
        self._viewport.update()

    # ----- project / export -------------------------------------------
    def _on_open_project(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, t("dialog.open_project.title"), "", t("dialog.project_filter"),
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            self._controller.project_root = path.parent
            self._controller.open_project(path)
        except Exception as err:                  # noqa: BLE001 — boundary
            _log.error("open project failed: %s", err)
            QMessageBox.warning(
                self,
                t("dialog.open_project.error_title"),
                t("dialog.open_project.error_body", error=err),
            )

    def _on_save_project_as(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self, t("dialog.save_project.title"), "", t("dialog.project_filter"),
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            self._controller.project_root = path.parent
            self._controller.save_project_to(path, name=path.stem)
        except Exception as err:                  # noqa: BLE001 — boundary
            _log.error("save project failed: %s", err)
            QMessageBox.warning(
                self,
                t("dialog.save_project.error_title"),
                t("dialog.save_project.error_body", error=err),
            )

    def _on_export(self) -> None:
        dialog = ExportDialog(self)
        dialog.export_requested.connect(self._on_export_requested)
        dialog.exec()

    def _on_export_requested(self, spec: ExportSpec) -> None:
        try:
            self._controller.run_export(
                spec,
                document=self._multi_track.document,
                render_frame_fn=self._render_frame_for_export,
            )
        except Exception as err:                  # noqa: BLE001 — boundary
            _log.error("export failed: %s", err)
            QMessageBox.warning(
                self,
                t("dialog.export.error_title"),
                t("dialog.export.error_body", error=err),
            )

    def _render_frame_for_export(self, frame: int) -> object:
        """Hook for offline rendering — Phase 15 wires this through the
        viewport's offscreen FBO. The shipping integrator typically
        replaces this with a headless renderer; the default returns a
        plain transparent buffer so the export pipeline still produces
        a file even before the offscreen path lands."""
        from numpy import uint8, zeros  # noqa: PLC0415 — late import
        del frame
        return zeros((1, 1, 4), dtype=uint8)

    # ----- controller callbacks ---------------------------------------
    def _on_timeline_frame_changed(self, frame: int) -> None:
        self._controller.on_frame_changed(int(frame))
        self._multi_track.set_current_frame(int(frame))
        self._viewport.update()

    def _on_controller_slots_changed(self, slots: SceneSlots) -> None:
        self._slot_list._slots = slots                 # noqa: SLF001 — cross-layer write
        self._slot_list.refresh()

    def _on_controller_chain_changed(self, chain: EffectChain) -> None:
        # The dock owns the live chain object; just trigger a refresh
        # so the entry list re-paints with the new state.
        self._effect_dock._chain = chain               # noqa: SLF001
        self._effect_dock.refresh()

    @staticmethod
    def _fallback_loader(path: Path) -> object:
        """Default importer hook for the controller — re-raises with a
        clearer message when nothing better is wired in. The bootstrap
        layer typically replaces this with a real
        :class:`ImporterManager` dispatch.
        """
        raise RuntimeError(
            f"no importer registered for project asset {path!r}; "
            f"bootstrap must wire pmx_loader / vmd_loader on the controller",
        )

    def _on_about(self) -> None:
        QMessageBox.about(self, t("dialog.about.title"), t("dialog.about.body"))

    def _on_language_chosen(self, code: str) -> None:
        """Persist the language choice and tell the user to restart.

        Errors bubble up from :func:`set_language` only when the menu
        was built from a directory listing that has since changed under
        us (locale JSON deleted mid-session). In that case log + warn —
        leaving the previous language active is the least surprising
        recovery.
        """
        try:
            set_language(code)
        except Exception as err:                  # noqa: BLE001 — boundary
            _log.error("failed to switch language to %s: %s", code, err)
            QMessageBox.warning(
                self,
                t("dialog.language.restart_title"),
                str(err),
            )
            return
        QMessageBox.information(
            self,
            t("dialog.language.restart_title"),
            t("dialog.language.restart_body", language=language_display_name(code)),
        )

    def _on_toggle_pause(self) -> None:
        self._paused = not self._paused
        self._play_pause.setText(
            t("toolbar.play") if self._paused else t("toolbar.pause"),
        )

    def _on_reset_physics(self) -> None:
        for chain in self._services.physics_host.chains():
            chain.reset()
        for binding in self._services.cloth_host.bindings():
            binding.piece.reset()

    def _on_outliner_selection(self, node: Node | None) -> None:
        self._inspector.set_node(node)
        # Mirror the outliner selection into the viewport's selection
        # overlay. Passing ``None`` clears the highlight, which is what
        # the outliner's right-click handler relies on (it clears the
        # tree's selection before showing the context menu so the
        # highlight disappears immediately).
        self._viewport.set_selected_holder(node)
        self._update_status_text(selected_name=(node.name if node is not None else None))

    def _on_outliner_delete(self, node: Node) -> None:
        """Cleanup hosts when a subtree is deleted: drop chains + cloth pieces it owned.

        The outliner has already detached ``node`` from its parent before this
        fires. We sweep the physics + cloth hosts so they stop simulating /
        uploading positions for now-orphaned nodes.
        """
        chains_dropped = self._services.physics_host.remove_chains_for_subtree(node)
        cloth_dropped = self._services.cloth_host.remove_pieces_for_subtree(node)
        if chains_dropped or cloth_dropped:
            _log.info(
                "deleted %r: removed %d spring chain(s), %d cloth piece(s)",
                node.name, chains_dropped, cloth_dropped,
            )
        # If the deleted node was selected, clear the inspector.
        if self._inspector._node is node:  # noqa: SLF001
            self._inspector.set_node(None)
            self._update_status_text()

    def _update_status_text(self, selected_name: str | None = None) -> None:
        parts = []
        if self._scene_path is not None:
            parts.append(t("status.scene_prefix", name=self._scene_path.name))
        else:
            parts.append(
                t("status.scene_prefix", name=t("app.scene_label_untitled")),
            )
        if self._script_path is not None:
            parts.append(t("status.script_prefix", name=self._script_path.name))
        if selected_name:
            parts.append(t("status.selected_prefix", name=selected_name))
        self._status_text.setText(" · ".join(parts))
