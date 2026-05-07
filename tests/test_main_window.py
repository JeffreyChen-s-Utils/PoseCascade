"""Smoke tests for the editor :class:`MainWindow` layout + behaviour.

Skips cleanly if PySide6 isn't available. Does NOT instantiate a real
``QApplication`` event loop — uses the ``qapp`` fixture's existing instance.
The viewport's GL context is not initialised here (no ``initializeGL`` is
called) so renderer-touching paths are exercised by ``test_render_smoke.py``,
not this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from posecascade.app.main_window import MainWindow
from posecascade.app.registry import Services, build_services
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene


@pytest.fixture
def services() -> Services:
    return build_services(project_root=Path(__file__).resolve().parent.parent)


def test_main_window_starts_with_empty_scene(qapp: Any, services: Services) -> None:
    """Launching without ``--scene`` should still leave the viewport / outliner usable."""
    del qapp
    window = MainWindow(services=services)
    try:
        scene = window.viewport.scene
        assert scene is not None, "expected a default empty scene to be installed"
        assert scene.root is not None
        assert scene.root.children == []  # no nodes other than the root
        # Outliner mirrors the empty scene.
        outliner_tree = window._outliner.widget()  # noqa: SLF001
        assert outliner_tree.topLevelItemCount() == 1
        # Status bar reflects the unsaved state.
        assert "Untitled" in window._status_text.text()  # noqa: SLF001
    finally:
        window.close()


def test_main_window_constructs_with_layout(qapp: Any, services: Services) -> None:
    del qapp
    from PySide6.QtWidgets import QDockWidget, QMenu  # noqa: PLC0415

    window = MainWindow(services=services)
    try:
        # Outliner + Inspector docks attached.
        dock_names = [d.objectName() for d in window.findChildren(QDockWidget)]
        assert "OutlinerDock" in dock_names
        assert "InspectorDock" in dock_names
        # Menu bar populated with the expected top-level menus.
        menu_titles = [m.title().replace("&", "") for m in window.menuBar().findChildren(QMenu)]
        for expected in ("File", "View", "Help"):
            assert any(t == expected for t in menu_titles), f"missing menu {expected!r}"
        # Status bar present.
        assert window.statusBar() is not None
    finally:
        window.close()


def test_pause_skips_simulations(qapp: Any, services: Services) -> None:
    del qapp
    window = MainWindow(services=services)
    try:
        # Track whether physics_host.tick is called.
        calls: list[float] = []
        original = services.physics_host.tick
        services.physics_host.tick = calls.append  # type: ignore[method-assign]

        window._on_toggle_pause()  # noqa: SLF001 — flips pause to True
        assert window._paused is True  # noqa: SLF001
        window._on_tick()  # noqa: SLF001
        assert calls == [], "physics tick fired while paused"

        window._on_toggle_pause()  # noqa: SLF001 — back to running
        assert window._paused is False  # noqa: SLF001
        window._on_tick()  # noqa: SLF001
        assert calls, "physics tick did not fire after unpause"

        services.physics_host.tick = original  # type: ignore[method-assign]
    finally:
        window.close()


def test_set_scene_path_populates_outliner(qapp: Any, services: Services) -> None:
    del qapp
    window = MainWindow(services=services)
    try:
        scene = Scene(root=Node(name="root"))
        window.viewport.set_scene(scene)
        window.set_scene_path(Path("dummy.glb"))
        assert window._outliner.widget().topLevelItemCount() == 1  # noqa: SLF001
        assert "dummy.glb" in window._status_text.text()  # noqa: SLF001
    finally:
        window.close()


def test_outliner_selection_routes_to_inspector(qapp: Any, services: Services) -> None:
    del qapp
    window = MainWindow(services=services)
    try:
        root = Node(name="root")
        child = Node(name="child")
        root.add_child(child)
        scene = Scene(root=root)
        window.viewport.set_scene(scene)
        window.set_scene_path(Path("dummy.glb"))

        outliner_tree = window._outliner.widget()  # noqa: SLF001
        outliner_tree.setCurrentItem(outliner_tree.topLevelItem(0).child(0))

        assert window._inspector._node is child  # noqa: SLF001
        assert "selected: child" in window._status_text.text()  # noqa: SLF001
    finally:
        window.close()


def test_script_host_restart_all_re_runs_start() -> None:
    """Loading a new scene should restart attached scripts so their start() re-fires
    against the now-populated scene."""
    from posecascade.scripting.host import ScriptHost  # noqa: PLC0415

    starts: list[str] = []
    updates: list[float] = []

    def make_hooks(name: str) -> dict:
        return {
            "start": lambda: starts.append(name),
            "update": updates.append,
        }

    host = ScriptHost()
    host.attach("demo", make_hooks("demo"))
    host.tick(1.0 / 60.0)  # first tick — start fires
    host.tick(1.0 / 60.0)  # second — start NOT re-fired

    assert starts == ["demo"]

    host.restart_all()
    host.tick(1.0 / 60.0)
    assert starts == ["demo", "demo"]


def test_reset_physics_clears_chain_state(qapp: Any, services: Services) -> None:
    del qapp
    from posecascade.animation.spring import SpringChain, SpringParams  # noqa: PLC0415
    from posecascade.scene.transform import Transform  # noqa: PLC0415
    from posecascade.utils.math3d import vec3  # noqa: PLC0415

    anchor = Node(name="anchor")
    parent = anchor
    joints = []
    for i in range(3):
        joint = Node(name=f"j{i}", transform=Transform(translation=vec3(0.0, -0.05, 0.0)))
        parent.add_child(joint)
        joints.append(joint)
        parent = joint
    chain = SpringChain.from_node_chain("test", anchor, joints, params=SpringParams())
    services.physics_host.simulator.add_chain(chain)
    # Disturb the chain to give it state.
    services.physics_host.tick(1.0 / 60.0)
    chain.joints[0].angular_velocity = vec3(1.0, 0.0, 0.0)

    window = MainWindow(services=services)
    try:
        window._on_reset_physics()  # noqa: SLF001
        for joint in chain.joints:
            assert joint.initialized is False
    finally:
        window.close()
