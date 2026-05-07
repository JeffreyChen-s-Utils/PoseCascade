"""Compose a scene from a character, a quadruped, and a (procedural) room.

Defaults to the CC-licensed Khronos sample assets fetched by
``fetch_demo_assets.py``. Override individual slots with ``--character``,
``--dog``, or ``--room`` to plug in your own ``.glb`` files.

Quick start::

    python examples/fetch_demo_assets.py    # downloads CesiumMan + Fox
    python examples/scene_compose.py        # opens the viewport

Notes:

- The character placeholder is Khronos CesiumMan (CC-BY 4.0, AGI). The
  Acheron model from Honkai: Star Rail is owned by HoYoverse and is not
  redistributed publicly; pass ``--character path/to/your.glb`` to use a
  model you have legitimately.
- The "dog" placeholder is Khronos Fox (CC0 by PixelMannen). Replace via
  ``--dog`` if you have an actual German-Shepherd model.
- The room is generated as a 6-quad inward-facing box. Pass
  ``--room path/to/your.glb`` to load a real scene instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "importers") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "importers"))

from PySide6.QtWidgets import QApplication  # noqa: E402

from posecascade.app.main_window import MainWindow  # noqa: E402
from posecascade.app.registry import Services, build_services  # noqa: E402
from posecascade.assets.types import ImportedScene  # noqa: E402
from posecascade.errors import PoseCascadeError  # noqa: E402
from posecascade.scene.anime_bedroom import make_anime_bedroom_scene  # noqa: E402
from posecascade.scene.node import Node  # noqa: E402
from posecascade.scene.scene import Scene  # noqa: E402
from posecascade.scripting.api import quat_axis_angle  # noqa: E402
from posecascade.scripting.sandbox import build_api, load_script  # noqa: E402
from posecascade.utils.logging import configure_logging, get_logger  # noqa: E402
from posecascade.utils.math3d import vec3  # noqa: E402

_log = get_logger(__name__)
_DEFAULT_ASSETS_DIR = Path(__file__).parent / "assets"
_DEFAULT_SCRIPT = Path(__file__).parent / "scripts" / "idle_orbit.py"

import math  # noqa: E402  # used for the Z-up→Y-up quaternion below

_DEFAULT_CHARACTER_SCALE = 8.0   # tuned for Sketchfab humanoids exported via FBX→glTF
                                 # (those carry an embedded ×0.01 cm→m at the .fbx node)
_DEFAULT_DOG_SCALE = 0.15        # tuned for the German Shepherd Rig (no embedded scale)
_DEFAULT_ROOM_SCALE = 1.0
_DEFAULT_CHARACTER_UP = "y"      # Sketchfab's root matrix already does Z-up→Y-up
_DEFAULT_DOG_UP = "y"
_DEFAULT_ROOM_UP = "y"

_DOG_SPAWN = (2.0, 0.0, 1.0)
_CHARACTER_SPAWN = (0.5, 0.5, 0.5)
# Default camera sits near the centre of the bedroom at adult eye height,
# pointed at the character. Y < ceiling (3.58 m) and X/Z are well inside the
# room AABB (X: -7.5..10.7, Z: -4.8..14.5) so the view is clear of any wall.
# Right-click-drag on the viewport orbits the camera around the look-at
# point — see :class:`~posecascade.ui.viewport.Viewport`.
_CAMERA_POSITION = (3.0, 1.6, 5.0)
_CAMERA_TARGET = (0.5, 1.4, 0.3)


_XYZ_COMPONENTS = 3


def _parse_xyz(text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != _XYZ_COMPONENTS:
        raise argparse.ArgumentTypeError(f"expected three comma-separated floats, got {text!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])

# Static Z-up → Y-up correction (rotate -90° about world X).
_Z_UP_TO_Y_UP = quat_axis_angle(vec3(1.0, 0.0, 0.0), -math.pi / 2)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--character", type=Path, default=_DEFAULT_ASSETS_DIR / "character.glb",
        help="Humanoid character .glb (default: examples/assets/character.glb).",
    )
    parser.add_argument(
        "--dog", type=Path, default=_DEFAULT_ASSETS_DIR / "dog.glb",
        help="Quadruped .glb (default: examples/assets/dog.glb).",
    )
    parser.add_argument(
        "--room", type=Path, default=_DEFAULT_ASSETS_DIR / "room.glb",
        help="Room .glb. If missing, a procedural anime bedroom is generated.",
    )
    parser.add_argument(
        "--script", type=Path, default=_DEFAULT_SCRIPT,
        help="User script driving idle motion (default: examples/scripts/idle_orbit.py).",
    )
    parser.add_argument("--character-scale", type=float, default=_DEFAULT_CHARACTER_SCALE)
    parser.add_argument("--dog-scale", type=float, default=_DEFAULT_DOG_SCALE)
    parser.add_argument("--room-scale", type=float, default=_DEFAULT_ROOM_SCALE)
    parser.add_argument("--character-up", choices=("y", "z"), default=_DEFAULT_CHARACTER_UP,
                        help="Source up-axis of the character model (Z-up models get rotated).")
    parser.add_argument("--dog-up", choices=("y", "z"), default=_DEFAULT_DOG_UP)
    parser.add_argument("--room-up", choices=("y", "z"), default=_DEFAULT_ROOM_UP)
    parser.add_argument(
        "--camera", type=_parse_xyz, default=None,
        help="Camera position as 'X,Y,Z' (default: tuned for the Sketchfab bedroom).",
    )
    parser.add_argument(
        "--look-at", type=_parse_xyz, default=None,
        help="Camera look-at point as 'X,Y,Z' (default: room/character centre).",
    )
    return parser.parse_args(argv)


def _load_optional(services: Services, path: Path, label: str) -> ImportedScene | None:
    if not path.exists():
        _log.info("%s not found at %s — falling back", label, path)
        return None
    try:
        return services.importer_manager.load(path)
    except PoseCascadeError as err:
        _log.error("failed to load %s %s: %s", label, path, err)
        return None


def _wrap_under_holder(
    composite: Scene,
    imported: ImportedScene,
    holder_name: str,
    translation: tuple[float, float, float],
    *,
    scale: float = 1.0,
    up_axis: str = "y",
) -> Node:
    """Mount the imported scene under a named holder node and add it to ``composite``.

    The holder is what user scripts find via :meth:`Scene.find` and what they
    spin/translate. When ``up_axis == "z"`` the imported root is parented to
    a fixed ``<name>_oriented`` inner node carrying the Z-up→Y-up correction,
    so a script-driven holder rotation does not clobber the orientation fix.

    The imported scene's root node itself is attached as a child rather than
    its children being moved out one-by-one, so that ``populate_from_scene``
    (which walks ``imported.scene.root.traverse()``) still finds every
    :class:`MeshRefComponent` after composition.
    """
    holder = Node(name=holder_name)
    holder.transform.set_translation(vec3(*translation))
    if scale != 1.0:
        holder.transform.set_scale(vec3(scale, scale, scale))
    if up_axis == "z":
        oriented = Node(name=f"{holder_name}_oriented")
        oriented.transform.set_rotation(_Z_UP_TO_Y_UP)
        holder.add_child(oriented)
        attach_to = oriented
    else:
        attach_to = holder
    if imported.scene is not None and imported.scene.root.parent is None:
        attach_to.add_child(imported.scene.root)
    composite.root.add_child(holder)
    return holder


def build_composite_scene(
    services: Services,
    args: argparse.Namespace,
) -> tuple[Scene, list[ImportedScene]]:
    """Build the composite scene; returns ``(scene, imports_to_upload)``."""
    composite = Scene(name="composite")
    imports: list[ImportedScene] = []

    room_imported = _load_optional(services, args.room, "room")
    if room_imported is None:
        _log.info("using procedural anime bedroom (override with --room)")
        room_imported = make_anime_bedroom_scene()
        room_up = "y"  # the procedural builder is Y-up by construction
        room_scale = 1.0
    else:
        room_up = args.room_up
        room_scale = args.room_scale
    _wrap_under_holder(
        composite, room_imported, "room", (0.0, 0.0, 0.0),
        scale=room_scale, up_axis=room_up,
    )
    imports.append(room_imported)

    character_imported = _load_optional(services, args.character, "character")
    if character_imported is not None:
        _wrap_under_holder(
            composite, character_imported, "character",
            _CHARACTER_SPAWN,
            scale=args.character_scale, up_axis=args.character_up,
        )
        imports.append(character_imported)
    else:
        _log.warning("character not loaded — pass --character or run fetch_sketchfab.py")

    dog_imported = _load_optional(services, args.dog, "dog")
    if dog_imported is not None:
        _wrap_under_holder(
            composite, dog_imported, "dog",
            _DOG_SPAWN,
            scale=args.dog_scale, up_axis=args.dog_up,
        )
        imports.append(dog_imported)
    else:
        _log.warning("dog not loaded — pass --dog or run fetch_sketchfab.py")

    return composite, imports


def _wire_renderer(window: MainWindow, imports: list[ImportedScene]) -> None:
    """Upload meshes to GPU once the viewport has reported its renderer is ready."""
    def _on_ready(_ctx: object) -> None:
        renderer = window.viewport.renderer
        if renderer is None:
            return
        for imported in imports:
            renderer.populate_from_scene(imported)
        _log_world_aabbs(window.viewport.scene, imports)
        window.viewport.update()

    if window.viewport.renderer is not None:
        _on_ready(None)
    else:
        window.viewport.initialized.connect(_on_ready)


def _log_world_aabbs(scene: Scene | None, imports: list[ImportedScene]) -> None:
    """Print each import's world AABB. Cheap diagnostic for empty frames.

    Each imported scene gets its OWN flat mesh list, so we walk per-import
    and index meshes locally — using a single global flat list would alias
    indices across imports and produce nonsense AABBs.
    """
    if scene is None:
        return
    import numpy as np  # noqa: PLC0415 — diagnostic-only

    from posecascade.render.renderer import _world_matrix  # noqa: PLC0415
    from posecascade.scene.component import MeshRefComponent  # noqa: PLC0415

    for imported in imports:
        if imported.scene is None:
            continue
        pts: list[np.ndarray] = []
        for node in imported.scene.root.traverse():
            for component in node.components:
                if not isinstance(component, MeshRefComponent):
                    continue
                world_m = _world_matrix(node)
                for idx in component.mesh_indices:
                    if 0 <= idx < len(imported.meshes):
                        verts = imported.meshes[idx].positions
                        h = np.hstack([verts, np.ones((verts.shape[0], 1), dtype=np.float32)])
                        pts.append((world_m @ h.T).T[:, :3])
        # Find the holder that ends up containing this import (climb until composite root).
        ancestor = imported.scene.root
        while ancestor.parent is not None and ancestor.parent.parent is not None:
            ancestor = ancestor.parent
        label = ancestor.name
        if not pts:
            _log.info("  %s: no points", label)
            continue
        all_pts = np.vstack(pts)
        mn, mx = all_pts.min(axis=0), all_pts.max(axis=0)
        _log.info(
            "  %s: world AABB min=%s max=%s extent=%s",
            label, mn.round(2).tolist(), mx.round(2).tolist(),
            (mx - mn).round(2).tolist(),
        )


def _attach_idle_script(
    window: MainWindow, services: Services, script_path: Path
) -> None:
    if not script_path.exists():
        _log.info("idle script not found at %s — skipping", script_path)
        return
    api = build_api(
        scene=window.viewport.scene,
        time_provider=lambda: window.clock.elapsed,
    )
    try:
        hooks = load_script(script_path.read_text(encoding="utf-8"), str(script_path), api)
    except PoseCascadeError as err:
        _log.error("script load failed: %s", err)
        return
    services.script_host.attach(name=script_path.stem, hooks=hooks)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging("INFO")
    app = QApplication.instance() or QApplication(sys.argv)

    services = build_services(project_root=_PROJECT_ROOT)
    window = MainWindow(services=services)
    window.resize(1280, 800)

    composite, imports = build_composite_scene(services, args)
    window.viewport.set_scene(composite)
    # Camera: CLI override wins, otherwise use the Sketchfab-bedroom default.
    cam_pos = args.camera if args.camera is not None else _CAMERA_POSITION
    cam_tgt = args.look_at if args.look_at is not None else _CAMERA_TARGET
    window.viewport.camera.position = vec3(*cam_pos)
    window.viewport.camera.target = vec3(*cam_tgt)
    _log.info("camera at %s looking at %s", tuple(cam_pos), tuple(cam_tgt))

    _wire_renderer(window, imports)
    _attach_idle_script(window, services, args.script)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
