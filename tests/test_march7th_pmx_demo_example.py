"""Smoke tests for ``examples/march7th_pmx_demo.py`` + the editor's PMX load path.

The bundled March 7th PMX exercises the renderer's PMX-native code
path (real ``MMDMaterial`` per mesh, edge flags from the file rather
than synthesised). These tests confirm:

1. The script imports without auto-running ``main()``.
2. ``ImporterManager`` loads the bundled PMX cleanly + the imported
   scene carries an ``MMDMaterial`` on at least one mesh (i.e. the
   renderer will dispatch through ``_draw_mmd``, not the fallback).
3. The editor's ``attach_scene`` chain handles a PMX without crashing.

End-to-end PNG generation needs a real GL context; ``main()`` itself
isn't run from these tests.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEMO_PATH = _REPO_ROOT / "examples" / "march7th_pmx_demo.py"
_MARCH7TH_PMX = _REPO_ROOT / "examples" / "assets" / "march7th" / "march7th.pmx"


def _import_demo_module() -> object:
    """Load ``examples/march7th_pmx_demo.py`` without executing ``main()``."""
    if str(_DEMO_PATH.parent) not in sys.path:
        sys.path.insert(0, str(_DEMO_PATH.parent))
    spec = importlib.util.spec_from_file_location("march7th_pmx_demo", _DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["march7th_pmx_demo"] = module
    spec.loader.exec_module(module)
    return module


def test_march7th_demo_module_imports() -> None:
    """The script's module-level code defines ``main()`` without running it."""
    module = _import_demo_module()
    assert hasattr(module, "main")
    assert module._MARCH7TH_PMX == _MARCH7TH_PMX                    # noqa: SLF001


def test_bundled_march7th_pmx_loads_with_mmd_materials() -> None:
    """The bundled PMX carries real MMDMaterial — exercises the native MMD pass."""
    if not _MARCH7TH_PMX.is_file():
        pytest.skip(
            "March 7th PMX not present; see examples/assets/march7th/NOTICE.md",
        )
    sys.path.insert(0, str(_REPO_ROOT / "importers"))
    from posecascade.assets.importer_manager import ImporterManager  # noqa: PLC0415

    manager = ImporterManager(importers_root=_REPO_ROOT / "importers")
    manager.discover()
    scene = manager.load(_MARCH7TH_PMX)
    assert len(scene.meshes) > 0
    assert any(m.mmd_material is not None for m in scene.meshes), (
        "every mesh missing mmd_material — renderer would fall back to forward path"
    )
    assert len(scene.skins) == 1
    assert len(scene.skins[0].joints) > 0


def test_editor_attach_chain_handles_pmx_scene() -> None:
    """``attach_scene``'s host registration path works for a PMX without crashing."""
    if not _MARCH7TH_PMX.is_file():
        pytest.skip("March 7th PMX not present")
    sys.path.insert(0, str(_REPO_ROOT / "importers"))
    from posecascade.animation.cloth_host import ClothHost  # noqa: PLC0415
    from posecascade.animation.physics_host import PhysicsHost  # noqa: PLC0415
    from posecascade.assets.importer_manager import ImporterManager  # noqa: PLC0415

    manager = ImporterManager(importers_root=_REPO_ROOT / "importers")
    manager.discover()
    scene = manager.load(_MARCH7TH_PMX)

    # March 7th's bones don't match any registered spring-chain profile
    # (Hair* JNTs, Skirt joints, etc. don't match hair_*/前髪/スカート
    # patterns), so auto-rig correctly emits no chains. The host's
    # register_imported_scene must still complete without error.
    physics_host = PhysicsHost()
    physics_host.register_imported_scene(scene)

    cloth_host = ClothHost()
    cloth_host.register_imported_scene(scene)
    assert list(cloth_host.iter_local_state()) == []
