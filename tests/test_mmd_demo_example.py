"""Smoke test for the bundled ``examples/mmd_demo.py`` script.

Imports the module (without running ``main()``) and pokes the helper
functions so the script stays importable across refactors. The
end-to-end ``main()`` is opt-in — it needs a GL context and writes
a PNG, both of which we skip for the unit-test default run.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEMO_PATH = _REPO_ROOT / "examples" / "mmd_demo.py"


def _import_demo_module() -> object:
    """Load ``examples/mmd_demo.py`` as a module without executing ``main()``.

    Standalone scripts under ``examples/`` aren't part of the
    ``posecascade`` package — they're meant to be run as ``python
    examples/foo.py``. To unit-test them we have to spec-load the
    file, which is what this helper does.
    """
    spec = importlib.util.spec_from_file_location("mmd_demo", _DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("mmd_demo", module)
    spec.loader.exec_module(module)
    return module


def test_mmd_demo_script_imports_without_running_main() -> None:
    """The script's module-level code defines helpers but doesn't auto-execute."""
    module = _import_demo_module()
    assert hasattr(module, "main")
    assert hasattr(module, "_build_mmd_renderer")
    assert hasattr(module, "_build_default_effect_chain")


def test_mmd_demo_effect_chain_contains_autoluminous_and_mmd_tone() -> None:
    """The canonical MMD post chain bundles AutoLuminous + ``mmd_tone`` in order."""
    module = _import_demo_module()
    chain = module._build_default_effect_chain()                     # noqa: SLF001
    names = [entry.descriptor.name for entry in chain]
    assert names == ["autoluminous", "mmd_tone"], (
        f"expected the canonical post chain, got {names!r}"
    )


def test_mmd_demo_renderer_helper_sets_every_mmd_toggle(
    gl_context: object,
) -> None:
    """``_build_mmd_renderer`` flips every MMD-fluence toggle, not just one or two."""
    _ = gl_context  # the renderer's initialize() needs a current GL context
    module = _import_demo_module()
    renderer = module._build_mmd_renderer(_REPO_ROOT / "shaders")     # noqa: SLF001
    # The whole point of the helper is the comprehensive preset — if a
    # future refactor accidentally drops one of these, the demo silently
    # stops being a fair MMD reference.
    assert renderer._force_toon_shading is True                       # noqa: SLF001
    assert renderer._dqs_enabled is True                              # noqa: SLF001
    assert renderer._ground_enabled is True                           # noqa: SLF001
    assert renderer._projected_shadow_enabled is True                 # noqa: SLF001
    assert renderer._self_shadow_enabled is True                      # noqa: SLF001
    assert renderer._sky_enabled is True                              # noqa: SLF001
    assert renderer._srgb_output_enabled is True                      # noqa: SLF001
    assert len(renderer._secondary_lights) == 2                       # noqa: SLF001
