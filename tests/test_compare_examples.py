"""Smoke tests for the side-by-side comparison scripts under ``examples/``.

The comparison scripts (compare_bloom / compare_tone / compare_dqs /
compare_lights) each demonstrate a single MMD-fluence feature against
its baseline. Tests here verify that the module imports cleanly,
the shared helpers in ``_demo_lib`` expose the documented surface,
and the chain-building helpers in each comparison return what the
description claims.

End-to-end PNG generation needs a GL context + a model on disk and
is gated by the ``gl_context`` fixture; ``main()`` is not run from
these tests.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"


def _import_example(name: str) -> object:
    """Load ``examples/<name>.py`` as a module.

    The comparison scripts live outside any package, so we spec-load
    by file path. ``examples/`` is prepended to ``sys.path`` first so
    each script's ``from _demo_lib import ...`` resolves.
    """
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_demo_lib_exposes_documented_helpers() -> None:
    """``_demo_lib`` carries the helpers every comparison script imports."""
    lib = _import_example("_demo_lib")
    for name in (
        "setup_offscreen_gl", "load_character", "make_renderer",
        "read_pixels", "save_side_by_side",
        "DEFAULT_WIDTH", "DEFAULT_HEIGHT",
    ):
        assert hasattr(lib, name), f"_demo_lib missing {name!r}"


def test_compare_bloom_chain_overrides_threshold() -> None:
    """The bloom comparison lowers the AutoLuminous threshold so it actually fires."""
    module = _import_example("compare_bloom")
    chain = module._build_bloom_chain()                              # noqa: SLF001
    assert len(chain) == 1
    entry = chain.entries[0]
    assert entry.descriptor.name == "autoluminous"
    # Default threshold is 0.85; the script overrides to 0.4 so the
    # bundled (non-emissive) character actually shows bloom.
    assert entry.uniform_overrides.get("threshold") == 0.40


def test_compare_tone_chain_is_mmd_tone_only() -> None:
    """The tone comparison applies the ``mmd_tone`` effect on its own."""
    module = _import_example("compare_tone")
    chain = module._build_tone_chain()                               # noqa: SLF001
    assert len(chain) == 1
    assert chain.entries[0].descriptor.name == "mmd_tone"


def test_compare_dqs_targets_right_upper_arm_at_extreme_angle() -> None:
    """The DQS comparison pose is large enough to surface candy-wrapper.

    Documenting the angle in the test so a future tweak doesn't silently
    drop it below the threshold where LBS and DQS diverge visibly.
    """
    module = _import_example("compare_dqs")
    assert module._TWIST_BONE == "J_Bip_R_UpperArm"                  # noqa: SLF001
    assert abs(module._TWIST_ANGLE_RADIANS) >= 1.5, (                # noqa: SLF001
        "twist angle too small — LBS / DQS divergence will be invisible"
    )


def test_compare_lights_module_imports() -> None:
    """The lights comparison module loads without running ``main()``."""
    module = _import_example("compare_lights")
    assert hasattr(module, "main")
    assert hasattr(module, "_draw_pane")
