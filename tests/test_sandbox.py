"""Tests for :mod:`posecascade.scripting.sandbox`."""
from __future__ import annotations

import pytest

from posecascade.errors import ScriptSecurityError
from posecascade.scripting.sandbox import build_api, load_script


def _empty_api() -> dict[str, object]:
    return build_api(scene=object(), time_provider=lambda: 0.0)


def test_load_script_returns_user_hooks() -> None:
    source = "def update(dt):\n    pass\n"
    hooks = load_script(source, "<test>", _empty_api())
    assert "update" in hooks
    assert callable(hooks["update"])


def test_load_script_picks_up_start_and_on_event() -> None:
    source = (
        "def start():\n    pass\n"
        "def update(dt):\n    pass\n"
        "def on_event(name, payload):\n    pass\n"
    )
    hooks = load_script(source, "<test>", _empty_api())
    assert set(hooks.keys()) == {"start", "update", "on_event"}


def test_load_script_rejects_dunder_import() -> None:
    source = "x = __import__('os').listdir('.')\n"
    with pytest.raises(ScriptSecurityError):
        load_script(source, "<test>", _empty_api())


def test_load_script_rejects_class_walk() -> None:
    source = "x = ().__class__.__bases__\n"
    with pytest.raises(ScriptSecurityError):
        load_script(source, "<test>", _empty_api())


def test_load_script_rejects_syntax_error() -> None:
    with pytest.raises(ScriptSecurityError):
        load_script("def update(dt:\n", "<test>", _empty_api())


def test_load_script_rejects_non_string_source() -> None:
    with pytest.raises(ScriptSecurityError):
        load_script(b"def update(dt): pass", "<test>", _empty_api())  # type: ignore[arg-type]


def test_load_script_blocks_open_via_missing_builtin() -> None:
    # `open` is not in the safe builtins list, so the script raises NameError.
    source = "x = open('/etc/passwd')\n"
    api = _empty_api()
    with pytest.raises(NameError):
        load_script(source, "<test>", api)


def test_load_script_exposes_curated_api() -> None:
    captured: list[object] = []
    api = build_api(scene="my-scene", time_provider=lambda: 1.5)
    source = "captured.append(scene)\ncaptured.append(time())\n"
    api["captured"] = captured  # type: ignore[assignment]
    load_script(source, "<test>", api)
    assert captured == ["my-scene", 1.5]


def test_module_level_constants_visible_inside_hooks() -> None:
    """Regression: ``def update`` must resolve module-level names.

    Earlier ``exec(code, globals_ns, locals_ns)`` made script-level assignments
    land in ``locals_ns``, which the nested ``update`` body could not see (free
    variables only resolve through globals). Single-dict exec fixes it.
    """
    captured: list[float] = []
    api = build_api(scene=None, time_provider=lambda: 0.0)
    api["captured"] = captured  # type: ignore[assignment]
    source = (
        "PERIOD = 8.0\n"
        "def update(dt):\n"
        "    captured.append(PERIOD)\n"
    )
    hooks = load_script(source, "<test>", api)
    hooks["update"](0.016)
    assert captured == [8.0]
