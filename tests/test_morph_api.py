"""Tests for the sandbox-facing :class:`MorphApi` weight-driver surface."""
from __future__ import annotations

from posecascade.scripting.morph_api import MorphApi


def test_set_writes_into_internal_dict_by_default() -> None:
    """Default constructor owns its own dict; ``set`` overwrites entries."""
    api = MorphApi()
    api.set("smile", 0.5)
    api.set("blink", 1.0)
    assert dict(api.current_weights()) == {"smile": 0.5, "blink": 1.0}


def test_set_writes_through_to_external_target() -> None:
    """Passing an explicit ``target`` lets the engine share the dict
    with the renderer / morph applier — ``set`` mutates it in place."""
    target: dict[str, float] = {}
    api = MorphApi(target)
    api.set("frown", 0.75)
    assert target == {"frown": 0.75}
    assert api.current_weights() is target


def test_set_invokes_setter_when_provided() -> None:
    """``setter`` callbacks let the engine react synchronously to each
    weight write (e.g. dirty-marking the morph applier)."""
    calls: list[tuple[str, float]] = []
    api = MorphApi(setter=lambda name, weight: calls.append((name, weight)))
    api.set("smile", 0.25)
    api.set("smile", 0.75)
    assert calls == [("smile", 0.25), ("smile", 0.75)]


def test_clear_drops_all_weights() -> None:
    api = MorphApi()
    api.set("a", 0.1)
    api.set("b", 0.2)
    api.clear()
    assert dict(api.current_weights()) == {}


def test_update_applies_batch() -> None:
    """``update`` is the bulk equivalent of ``set`` — useful for the
    declarative runtime's per-phase morph evaluation."""
    api = MorphApi()
    api.update({"a": 0.1, "b": 0.5})
    assert dict(api.current_weights()) == {"a": 0.1, "b": 0.5}


def test_update_does_not_clear_previous_keys() -> None:
    """Calling ``update`` accumulates over previous frames unless the
    caller explicitly calls ``clear`` — matches the declarative
    runtime's "previous-phase morph stays at its last value until
    overwritten" semantics."""
    api = MorphApi()
    api.set("a", 0.5)
    api.update({"b": 1.0})
    assert dict(api.current_weights()) == {"a": 0.5, "b": 1.0}
