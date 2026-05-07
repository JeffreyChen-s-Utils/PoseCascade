"""Per-frame driver for loaded user scripts."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from posecascade.errors import ScriptRuntimeError
from posecascade.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class _LoadedScript:
    name: str
    hooks: dict[str, Callable[..., Any]]
    started: bool = False
    disabled: bool = False


@dataclass
class ScriptHost:
    """Owns the loaded scripts and dispatches per-frame ``update(dt)`` calls.

    Any user-script exception is caught, logged, and the offending script is
    disabled — one bad script must never freeze the timeline.
    """

    _scripts: list[_LoadedScript] = field(default_factory=list)

    def attach(self, name: str, hooks: dict[str, Callable[..., Any]]) -> None:
        """Register a loaded script. ``hooks`` comes from
        :func:`posecascade.scripting.sandbox.load_script`."""
        self._scripts.append(_LoadedScript(name=name, hooks=hooks))

    def restart_all(self) -> None:
        """Mark every attached script as not-started so ``start()`` fires again next tick.

        Used when the active scene changes — scripts that ran their ``start()``
        hook against the previous (or empty) scene need to re-initialise so
        ``scene.find(...)`` lookups land on the newly loaded nodes.
        """
        for script in self._scripts:
            script.started = False
            script.disabled = False

    def tick(self, delta_seconds: float) -> None:
        """Run one frame: call ``start()`` on first tick, then ``update(dt)``."""
        for script in self._scripts:
            if script.disabled:
                continue
            if not script.started:
                self._safe_call(script, "start")
                script.started = True
            self._safe_call(script, "update", delta_seconds)

    def dispatch_event(self, name: str, payload: object) -> None:
        """Forward an arbitrary event to every script's ``on_event(name, payload)`` hook."""
        for script in self._scripts:
            if script.disabled:
                continue
            self._safe_call(script, "on_event", name, payload)

    def _safe_call(self, script: _LoadedScript, hook_name: str, *args: object) -> None:
        hook = script.hooks.get(hook_name)
        if hook is None:
            return
        try:
            hook(*args)
        except Exception as err:  # noqa: BLE001 — boundary; convert + disable
            script.disabled = True
            _log.error("script %r disabled: %s in %s", script.name, err, hook_name)
            raise ScriptRuntimeError(
                f"script {script.name!r} raised in {hook_name}"
            ) from err
