"""Restricted-globals loader for user scripts.

This module owns the ONE allowed ``exec`` call in the codebase. User scripts
run in a globals namespace whose ``__builtins__`` is a curated whitelist plus
the engine-provided API objects. This is *not* a hardened sandbox — Python's
introspection surface is too large to fully contain inside one process — but
it does block the obvious escape routes (``__import__``, ``open``, ``eval``,
``exec``, ``compile``, ``os``, ``sys``, ``subprocess``) so well-meaning user
scripts can't accidentally touch the filesystem or network.

For untrusted scripts, run a separate restricted process and IPC scene state
across — the API surface (``scene``, ``nodes``, ``time``, ``input``) is the
same.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from posecascade.errors import ScriptSecurityError
from posecascade.scripting.api import SCRIPT_API_HELPERS
from posecascade.scripting.physics_lite import PhysicsLite

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_FORBIDDEN_TOKENS: tuple[str, ...] = (
    "__import__",
    "__builtins__",
    "__class__",
    "__bases__",
    "__subclasses__",
    "__globals__",
    "__getattribute__",
)

_HOOK_NAMES: tuple[str, ...] = ("start", "update", "on_event")


def build_api(
    scene: Any,
    time_provider: Callable[[], float],
    physics_host: Any | None = None,
    cloth_host: Any | None = None,
    foot_planter: Any | None = None,
    skins: Any = (),
    meshes: Any = (),
    camera: Any | None = None,
    overlay: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Construct the curated API dict that user scripts receive as globals.

    Extend this surface with new keys as the engine grows; never expose engine
    internals (importer manager, GL context, asset cache) here. ``physics_host``,
    ``cloth_host`` and ``foot_planter`` are optional so unit tests can build the
    API without spinning up the full Services bundle.
    """
    api: dict[str, Any] = {
        "scene": scene,
        "time": time_provider,
        **SCRIPT_API_HELPERS,
    }
    if physics_host is not None:
        api["physics_lite"] = PhysicsLite(physics_host, cloth_host=cloth_host)
    if cloth_host is not None:
        # Exposed directly so the declarative runtime can register
        # cloth pieces + bone-following colliders without going
        # through the simpler PhysicsLite façade.
        api["cloth_host"] = cloth_host
    if foot_planter is not None:
        from posecascade.scripting.floor_api import FloorApi  # noqa: PLC0415
        api["floor"] = FloorApi(foot_planter, skins=skins, meshes=meshes)
    if camera is not None:
        api["camera"] = camera
    if overlay is not None:
        api["overlay"] = overlay
    from posecascade.scripting.morph_api import MorphApi  # noqa: PLC0415
    api["morphs"] = MorphApi()
    return api


def load_script(
    source: str,
    filename: str,
    api: dict[str, Any],
) -> dict[str, Callable[..., Any]]:
    """Compile and execute ``source``; return the user-defined hook callables.

    Raises :class:`ScriptSecurityError` if the script references a forbidden
    identifier or fails to compile.
    """
    if not isinstance(source, str):
        raise ScriptSecurityError("script source must be str")
    _reject_forbidden_tokens(source)
    try:
        compiled = compile(source, filename, "exec")
    except SyntaxError as err:
        raise ScriptSecurityError(f"syntax error in {filename}: {err}") from err

    # Use a single namespace for both globals and locals: with separate dicts
    # `exec` treats the script as if it ran inside a function body, so module-
    # level constants land in `locals_ns` and nested ``def update(dt): ...``
    # bodies cannot resolve them as free variables. Single dict = module-style
    # execution, which is what every user script expects.
    globals_ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, **api}
    # Single allowed exec — globals are restricted, source has been screened.
    exec(compiled, globals_ns)  # nosec B102  # restricted globals; see module docstring  # noqa: S102

    hooks: dict[str, Callable[..., Any]] = {}
    for hook_name in _HOOK_NAMES:
        candidate = globals_ns.get(hook_name)
        if callable(candidate):
            hooks[hook_name] = candidate
    return hooks


def _reject_forbidden_tokens(source: str) -> None:
    """Reject scripts that reference dunder attributes used to break the sandbox."""
    for token in _FORBIDDEN_TOKENS:
        if token in source:
            raise ScriptSecurityError(f"forbidden identifier in script: {token}")
