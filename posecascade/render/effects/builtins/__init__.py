"""Built-in PoseCascade effects.

Each effect ships as a TOML descriptor + a GLSL fragment shader, both
loaded lazily so the renderer doesn't pay the file-read cost until
:func:`register_builtins` is called. The descriptors live in this
directory; the fragments live in ``shaders/effects/``.
"""
from __future__ import annotations

from pathlib import Path

from posecascade.render.effects.chain import EffectLibrary
from posecascade.render.effects.descriptor import EffectDescriptor
from posecascade.render.effects.loader import load_descriptor_from_toml

_BUILTIN_NAMES: tuple[str, ...] = (
    "autoluminous",
    "hgshadow",
    "o_greener",
    "ikeshita_ray",
)


def builtin_descriptor_path(name: str) -> Path:
    """Return the on-disk path of the named built-in's TOML descriptor."""
    return Path(__file__).resolve().parent / f"{name}.toml"


def load_builtin(name: str) -> EffectDescriptor:
    """Load one built-in by short name (raises :class:`MalformedAssetError`)."""
    return load_descriptor_from_toml(builtin_descriptor_path(name))


def register_builtins(library: EffectLibrary) -> EffectLibrary:
    """Register every built-in descriptor on ``library`` and return it.

    Idempotent — re-calling overwrites the registry entry with the same
    descriptor instance.
    """
    for name in _BUILTIN_NAMES:
        library.register(load_builtin(name))
    return library


def builtin_names() -> tuple[str, ...]:
    return _BUILTIN_NAMES
