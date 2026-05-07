"""TOML I/O for effect descriptors + chains.

We pick TOML over a custom-grammar text format for two reasons: it
ships in stdlib (Python 3.11+ ``tomllib``), and the strict structure
keeps a hand-edited ``.toml`` from quietly producing a half-loaded
descriptor. Unknown keys are ignored — forward-compat for descriptors
that reference effect-engine features the runtime hasn't implemented
yet.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from posecascade.errors import MalformedAssetError
from posecascade.render.effects.chain import ChainEntry, EffectChain, EffectLibrary
from posecascade.render.effects.descriptor import (
    EffectBlendMode,
    EffectDescriptor,
    EffectInput,
    EffectUniform,
    EffectUniformKind,
)


def load_descriptor_from_toml(source: str | bytes | Path) -> EffectDescriptor:
    """Decode an effect descriptor from a TOML source.

    ``source`` may be a path, raw bytes, or an already-decoded string.
    The function dispatches by type so callers can drop in a file path
    or ship inline TOML in a test fixture without translating it first.
    """
    return _build_descriptor(_load_table(source))


def serialize_chain_to_toml(chain: EffectChain) -> str:
    """Round-trip a :class:`EffectChain` to a TOML string.

    The output is keyed under ``[[entry]]`` blocks; loading the same
    string back through :func:`load_chain_from_toml` (with a matching
    library) reproduces the chain.
    """
    parts: list[str] = []
    for entry in chain.entries:
        parts.append("[[entry]]")
        parts.append(f'name = "{_escape(entry.descriptor.name)}"')
        parts.append(f"enabled = {str(entry.enabled).lower()}")
        if entry.uniform_overrides:
            parts.append("[entry.uniforms]")
            for uniform_name, value in entry.uniform_overrides.items():
                parts.append(f"{uniform_name} = {_format_value(value)}")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def load_chain_from_toml(
    source: str | bytes | Path, library: EffectLibrary,
) -> EffectChain:
    """Parse a serialised chain + resolve descriptor references via ``library``.

    Entries whose descriptor name isn't registered in ``library`` are
    silently skipped — the chain UI shows the user what loaded and the
    rest can be re-added when the missing descriptor ships.
    """
    table = _load_table(source)
    chain = EffectChain()
    for entry_table in table.get("entry", ()):
        name = entry_table.get("name")
        if not isinstance(name, str):
            continue
        descriptor = library.find(name)
        if descriptor is None:
            continue
        entry = ChainEntry(
            descriptor=descriptor,
            enabled=bool(entry_table.get("enabled", True)),
            uniform_overrides=dict(entry_table.get("uniforms", {})),
        )
        chain.entries.append(entry)
    return chain


# ----- internal -------------------------------------------------------
def _load_table(source: str | bytes | Path) -> dict[str, Any]:
    if isinstance(source, Path):
        try:
            data = source.read_bytes()
        except OSError as err:
            raise MalformedAssetError(f"effect descriptor not found: {source}") from err
    elif isinstance(source, str):
        data = source.encode("utf-8")
    else:
        data = source
    try:
        return tomllib.loads(data.decode("utf-8"))
    except tomllib.TOMLDecodeError as err:
        raise MalformedAssetError(f"invalid effect TOML: {err}") from err


def _build_descriptor(table: dict[str, Any]) -> EffectDescriptor:
    name = _required(table, "name", str)
    fragment = _required(table, "fragment_shader", str)
    return EffectDescriptor(
        name=name,
        fragment_shader=fragment,
        inputs=tuple(_build_input(item) for item in table.get("inputs", ())),
        uniforms=tuple(_build_uniform(item) for item in table.get("uniforms", ())),
        blend_mode=_parse_blend_mode(table.get("blend_mode", "replace")),
        output_name=str(table.get("output_name", "result")),
        description=str(table.get("description", "")),
    )


def _build_input(item: dict[str, Any]) -> EffectInput:
    return EffectInput(
        sampler_name=_required(item, "sampler_name", str),
        source=str(item.get("source", "main_color")),
    )


def _build_uniform(item: dict[str, Any]) -> EffectUniform:
    kind = _parse_uniform_kind(item.get("kind", "scalar"))
    default = item.get("default", 0.0)
    if kind in (EffectUniformKind.VEC3_COLOR, EffectUniformKind.VEC4_COLOR):
        default = tuple(float(v) for v in default)
    return EffectUniform(
        name=_required(item, "name", str),
        kind=kind,
        default=default,
        minimum=_optional_float(item.get("minimum")),
        maximum=_optional_float(item.get("maximum")),
        step=_optional_float(item.get("step")),
        enum_labels=tuple(str(label) for label in item.get("enum_labels", ())),
        description=str(item.get("description", "")),
    )


def _required(table: dict[str, Any], key: str, expected_type: type) -> Any:
    value = table.get(key)
    if not isinstance(value, expected_type):
        raise MalformedAssetError(
            f"effect descriptor missing required {expected_type.__name__} "
            f"field {key!r}",
        )
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _parse_blend_mode(value: Any) -> EffectBlendMode:
    if not isinstance(value, str):
        return EffectBlendMode.REPLACE
    try:
        return EffectBlendMode[value.upper()]
    except KeyError as err:
        raise MalformedAssetError(f"unknown effect blend_mode: {value!r}") from err


def _parse_uniform_kind(value: Any) -> EffectUniformKind:
    if not isinstance(value, str):
        raise MalformedAssetError(f"uniform kind must be a string, got {value!r}")
    try:
        return EffectUniformKind[value.upper()]
    except KeyError as err:
        raise MalformedAssetError(f"unknown uniform kind: {value!r}") from err


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(float(value))
    if isinstance(value, tuple | list):
        items = ", ".join(_format_value(v) for v in value)
        return f"[{items}]"
    if isinstance(value, str):
        return f'"{_escape(value)}"'
    raise MalformedAssetError(f"unsupported uniform value type: {type(value).__name__}")
