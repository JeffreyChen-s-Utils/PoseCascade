"""Load a :class:`ProjectFile` from disk + run schema migrations.

The loader:

1. Reads the raw bytes (or accepts pre-decoded text / dict).
2. Asserts the JSON top-level has a numeric ``version`` field — without
   that we refuse to guess.
3. Walks any registered :data:`MIGRATIONS` to bring older files up to
   :data:`~posecascade.project.schema.CURRENT_SCHEMA_VERSION`. v1 has no
   migrations yet; the framework is wired in so a v2 bump won't
   regress old projects.
4. Hydrates the resulting dict into the dataclass tree.

Path safety is *not* applied here — paths come back as plain relative
strings. The caller (typically :mod:`posecascade.project.sync`) runs
each through :func:`posecascade.assets.path_safety.resolve_safe` once
it knows the project root.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from posecascade.errors import MalformedAssetError
from posecascade.project.schema import (
    CURRENT_SCHEMA_VERSION,
    ProjectAudio,
    ProjectExternalParent,
    ProjectFile,
    ProjectPlayback,
    ProjectSlot,
    ProjectVersionError,
)

# Migration table: ``MIGRATIONS[version_from](payload) -> updated_payload``.
# v1 is the introductory schema, so this is empty. Adding a v2 means
# registering a migration here that bumps ``payload['version'] = 2``
# alongside any field renames / removals.
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def load_project(path: Path) -> ProjectFile:
    """Read the project file at ``path`` and return the hydrated schema."""
    path = Path(path)
    if not path.is_file():
        raise MalformedAssetError(f"project file not found: {path}")
    return parse_project(path.read_bytes())


def parse_project(source: str | bytes | dict[str, Any]) -> ProjectFile:
    """Decode ``source`` (any of: dict, JSON bytes, JSON string) into a project.

    Dict input is convenient for tests + integration code that has the
    schema in hand from another path; string / bytes get JSON-decoded.
    """
    payload = dict(source) if isinstance(source, dict) else _decode_json(source)
    payload = _migrate_to_current(payload)
    return _build_project(payload)


# ----- internal -------------------------------------------------------
_VEC3_LEN = 3
_VEC4_LEN = 4


def _decode_json(source: str | bytes) -> dict[str, Any]:
    text = source.decode("utf-8") if isinstance(source, bytes) else source
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as err:
        raise MalformedAssetError(f"invalid project JSON: {err}") from err
    if not isinstance(decoded, dict):
        raise MalformedAssetError(
            f"project JSON must be an object, got {type(decoded).__name__}",
        )
    return decoded


def _migrate_to_current(payload: dict[str, Any]) -> dict[str, Any]:
    version = payload.get("version")
    if not isinstance(version, int):
        raise MalformedAssetError("project missing integer ``version`` field")
    while version < CURRENT_SCHEMA_VERSION:
        migration = MIGRATIONS.get(version)
        if migration is None:
            raise ProjectVersionError(
                f"no migration registered from project version {version}",
            )
        payload = migration(payload)
        new_version = payload.get("version")
        if not isinstance(new_version, int) or new_version <= version:
            raise ProjectVersionError(
                f"migration from {version} did not advance the version",
            )
        version = new_version
    if version > CURRENT_SCHEMA_VERSION:
        raise ProjectVersionError(
            f"project version {version} is newer than the engine's "
            f"{CURRENT_SCHEMA_VERSION} — refusing to downgrade",
        )
    return payload


def _build_project(payload: dict[str, Any]) -> ProjectFile:
    return ProjectFile(
        version=int(payload.get("version", CURRENT_SCHEMA_VERSION)),
        name=str(payload.get("name", "")),
        slots=tuple(_build_slot(item) for item in payload.get("slots", ())),
        audio=_build_audio(payload.get("audio")),
        playback=_build_playback(payload.get("playback") or {}),
        effect_chain_toml=str(payload.get("effect_chain_toml", "")),
    )


def _build_slot(payload: dict[str, Any]) -> ProjectSlot:
    return ProjectSlot(
        name=_required(payload, "name", str),
        model_path=_required(payload, "model_path", str),
        motion_path=str(payload.get("motion_path", "")),
        visible=bool(payload.get("visible", True)),
        translation=_tuple3(payload.get("translation"), default=(0.0, 0.0, 0.0)),
        rotation=_tuple4(payload.get("rotation"), default=(0.0, 0.0, 0.0, 1.0)),
        external_parents=tuple(
            _build_external_parent(item)
            for item in payload.get("external_parents", ())
        ),
    )


def _build_external_parent(payload: dict[str, Any]) -> ProjectExternalParent:
    return ProjectExternalParent(
        self_bone_name=_required(payload, "self_bone_name", str),
        target_slot_name=_required(payload, "target_slot_name", str),
        target_bone_name=_required(payload, "target_bone_name", str),
    )


def _build_audio(payload: dict[str, Any] | None) -> ProjectAudio | None:
    if payload is None:
        return None
    return ProjectAudio(
        path=_required(payload, "path", str),
        offset_seconds=float(payload.get("offset_seconds", 0.0)),
    )


def _build_playback(payload: dict[str, Any]) -> ProjectPlayback:
    return ProjectPlayback(
        fps=int(payload.get("fps", 30)),
        start_frame=int(payload.get("start_frame", 0)),
        end_frame=int(payload.get("end_frame", 1000)),
        loop=bool(payload.get("loop", True)),
        current_frame=int(payload.get("current_frame", 0)),
    )


def _required(payload: dict[str, Any], key: str, expected_type: type) -> Any:
    value = payload.get(key)
    if not isinstance(value, expected_type):
        raise MalformedAssetError(
            f"project schema field {key!r} must be {expected_type.__name__}, "
            f"got {type(value).__name__}",
        )
    return value


def _tuple3(
    value: Any, default: tuple[float, float, float],
) -> tuple[float, float, float]:
    if value is None:
        return default
    if not isinstance(value, list | tuple) or len(value) != _VEC3_LEN:
        raise MalformedAssetError(f"vec3 expected, got {value!r}")
    return float(value[0]), float(value[1]), float(value[2])


def _tuple4(
    value: Any, default: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if value is None:
        return default
    if not isinstance(value, list | tuple) or len(value) != _VEC4_LEN:
        raise MalformedAssetError(f"vec4 expected, got {value!r}")
    return float(value[0]), float(value[1]), float(value[2]), float(value[3])
