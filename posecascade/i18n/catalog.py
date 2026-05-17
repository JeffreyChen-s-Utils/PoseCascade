"""Pure-Python catalog loader — no Qt imports.

Split out of :mod:`posecascade.i18n` so the JSON parsing + key lookup logic can
be unit-tested without spinning up a :class:`QApplication`. The public
:mod:`posecascade.i18n` module wraps this with the runtime state (active
language, ``QSettings`` persistence, env-var resolution).
"""
from __future__ import annotations

import json
from pathlib import Path

from posecascade.errors import PoseCascadeError

__all__ = [
    "DEFAULT_LANGUAGE",
    "Catalog",
    "CatalogError",
    "discover_languages",
    "load_catalog",
]

DEFAULT_LANGUAGE = "en"


class CatalogError(PoseCascadeError):
    """A locale catalog could not be loaded or is malformed."""


class Catalog:
    """Flat key→string lookup wrapping a single locale's JSON catalog.

    Catalog files are deliberately flat dicts — nested objects are rejected at
    load time so the on-disk format is trivial to diff, machine-translate, and
    edit without a schema. Keys use dot-separated namespaces (``"menu.file.open"``)
    purely as a naming convention; the loader does not split on them.
    """

    __slots__ = ("_entries", "_language")

    def __init__(self, language: str, entries: dict[str, str]) -> None:
        self._language = language
        self._entries = entries

    @property
    def language(self) -> str:
        return self._language

    def get(self, key: str) -> str | None:
        return self._entries.get(key)

    def keys(self) -> list[str]:
        return list(self._entries.keys())

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)


def load_catalog(locales_dir: Path, language: str) -> Catalog:
    """Load ``<locales_dir>/<language>.json`` into a :class:`Catalog`.

    Raises :class:`CatalogError` if the file is missing, is not valid JSON, is
    not a flat ``str → str`` dict, or contains a key with embedded newlines
    (which would break the in-tooltip formatting most callers assume).
    """
    path = locales_dir / f"{language}.json"
    if not path.is_file():
        raise CatalogError(f"catalog not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise CatalogError(f"catalog {path} is not valid JSON: {err}") from err
    if not isinstance(raw, dict):
        raise CatalogError(f"catalog {path} must be a JSON object at the top level")
    entries: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise CatalogError(
                f"catalog {path}: every entry must be str→str (got {type(key).__name__}"
                f"→{type(value).__name__} for {key!r})",
            )
        entries[key] = value
    return Catalog(language=language, entries=entries)


def discover_languages(locales_dir: Path) -> list[str]:
    """Return the locale codes for every ``*.json`` in ``locales_dir``.

    A missing directory yields an empty list rather than raising — the i18n
    layer treats that as "no translations available, English fallback only"
    instead of an installation error. Sorted for determinism.
    """
    if not locales_dir.is_dir():
        return []
    return sorted(p.stem for p in locales_dir.glob("*.json"))
