"""Pure-Python tests for :mod:`posecascade.i18n.catalog`.

The catalog module deliberately has no Qt imports so these tests stay headless
— they don't need a ``QApplication`` and run quickly under any CI matrix slot.
Coverage maps to the Definition-of-Done expectations in CLAUDE.md: happy path,
empty edge case, malformed JSON, schema rejection, and the discovery of an
absent locales dir.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from posecascade.errors import PoseCascadeError
from posecascade.i18n.catalog import (
    DEFAULT_LANGUAGE,
    Catalog,
    CatalogError,
    discover_languages,
    load_catalog,
)


def _write(tmp_path: Path, name: str, payload: object) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_default_language_is_en() -> None:
    """The English catalog is the universal fallback — codify the contract."""
    assert DEFAULT_LANGUAGE == "en"


def test_load_catalog_happy_path(tmp_path: Path) -> None:
    _write(tmp_path, "en", {"menu.file": "&File", "status.ready": "Ready"})
    catalog = load_catalog(tmp_path, "en")
    assert isinstance(catalog, Catalog)
    assert catalog.language == "en"
    assert catalog.get("menu.file") == "&File"
    assert catalog.get("status.ready") == "Ready"


def test_load_catalog_missing_key_returns_none(tmp_path: Path) -> None:
    """Missing keys must be ``None`` so the i18n layer can fall back, not raise."""
    _write(tmp_path, "en", {"a": "1"})
    catalog = load_catalog(tmp_path, "en")
    assert catalog.get("does.not.exist") is None


def test_load_catalog_contains_and_len(tmp_path: Path) -> None:
    _write(tmp_path, "en", {"a": "1", "b": "2", "c": "3"})
    catalog = load_catalog(tmp_path, "en")
    assert "a" in catalog
    assert "z" not in catalog
    assert len(catalog) == 3
    assert sorted(catalog.keys()) == ["a", "b", "c"]


def test_load_catalog_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="catalog not found"):
        load_catalog(tmp_path, "xx")


def test_load_catalog_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "en.json"
    path.write_text("not json {", encoding="utf-8")
    with pytest.raises(CatalogError, match="not valid JSON"):
        load_catalog(tmp_path, "en")


def test_load_catalog_non_object_root_raises(tmp_path: Path) -> None:
    """The schema demands a flat object — arrays and scalars get rejected."""
    _write(tmp_path, "en", ["a", "b"])
    with pytest.raises(CatalogError, match="JSON object at the top level"):
        load_catalog(tmp_path, "en")


def test_load_catalog_non_string_value_raises(tmp_path: Path) -> None:
    _write(tmp_path, "en", {"key": 123})
    with pytest.raises(CatalogError, match="every entry must be str→str"):
        load_catalog(tmp_path, "en")


def test_load_catalog_nested_object_value_raises(tmp_path: Path) -> None:
    """Nested values would invite ambiguity about key paths; reject up-front."""
    _write(tmp_path, "en", {"key": {"inner": "value"}})
    with pytest.raises(CatalogError, match="every entry must be str→str"):
        load_catalog(tmp_path, "en")


def test_catalog_error_is_posecascade_error() -> None:
    """``CatalogError`` slots into the domain hierarchy so callers can use the
    base type when they want any i18n failure."""
    assert issubclass(CatalogError, PoseCascadeError)


def test_discover_languages_finds_all_json(tmp_path: Path) -> None:
    _write(tmp_path, "en", {})
    _write(tmp_path, "zh-TW", {})
    _write(tmp_path, "ja", {})
    # Non-JSON files are ignored.
    (tmp_path / "README.md").write_text("not a catalog", encoding="utf-8")
    assert discover_languages(tmp_path) == ["en", "ja", "zh-TW"]


def test_discover_languages_missing_dir_is_empty() -> None:
    """A missing locales dir is treated as 'no catalogs' rather than an error."""
    assert discover_languages(Path("/definitely/not/a/real/path")) == []


def test_discover_languages_empty_dir(tmp_path: Path) -> None:
    assert discover_languages(tmp_path) == []
