"""Tests for :class:`posecascade.assets.importer_manager.ImporterManager`."""
from __future__ import annotations

from pathlib import Path

import pytest

from posecascade.assets.importer_manager import ImporterManager
from posecascade.errors import UnsupportedFormatError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _manager() -> ImporterManager:
    manager = ImporterManager(importers_root=PROJECT_ROOT / "importers")
    manager.discover()
    return manager


def test_discovers_core_extensions() -> None:
    manager = _manager()
    # At least the core formats should always register.
    for ext in (".gltf", ".glb", ".obj", ".stl", ".ply"):
        assert manager.importer_for(Path(f"x{ext}")) is not None


def test_discover_is_idempotent() -> None:
    manager = ImporterManager(importers_root=PROJECT_ROOT / "importers")
    manager.discover()
    first_size = len(manager._by_extension)  # noqa: SLF001 — testing internal state
    manager.discover()
    assert len(manager._by_extension) == first_size  # noqa: SLF001


def test_unknown_extension_raises() -> None:
    manager = _manager()
    with pytest.raises(UnsupportedFormatError):
        manager.importer_for(Path("model.xyz"))


def test_load_dispatches_to_importer(tmp_path: Path) -> None:
    manager = _manager()
    # Dispatch picks the right importer; missing-file then raises
    # MalformedAssetError. The signal is "we reached the loader".
    from posecascade.errors import MalformedAssetError  # noqa: PLC0415
    with pytest.raises(MalformedAssetError):
        manager.load(tmp_path / "missing.gltf")


def test_missing_importers_root_is_no_op(tmp_path: Path) -> None:
    manager = ImporterManager(importers_root=tmp_path / "does_not_exist")
    manager.discover()  # must not raise
    with pytest.raises(UnsupportedFormatError):
        manager.importer_for(Path("model.gltf"))
