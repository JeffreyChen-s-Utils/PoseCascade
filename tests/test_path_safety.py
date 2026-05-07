"""Tests for :mod:`posecascade.assets.path_safety`."""
from __future__ import annotations

from pathlib import Path

import pytest

from posecascade.assets.path_safety import resolve_safe
from posecascade.errors import UnsafePathError


def test_resolve_safe_happy_path(tmp_path: Path) -> None:
    asset = tmp_path / "textures" / "wall.png"
    asset.parent.mkdir()
    asset.write_bytes(b"fake")
    resolved = resolve_safe(tmp_path, "textures/wall.png")
    assert resolved == asset.resolve()


def test_resolve_safe_allows_nonexistent_target(tmp_path: Path) -> None:
    # Asset references resolve before the file is opened — nonexistent is allowed.
    resolved = resolve_safe(tmp_path, "textures/will_be_created.png")
    assert resolved.parent == (tmp_path / "textures").resolve()


def test_resolve_safe_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(UnsafePathError):
        resolve_safe(tmp_path, "")


def test_resolve_safe_rejects_absolute(tmp_path: Path) -> None:
    absolute_ref = "/etc/passwd" if Path("/").exists() else "C:/Windows/system32"
    with pytest.raises(UnsafePathError):
        resolve_safe(tmp_path, absolute_ref)


def test_resolve_safe_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(UnsafePathError):
        resolve_safe(tmp_path, "../../../etc/passwd")


def test_resolve_safe_rejects_traversal_disguised_as_subdir(tmp_path: Path) -> None:
    with pytest.raises(UnsafePathError):
        resolve_safe(tmp_path, "subdir/../../../escape.txt")


def test_resolve_safe_rejects_missing_root(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist"
    with pytest.raises(UnsafePathError):
        resolve_safe(nonexistent, "anything.txt")
