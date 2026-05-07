"""Smoke tests for the exception hierarchy."""
from __future__ import annotations

from posecascade.errors import (
    AssetError,
    GLError,
    MalformedAssetError,
    PoseCascadeError,
    SceneError,
    ScriptError,
    ScriptRuntimeError,
    ScriptSecurityError,
    UnsafePathError,
    UnsupportedFormatError,
)


def test_all_errors_derive_from_root() -> None:
    leaves = (
        AssetError,
        MalformedAssetError,
        UnsupportedFormatError,
        UnsafePathError,
        GLError,
        SceneError,
        ScriptError,
        ScriptSecurityError,
        ScriptRuntimeError,
    )
    for cls in leaves:
        assert issubclass(cls, PoseCascadeError), cls


def test_security_error_is_a_script_error() -> None:
    assert issubclass(ScriptSecurityError, ScriptError)
    assert issubclass(ScriptRuntimeError, ScriptError)


def test_unsafe_path_is_an_asset_error() -> None:
    assert issubclass(UnsafePathError, AssetError)
