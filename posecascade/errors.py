"""Domain exception hierarchy.

All engine-raised exceptions inherit from :class:`PoseCascadeError`. Catch the
narrowest type that conveys intent; never catch :class:`Exception` directly.
"""
from __future__ import annotations


class PoseCascadeError(Exception):
    """Base class for all engine-raised exceptions."""


class AssetError(PoseCascadeError):
    """Asset loading or resolution failed."""


class MalformedAssetError(AssetError):
    """The asset file did not match the format's spec."""


class UnsupportedFormatError(AssetError):
    """No importer is registered for the file extension."""


class UnsafePathError(AssetError):
    """An asset reference attempted to escape the project root."""


class GLError(PoseCascadeError):
    """An OpenGL call failed or the context is invalid."""


class SceneError(PoseCascadeError):
    """Scene-graph operation failed (missing node, cyclic parent, etc.)."""


class ScriptError(PoseCascadeError):
    """User script failed to load or run."""


class ScriptSecurityError(ScriptError):
    """A script tried to use a forbidden identifier or builtin."""


class ScriptRuntimeError(ScriptError):
    """A user-script callback raised at runtime."""
