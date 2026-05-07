"""FBX importer (Autodesk FBX SDK or pyfbx)."""
from __future__ import annotations

from pathlib import Path

from posecascade.assets.types import ImportedScene


class FbxImporter:
    """Loads ``.fbx`` files into :class:`ImportedScene`."""

    supported_extensions: tuple[str, ...] = (".fbx",)

    def load(self, path: Path) -> ImportedScene:
        raise NotImplementedError("FBX importer pending; install pyfbx or the FBX SDK")
