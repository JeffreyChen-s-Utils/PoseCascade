"""COLLADA (.dae) importer — XML parsed via defusedxml."""
from __future__ import annotations

from pathlib import Path

from posecascade.assets.types import ImportedScene


class ColladaImporter:
    """Loads ``.dae`` files into :class:`ImportedScene`."""

    supported_extensions: tuple[str, ...] = (".dae",)

    def load(self, path: Path) -> ImportedScene:
        raise NotImplementedError("COLLADA importer pending; use defusedxml.ElementTree")
