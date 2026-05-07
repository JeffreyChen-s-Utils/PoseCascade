"""PMD importer adapter (legacy MikuMikuDance Polygon Model Data).

PMD parses to the same :class:`~pmx.types.PmxDocument` schema as PMX, so
this adapter delegates the document → :class:`~posecascade.assets.types.ImportedScene`
mapping to the PMX adapter and only owns the file-format-specific bits
(magic byte sequence, ``.pmd`` extension, file-size validation).
"""
from __future__ import annotations

from pathlib import Path

from pmx.importer import _build_imported_scene

from pmd.reader import parse_pmd
from posecascade.assets.types import ImportedScene
from posecascade.errors import MalformedAssetError


class PmdImporter:
    """Loads ``.pmd`` files into :class:`ImportedScene`."""

    supported_extensions: tuple[str, ...] = (".pmd",)

    def load(self, path: Path) -> ImportedScene:
        path = path.resolve()
        if not path.is_file():
            raise MalformedAssetError(f"PMD file not found: {path}")
        return _build_imported_scene(parse_pmd(path.read_bytes()), path.parent)
