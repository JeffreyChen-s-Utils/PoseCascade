"""VPD importer adapter."""
from __future__ import annotations

from pathlib import Path

from posecascade.errors import MalformedAssetError
from vpd.reader import parse_vpd_bytes
from vpd.types import VpdPose


class VpdImporter:
    """Loads ``.vpd`` files into :class:`VpdPose`."""

    supported_extensions: tuple[str, ...] = (".vpd",)

    def load(self, path: Path) -> VpdPose:
        path = path.resolve()
        if not path.is_file():
            raise MalformedAssetError(f"VPD file not found: {path}")
        return parse_vpd_bytes(path.read_bytes())
