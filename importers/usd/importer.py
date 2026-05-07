"""USD / USDZ importer."""
from __future__ import annotations

from pathlib import Path

from posecascade.assets.types import ImportedScene


class UsdImporter:
    """Loads ``.usd``, ``.usda``, ``.usdc``, ``.usdz`` into :class:`ImportedScene`."""

    supported_extensions: tuple[str, ...] = (".usd", ".usda", ".usdc", ".usdz")

    def load(self, path: Path) -> ImportedScene:
        raise NotImplementedError("USD importer pending; install pyusd")
