"""Discover importer plugins under ``importers/`` and dispatch by extension.

Each importer plugin is a folder under the project's ``importers/`` directory
whose ``__init__.py`` sets a module-level ``importer_class`` referring to a
class with:

- a ``supported_extensions: tuple[str, ...]`` class attribute, and
- a ``load(self, path: pathlib.Path) -> ImportedScene`` method.

The manager prepends ``importers/`` to ``sys.path`` so each plugin folder is
importable as a top-level package.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from posecascade.assets.auto_rig import apply_post_import_rig
from posecascade.assets.types import ImportedScene
from posecascade.errors import UnsupportedFormatError


class Importer(Protocol):
    supported_extensions: tuple[str, ...]

    def load(self, path: Path) -> ImportedScene: ...


@dataclass
class ImporterManager:
    """Registry mapping file extensions to importer instances."""

    importers_root: Path
    _by_extension: dict[str, Importer] = field(default_factory=dict)

    def discover(self) -> None:
        """Walk ``importers_root`` and register every plugin that loads cleanly."""
        if not self.importers_root.is_dir():
            return
        if str(self.importers_root) not in sys.path:
            sys.path.insert(0, str(self.importers_root))
        for child in sorted(self.importers_root.iterdir()):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            self._register_plugin(child.name)

    def _register_plugin(self, plugin_name: str) -> None:
        try:
            module = importlib.import_module(plugin_name)
        except ImportError:
            return
        importer_cls = getattr(module, "importer_class", None)
        if importer_cls is None:
            return
        instance: Importer = importer_cls()
        for ext in instance.supported_extensions:
            self._by_extension[ext.lower()] = instance

    def importer_for(self, path: Path) -> Importer:
        """Return the importer registered for ``path.suffix``; raise if none."""
        ext = path.suffix.lower()
        importer = self._by_extension.get(ext)
        if importer is None:
            raise UnsupportedFormatError(f"no importer registered for extension {ext!r}")
        return importer

    def load(self, path: Path) -> ImportedScene:
        """Look up the right importer, load the scene, run universal auto-rigs.

        Every format goes through the same post-import pipeline (see
        :mod:`posecascade.assets.auto_rig`): spring chains tagged by bone
        name, etc. The glTF importer historically did this in-place;
        moving it here means PMD / PMX / FBX / OBJ imports get the same
        treatment without duplicating the logic in each plugin. The
        pass is idempotent so the glTF importer's in-house call (still
        present for backwards-compat) does no extra work.
        """
        imported = self.importer_for(path).load(path)
        apply_post_import_rig(imported)
        return imported
