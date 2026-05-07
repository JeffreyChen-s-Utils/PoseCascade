"""PoseCascade — PySide6 + OpenGL engine for importing 3D models and scripted animation.

Importing :mod:`posecascade` prepends the project's ``importers/``
directory to ``sys.path`` so engine modules that pull format-internal
helpers (e.g. ``from vmd.types import ...`` inside the animation
layer) resolve without forcing every entry point to call
:meth:`ImporterManager.discover` first. This mirrors the path-injection
that :class:`~posecascade.assets.importer_manager.ImporterManager`
performs at runtime + the matching block in :mod:`tests.conftest`.
"""
from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.0.1"
__all__ = ["__version__"]

_IMPORTERS_ROOT = Path(__file__).resolve().parent.parent / "importers"
if _IMPORTERS_ROOT.is_dir() and str(_IMPORTERS_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORTERS_ROOT))
