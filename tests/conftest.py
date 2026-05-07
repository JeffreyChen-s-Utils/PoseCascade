"""Shared pytest fixtures.

Mirrors the ``importers/`` path injection that
:class:`posecascade.assets.importer_manager.ImporterManager` performs at
runtime, so tests can ``from <format> import …`` directly.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMPORTERS_ROOT = PROJECT_ROOT / "importers"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(IMPORTERS_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORTERS_ROOT))


@pytest.fixture(autouse=True)
def _isolate_user_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect any user-settings path used by the engine into a per-test tmpdir.

    The engine does not yet read settings from disk, so this is a no-op for the
    skeleton. It exists so future settings code can rely on the redirect being
    universally applied.
    """
    monkeypatch.setenv("POSECASCADE_USER_DIR", str(tmp_path))


@pytest.fixture
def qapp() -> Iterator[object]:
    """Provide a singleton ``QApplication`` for tests that touch Qt widgets.

    Uses the platform's native plugin so OpenGL contexts work; tests must use
    :class:`QOffscreenSurface` rather than visible windows to stay headless.
    The Qt 'offscreen' platform plugin does not provide a working GL backend
    on Windows, so we deliberately do not set ``QT_QPA_PLATFORM=offscreen``.
    """
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415 — lazy Qt import

    app = QApplication.instance() or QApplication([])
    _ = os  # keep the import for tests that opt in to env tweaks
    yield app


@pytest.fixture
def gl_context(qapp: object) -> Iterator[object]:
    """Offscreen GL context for renderer tests; skips if creation fails."""
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat  # noqa: PLC0415

    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    if not surface.isValid():
        pytest.skip("offscreen surface unavailable on this CI runner")
    context = QOpenGLContext()
    context.setFormat(fmt)
    if not context.create() or not context.makeCurrent(surface):
        pytest.skip("could not create or make-current an offscreen GL context")
    try:
        yield context
    finally:
        context.doneCurrent()


@pytest.fixture
def sample_gltf_cube() -> dict[str, np.ndarray]:
    """A minimal cube as numpy arrays — what a glTF importer would produce."""
    positions = np.array(
        [
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            0, 1, 2, 0, 2, 3,
            4, 6, 5, 4, 7, 6,
            0, 4, 5, 0, 5, 1,
            2, 6, 7, 2, 7, 3,
            1, 5, 6, 1, 6, 2,
            0, 3, 7, 0, 7, 4,
        ],
        dtype=np.uint32,
    )
    return {"positions": positions, "indices": indices}


@pytest.fixture
def sample_skinned_mesh() -> dict[str, np.ndarray]:
    """A 2-bone, 4-vertex strip used by skinning unit tests."""
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    joints = np.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
        dtype=np.uint16,
    )
    weights = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    return {"positions": positions, "joints": joints, "weights": weights}
