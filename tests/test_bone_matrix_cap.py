"""Regression test: catch skins whose joint count exceeds the renderer cap.

VRoid-style anime characters often ship 130+ joints (skirt, hair, breast,
collider chains). Before this guard the renderer silently truncated the
joint list; any vertex weighted to a clamped joint sampled garbage out of
the GLSL ``u_boneMatrices`` array, which surfaced as a single limb
"stretched" in the viewport — extremely hard to bisect.

This test pins the warning behaviour so a future regression that bumps the
cap or refactors the function still loudly tells the user when a rig
won't fit.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from posecascade.assets.types import Skin
from posecascade.render.renderer import _MAX_BONES, _compute_bone_matrices
from posecascade.scene.node import Node


def _make_skin(joint_count: int, *, name: str = "rig") -> Skin:
    joints = tuple(Node(name=f"j{i}") for i in range(joint_count))
    inv_bind = np.tile(np.eye(4, dtype=np.float32), (joint_count, 1, 1))
    return Skin(name=name, joints=joints, inverse_bind_matrices=inv_bind)


def test_compute_bone_matrices_under_cap_returns_full_count() -> None:
    skin = _make_skin(_MAX_BONES - 4)
    matrices = _compute_bone_matrices(skin)
    assert matrices.shape == (_MAX_BONES - 4, 4, 4)


def test_compute_bone_matrices_at_cap_returns_full_count() -> None:
    skin = _make_skin(_MAX_BONES)
    matrices = _compute_bone_matrices(skin)
    assert matrices.shape == (_MAX_BONES, 4, 4)


def test_compute_bone_matrices_over_cap_truncates_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Galaxia-style 135-joint rig: matrix array MUST cap at _MAX_BONES, and
    the renderer MUST emit a warning so a deformed limb is not a silent
    mystery — the warning text names the cap and tells the user how to lift it."""
    over = _MAX_BONES + 7
    skin = _make_skin(over, name="oversize_rig")
    with caplog.at_level(logging.WARNING, logger="posecascade.render.renderer"):
        matrices = _compute_bone_matrices(skin)
    assert matrices.shape == (_MAX_BONES, 4, 4)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "oversize_rig" in r.getMessage() and str(over) in r.getMessage()
        for r in warnings
    ), f"expected warning naming the rig and joint count, got: {[r.getMessage() for r in warnings]}"


def test_compute_bone_matrices_warns_only_once_per_skin(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Per-frame log spam would drown the journal — warn once per skin."""
    skin = _make_skin(_MAX_BONES + 3, name="loud_rig")
    with caplog.at_level(logging.WARNING, logger="posecascade.render.renderer"):
        for _ in range(5):
            _compute_bone_matrices(skin)
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "loud_rig" in r.getMessage()
    ]
    assert len(warnings) == 1, f"expected 1 warning, got {len(warnings)}"
