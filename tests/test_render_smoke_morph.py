"""Phase 4 visual smoke: vertex / material morphs visibly change the render."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pmx.importer import PmxImporter
from vmd.importer import VmdImporter

from posecascade.animation.player import VmdAnimationPlayer
from posecascade.animation.vmd_track import VMD_FRAMES_PER_SECOND
from posecascade.render.camera import Camera
from posecascade.render.renderer import Renderer
from posecascade.utils.math3d import vec3

VIEWPORT_SIZE = (256, 256)
_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "mmd"


def _shaders_root() -> Path:
    return Path(__file__).resolve().parent.parent / "shaders"


@pytest.fixture
def offscreen_fbo(gl_context: object) -> int:
    from OpenGL.GL import (  # noqa: PLC0415
        GL_COLOR_ATTACHMENT0,
        GL_DEPTH_ATTACHMENT,
        GL_DEPTH_COMPONENT24,
        GL_FRAMEBUFFER,
        GL_FRAMEBUFFER_COMPLETE,
        GL_LINEAR,
        GL_RENDERBUFFER,
        GL_RGBA,
        GL_RGBA8,
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_UNSIGNED_BYTE,
        glBindFramebuffer,
        glBindRenderbuffer,
        glBindTexture,
        glCheckFramebufferStatus,
        glDeleteFramebuffers,
        glDeleteRenderbuffers,
        glDeleteTextures,
        glFramebufferRenderbuffer,
        glFramebufferTexture2D,
        glGenFramebuffers,
        glGenRenderbuffers,
        glGenTextures,
        glRenderbufferStorage,
        glTexImage2D,
        glTexParameteri,
    )
    width, height = VIEWPORT_SIZE
    fbo = int(glGenFramebuffers(1))
    color_tex = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, color_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    depth_rb = int(glGenRenderbuffers(1))
    glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb)
    if int(glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE:
        glDeleteFramebuffers(1, [fbo])
        glDeleteTextures(1, [color_tex])
        glDeleteRenderbuffers(1, [depth_rb])
        pytest.skip("offscreen FBO incomplete on this driver")
    try:
        yield fbo
    finally:
        glDeleteFramebuffers(1, [fbo])
        glDeleteTextures(1, [color_tex])
        glDeleteRenderbuffers(1, [depth_rb])


def _read_color_pixels() -> np.ndarray:
    from OpenGL.GL import (  # noqa: PLC0415
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        glReadPixels,
    )
    width, height = VIEWPORT_SIZE
    raw = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    return np.flipud(pixels).copy()


def test_morph_changes_render(gl_context: object, offscreen_fbo: int) -> None:
    """Frame 0 (rest) vs frame 10 (peak morph weight) must differ.

    The ``morphs.vmd`` track ramps the ``all`` group morph from 0 → 1 → 0
    over 20 frames. ``all`` recursively pulls the vertex + bone child
    morphs; either path is enough to push significant pixel diff between
    rest and peak.
    """
    imported = PmxImporter().load(_FIXTURE_DIR / "tiny_morphs.pmx")
    motion = VmdImporter().load(_FIXTURE_DIR / "morphs.vmd")
    renderer = Renderer(shaders_root=_shaders_root())
    renderer.initialize()
    renderer.populate_from_scene(imported)
    player = VmdAnimationPlayer.for_imported_scene(motion, imported, renderer=renderer)

    camera = Camera(position=vec3(3.0, 2.0, 3.0), target=vec3(0.0, 0.0, 0.0))

    player.apply(0.0)
    renderer.draw(imported.scene, camera, VIEWPORT_SIZE)
    rest_pixels = _read_color_pixels()

    player.apply(10.0 / VMD_FRAMES_PER_SECOND)
    renderer.draw(imported.scene, camera, VIEWPORT_SIZE)
    morphed_pixels = _read_color_pixels()

    diff = np.abs(rest_pixels.astype(np.int32) - morphed_pixels.astype(np.int32))
    changed_fraction = float((diff.max(axis=-1) > 10).mean())
    # Threshold tuned for the synthetic 8-vertex cube fixture: a single
    # vertex morph shifts two corners and a bone morph tilts the child by
    # 30°; together they perturb a small fraction of the silhouette. A
    # real model rebinds the same code path against thousands of vertices
    # — this assertion is just a "wired-up" smoke test, not a fidelity gate.
    assert changed_fraction > 0.03, (
        "morph drive produced near-identical pixels at frame 0 vs frame 10 "
        f"(only {changed_fraction:.3%} of pixels differ — morph state not flowing)"
    )
