"""Phase 3 visual smoke: render a PMX cube animated by VMD bone keyframes.

We:

1. Load the canonical PMX cube + the wave VMD.
2. Build a player against the cube's skin.
3. Render two frames at very different times (frame 0 = identity, frame 5
   = quarter-turn around X) and assert the readback luminance maps differ
   by more than the noise floor — proves the VMD actually drove the bones
   and the renderer picked up the new TRS.
"""
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
_TINY_PMX = Path(__file__).resolve().parent / "fixtures" / "mmd" / "tiny.pmx"
_WAVE_VMD = Path(__file__).resolve().parent / "fixtures" / "mmd" / "wave.vmd"


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


def test_vmd_drives_bones_and_changes_render(
    gl_context: object, offscreen_fbo: int,
) -> None:
    imported = PmxImporter().load(_TINY_PMX)
    motion = VmdImporter().load(_WAVE_VMD)
    player = VmdAnimationPlayer.for_skin(motion, imported.skins[0])

    renderer = Renderer(shaders_root=_shaders_root())
    renderer.initialize()
    renderer.populate_from_scene(imported)

    camera = Camera(position=vec3(3.0, 2.0, 3.0), target=vec3(0.0, 0.0, 0.0))

    player.apply(0.0)
    renderer.draw(imported.scene, camera, VIEWPORT_SIZE)
    rest_pixels = _read_color_pixels()

    player.apply(5.0 / VMD_FRAMES_PER_SECOND)   # quarter-turn around X
    renderer.draw(imported.scene, camera, VIEWPORT_SIZE)
    moved_pixels = _read_color_pixels()

    diff = np.abs(rest_pixels.astype(np.int32) - moved_pixels.astype(np.int32))
    changed_fraction = float((diff.max(axis=-1) > 10).mean())
    assert changed_fraction > 0.05, (
        "VMD bone drive produced near-identical pixels at frame 0 vs frame 5 "
        f"(only {changed_fraction:.3%} of pixels differ — bone TRS not flowing)"
    )
