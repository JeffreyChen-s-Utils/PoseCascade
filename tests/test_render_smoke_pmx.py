"""Phase 1 visual smoke: render the canonical PMX fixture to an offscreen FBO.

Loads ``tests/fixtures/mmd/tiny.pmx`` through the real PMX importer +
:class:`Renderer`, draws one frame at 256×256, and asserts the output is
non-trivial (alpha = 1 everywhere, more than a small fraction of the pixels
look like geometry rather than clear-colour). Skips cleanly on CI runners
that cannot create an offscreen GL context.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pmx.importer import PmxImporter

from posecascade.render.camera import Camera
from posecascade.render.renderer import Renderer
from posecascade.utils.math3d import vec3

VIEWPORT_SIZE = (256, 256)
_TINY_PMX = Path(__file__).resolve().parent / "fixtures" / "mmd" / "tiny.pmx"


def _shaders_root() -> Path:
    return Path(__file__).resolve().parent.parent / "shaders"


@pytest.fixture
def offscreen_fbo(gl_context: object) -> int:
    """Single-test FBO; mirrors the layout used by :mod:`test_render_smoke`."""
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


def test_renders_pmx_tiny(gl_context: object, offscreen_fbo: int) -> None:
    """End-to-end Phase 1 smoke: PMX → ImportedScene → Renderer → FBO pixels."""
    imported = PmxImporter().load(_TINY_PMX)
    renderer = Renderer(shaders_root=_shaders_root())
    renderer.initialize()
    renderer.populate_from_scene(imported)

    camera = Camera(position=vec3(3.0, 2.0, 3.0), target=vec3(0.0, 0.0, 0.0))
    renderer.draw(imported.scene, camera, VIEWPORT_SIZE)

    pixels = _read_color_pixels()
    alpha = pixels[..., 3]
    assert int(alpha.min()) == 255, "alpha not 1.0 — readback hit the wrong FBO"
    rgb = pixels[..., :3].astype(np.float32)
    luminance = rgb.mean(axis=-1)
    geometry_pixels = float((luminance > 60.0).mean())
    assert geometry_pixels > 0.05, (
        f"only {geometry_pixels:.3%} of pixels look like geometry — PMX render failed"
    )
