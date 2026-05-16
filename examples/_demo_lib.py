"""Shared helpers for the standalone demo scripts under ``examples/``.

Each ``compare_*.py`` script needs the same offscreen GL boot,
character load, side-by-side composition, and "MMD preset" renderer
configuration. Keeping the boilerplate here lets the comparison
scripts focus on the *one* knob each is demonstrating.

Module-private (underscore-prefixed) so it doesn't pretend to be a
public API of the engine — these helpers exist for the demos only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "importers") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "importers"))

# Public-facing colour / scale constants the comparison scripts share.
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 800
LABEL_BAND_HEIGHT = 36
GAP_BETWEEN_PANES = 6
LABEL_BG = (32, 36, 42, 255)
LABEL_FG = (242, 242, 240, 255)


def setup_offscreen_gl(
    width: int, height: int,
) -> tuple[object, object, int, int]:
    """Boot Qt + GL 4.1 Core + an RGBA8/depth24 FBO.

    Returns ``(surface, context, fbo, color_texture_id)``. The caller
    MUST hold the surface AND context references for the lifetime of
    GL work — Qt ties ``makeCurrent`` to a live surface.
    """
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
        glFramebufferRenderbuffer,
        glFramebufferTexture2D,
        glGenFramebuffers,
        glGenRenderbuffers,
        glGenTextures,
        glRenderbufferStorage,
        glTexImage2D,
        glTexParameteri,
    )
    from PySide6.QtGui import (  # noqa: PLC0415
        QOffscreenSurface,
        QOpenGLContext,
        QSurfaceFormat,
    )
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    QApplication.instance() or QApplication([])
    fmt = QSurfaceFormat()
    # 4.3 is required for compute shaders + SSBOs, which the passive
    # skin-deform path uses to GPU-accelerate LBS for 30 k+ vert meshes.
    # Drivers usually grant the highest GL version they support — on
    # Windows / Linux that's 4.6; macOS legacy GL caps out at 4.1 and
    # will fall through to the CPU LBS path.
    fmt.setVersion(4, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    context = QOpenGLContext()
    context.setFormat(fmt)
    if not context.create() or not context.makeCurrent(surface):
        raise RuntimeError("could not create an offscreen GL context")

    fbo = int(glGenFramebuffers(1))
    color = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, color)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, None,
    )
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    depth = int(glGenRenderbuffers(1))
    glBindRenderbuffer(GL_RENDERBUFFER, depth)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth)
    if int(glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("offscreen FBO incomplete on this driver")
    return surface, context, fbo, color


def load_character() -> object:
    """Import the bundled demo character (The Herta glTF) once per process.

    CC-BY 4.0 via X9_YT on Sketchfab; character © HoYoverse, used under
    their Fan Content Guidelines — see ``examples/assets/herta/NOTICE.md``.
    """
    from posecascade.assets.importer_manager import ImporterManager  # noqa: PLC0415

    manager = ImporterManager(importers_root=REPO_ROOT / "importers")
    manager.discover()
    herta = REPO_ROOT / "examples" / "assets" / "herta" / "herta.glb"
    return manager.load(herta)


def make_renderer(*, force_toon: bool = True, highdef_lights: bool = True) -> object:
    """Renderer with the MMD-preset toggles applied; comparison scripts override
    individual settings before drawing."""
    from posecascade.render.renderer import Renderer  # noqa: PLC0415

    renderer = Renderer(shaders_root=REPO_ROOT / "shaders")
    renderer.initialize()
    if force_toon:
        renderer.set_force_toon_shading(True)
    if highdef_lights:
        renderer.apply_highdef_light_preset()
    return renderer


def read_pixels(width: int, height: int) -> np.ndarray:
    """Glreadback the colour attachment as ``(H, W, 4)`` uint8 (top-left origin)."""
    from OpenGL.GL import GL_RGBA, GL_UNSIGNED_BYTE, glReadPixels  # noqa: PLC0415

    raw = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    return np.flipud(pixels).copy()


def save_side_by_side(
    left: np.ndarray, right: np.ndarray,
    label_left: str, label_right: str,
    output: Path,
) -> None:
    """Compose two RGBA frames horizontally with labelled strips on top.

    The labels read against ``LABEL_BG`` so each pane's title sits in
    a dark band that doesn't compete with the scene's gradient sky.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    if left.shape != right.shape:
        raise ValueError(
            f"left {left.shape} and right {right.shape} must match for side-by-side",
        )
    h, w, _ = left.shape
    canvas_w = 2 * w + GAP_BETWEEN_PANES
    canvas_h = h + LABEL_BAND_HEIGHT
    canvas = Image.new("RGBA", (canvas_w, canvas_h), LABEL_BG)
    canvas.paste(Image.fromarray(left, "RGBA"), (0, LABEL_BAND_HEIGHT))
    canvas.paste(Image.fromarray(right, "RGBA"), (w + GAP_BETWEEN_PANES, LABEL_BAND_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = _load_label_font()
    _label_pane(draw, label_left, 0, w, font)
    _label_pane(draw, label_right, w + GAP_BETWEEN_PANES, w, font)
    canvas.save(output)


def _load_label_font() -> object:
    """Return a Pillow font; falls back to the default bitmap font if TTF lookup fails."""
    from PIL import ImageFont  # noqa: PLC0415

    for candidate in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(candidate, 20)
        except OSError:
            continue
    return ImageFont.load_default()


def _label_pane(draw: object, text: str, x: int, w: int, font: object) -> None:
    """Draw ``text`` centred horizontally inside the label band of pane ``x..x+w``."""
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    px = x + (w - text_w) // 2
    py = (LABEL_BAND_HEIGHT - text_h) // 2
    draw.text((px, py), text, fill=LABEL_FG, font=font)
