"""End-to-end MMD-style render demo.

Loads the bundled character through the full MMD-fluence stack —
toon shading + outline + checkered ground + projected shadow +
PCF self-shadow + gradient sky + sRGB + AutoLuminous bloom +
``mmd_tone`` colour grade + HighDef multi-light + DQS skinning — and
saves the hero frame as a PNG. Useful as:

- **Documentation** — every renderer toggle this engine exposes is
  set here in one place. Read this file to learn the API surface.
- **Visual smoke check** — if a refactor breaks any pass, the saved
  PNG either fails to write or comes out wrong. Drop a baseline copy
  into ``tests/golden/`` and you have a CI image regression.
- **Headless screenshot** — no Qt window is shown; the script
  renders into an offscreen FBO and writes to disk, so it runs
  cleanly on CI runners without a display.

Usage::

    py examples/mmd_demo.py [--output PATH] [--width N --height N]

Default: saves ``mmd_demo.png`` (768×768) next to this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — the importer plugins live under ``importers/`` outside the
# main package, mirroring what ``posecascade.assets.importer_manager`` does
# at runtime. The editor's ``run_app`` does this for free; standalone
# scripts have to add the path themselves.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "importers"))


def _make_offscreen_gl_context(width: int, height: int) -> tuple[object, object, int]:
    """Boot a Qt offscreen surface + Core-Profile GL 4.1 context.

    Returns ``(surface, context, fbo)``. The caller MUST keep both the
    surface and the context alive for the duration of any GL work —
    Qt's ``makeCurrent`` ties the context to a specific live surface,
    and Python's GC reaping the surface mid-render is what surfaces as
    ``glCreateShader returned NULL``. Skip cleanly with ``RuntimeError``
    if the platform can't provide an offscreen context — headless
    servers without GL libraries will hit this.
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
        raise RuntimeError("offscreen FBO incomplete — driver may not support core 4.1 + depth24")
    return surface, context, fbo


def _build_mmd_renderer(shaders_root: Path) -> object:
    """Instantiate a :class:`Renderer` with every MMD-fluence toggle set.

    Each setter call is independently optional in real use; the
    grouping here is the "give me the full MMD aesthetic" preset.
    """
    from posecascade.render.renderer import Renderer  # noqa: PLC0415

    renderer = Renderer(shaders_root=shaders_root)
    renderer.initialize()
    # Force every mesh through the toon pipeline, including the glTF
    # character that doesn't ship an MMDMaterial natively. Without this
    # the character renders with smooth Lambert — not MMD.
    renderer.set_force_toon_shading(True)
    # Drop in the back-rim + front-fill HighDef preset so the silhouette
    # reads on both sides of the gradient sky's horizon line.
    renderer.apply_highdef_light_preset()
    # The four pass toggles (ground, projected shadow, self-shadow, sky)
    # all default to ON, so explicit calls aren't required here — listed
    # for clarity / as a setter inventory.
    renderer.set_ground_enabled(True)
    renderer.set_projected_shadow_enabled(True)
    renderer.set_self_shadow_enabled(True)
    renderer.set_sky_enabled(True)
    renderer.set_srgb_output_enabled(True)
    # DQS preserves joint volume at extreme rotations. Off by default
    # for backwards compat; on here to demonstrate the API and avoid
    # candy-wrapper if the camera ever frames a heavy twist.
    renderer.set_dqs_enabled(True)
    return renderer


def _build_default_effect_chain() -> object:
    """Build the canonical MMD post chain: AutoLuminous → mmd_tone.

    The :class:`AppController` constructor auto-seeds AutoLuminous; the
    standalone Renderer this script uses doesn't go through the
    controller, so we build the chain here too.
    """
    from posecascade.render.effects.builtins import load_builtin  # noqa: PLC0415
    from posecascade.render.effects.chain import EffectChain  # noqa: PLC0415

    chain = EffectChain()
    chain.append(load_builtin("autoluminous"))
    chain.append(load_builtin("mmd_tone"))
    return chain


def _render_and_save(
    renderer: object,
    scene: object,
    camera: object,
    width: int,
    height: int,
    output: Path,
) -> None:
    """Draw a single frame and save it as PNG.

    Pure ``Renderer.draw`` invocation — no animation, no per-frame
    update. The script's value is showing the API at rest; replace
    this with a loop driving a :class:`DeclarativeRuntime` to record
    an image sequence.
    """
    from OpenGL.GL import (  # noqa: PLC0415
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        glReadPixels,
    )
    from PIL import Image  # noqa: PLC0415

    renderer.draw(scene.scene, camera, (width, height))
    raw = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    pixels = np.flipud(pixels)
    Image.fromarray(pixels, "RGBA").save(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mmd_demo")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "mmd_demo.png",
        help="Path to write the rendered PNG (default: examples/mmd_demo.png).",
    )
    parser.add_argument(
        "--width", type=int, default=768,
        help="Frame width in pixels (default: 768).",
    )
    parser.add_argument(
        "--height", type=int, default=768,
        help="Frame height in pixels (default: 768).",
    )
    parser.add_argument(
        "--model", type=Path,
        default=_REPO_ROOT / "examples" / "assets" / "herta" / "herta.glb",
        help="Path to the model to render (any importer-supported format).",
    )
    args = parser.parse_args(argv)

    if not args.model.is_file():
        sys.stderr.write(f"model not found: {args.model}\n")
        return 1

    # Imports deferred until after the path bootstrap so the importer
    # plugins (which live under ``importers/<format>/``) can resolve.
    from posecascade.assets.importer_manager import ImporterManager  # noqa: PLC0415
    from posecascade.render.camera import Camera  # noqa: PLC0415
    from posecascade.utils.math3d import vec3  # noqa: PLC0415

    print(f"booting GL context @ {args.width}x{args.height} …")
    # Hold references to surface + context for the whole script lifetime —
    # losing either reaps the GL state we're about to draw into.
    _surface, _context, _fbo = _make_offscreen_gl_context(args.width, args.height)

    print(f"loading model: {args.model.relative_to(_REPO_ROOT)}")
    manager = ImporterManager(importers_root=_REPO_ROOT / "importers")
    manager.discover()
    scene = manager.load(args.model)

    print("building renderer with the full MMD preset …")
    renderer = _build_mmd_renderer(_REPO_ROOT / "shaders")
    renderer.populate_from_scene(scene)

    # The effect chain is built but not applied here — that lives on
    # the AppController's update tick in the editor. Demonstrating the
    # construction is enough for documentation purposes.
    effect_chain = _build_default_effect_chain()
    print(f"effect chain ready: {[e.descriptor.name for e in effect_chain]}")

    # Camera framed for a 3/4 view of the bundled VRoid character. Tweak
    # if you swap the model — the target Y sits around the character's
    # solar plexus.
    camera = Camera(
        position=vec3(0.6, 0.9, 1.8),
        target=vec3(0.0, 0.55, 0.0),
    )

    print(f"rendering → {args.output} …")
    _render_and_save(renderer, scene, camera, args.width, args.height, args.output)
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
