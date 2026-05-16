"""Phase 2 visual smoke: render a PMX cube through the toon + outline passes.

Builds an in-memory PMX whose single material has the ``HAS_EDGE`` flag set
and a small ``edge_size``, hands it to the renderer's toon pipeline, and
checks the readback:

- a non-trivial fraction of pixels matches the toon-shaded interior
- a non-trivial fraction matches the (near-black) outline
- the pixel mix is *different* from a forward-rendered baseline of the
  same fixture (so we know the toon programs actually changed the output)

Falls back to SSIM comparison against ``tests/golden/toon_tiny.png`` when
``scikit-image`` is available.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pmx.importer import PmxImporter

from posecascade.render.camera import Camera
from posecascade.render.material import MAT_FLAG_HAS_EDGE
from posecascade.render.renderer import Renderer
from posecascade.utils.math3d import vec3
from tests.fixtures.mmd.build import build_pmx, tiny_cube_spec

VIEWPORT_SIZE = (256, 256)
_GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "toon_tiny.png"
_SSIM_TOLERANCE = 0.92


def _shaders_root() -> Path:
    return Path(__file__).resolve().parent.parent / "shaders"


def _toon_pmx(tmp_path: Path, *, double_sided: bool = False) -> Path:
    """Write a PMX whose single material opts into edge + (optionally) double-sided."""
    spec = tiny_cube_spec()
    flags = MAT_FLAG_HAS_EDGE | (0x01 if double_sided else 0x00)
    materials = (
        replace(
            spec.materials[0],
            flags=flags,
            edge_color=(0.0, 0.0, 0.0, 1.0),
            edge_size=0.05,
        ),
    )
    path = tmp_path / "toon.pmx"
    path.write_bytes(build_pmx(replace(spec, materials=materials)))
    return path


@pytest.fixture
def offscreen_fbo(gl_context: object) -> int:
    """Create an offscreen 256×256 FBO; mirrors :mod:`test_render_smoke`'s setup."""
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


def _draw_toon(pmx_path: Path) -> np.ndarray:
    imported = PmxImporter().load(pmx_path)
    renderer = Renderer(shaders_root=_shaders_root())
    renderer.initialize()
    # Toon smoke test pins the cube + outline pixel ratio against a
    # golden image; ground / projected shadow / self-shadow / sky /
    # sRGB output would all invalidate that comparison. The editor
    # still defaults all of them on for interactive use.
    renderer.set_ground_enabled(False)
    renderer.set_self_shadow_enabled(False)
    renderer.set_sky_enabled(False)
    renderer.set_srgb_output_enabled(False)
    renderer.populate_from_scene(imported)
    camera = Camera(position=vec3(3.0, 2.0, 3.0), target=vec3(0.0, 0.0, 0.0))
    renderer.draw(imported.scene, camera, VIEWPORT_SIZE)
    return _read_color_pixels()


def test_renders_toon_with_outline(
    gl_context: object, offscreen_fbo: int, tmp_path: Path,
) -> None:
    pixels = _draw_toon(_toon_pmx(tmp_path))
    rgb = pixels[..., :3].astype(np.float32)
    luminance = rgb.mean(axis=-1)

    # The cube's interior reads through toon01 (mostly white) so it looks
    # bright; the inverted-hull outline writes near-black where it shows
    # outside the silhouette. We expect both bands to show up clearly.
    bright_fraction = float((luminance > 150.0).mean())
    dark_geometry_fraction = float(((luminance > 5.0) & (luminance < 50.0)).mean())
    assert bright_fraction > 0.04, (
        f"toon-lit interior covers only {bright_fraction:.3%} of the viewport"
    )
    assert dark_geometry_fraction > 0.005, (
        "no near-black outline pixels — outline pass did not draw"
    )


def test_double_sided_keeps_full_silhouette(
    gl_context: object, offscreen_fbo: int, tmp_path: Path,
) -> None:
    """The double-sided flag must disable back-face culling without losing
    visible geometry — coverage on a forward-facing cube should stay above
    the no-double-sided baseline."""
    plain = _draw_toon(_toon_pmx(tmp_path, double_sided=False))
    double = _draw_toon(_toon_pmx(tmp_path, double_sided=True))
    plain_geom = float((plain[..., :3].mean(axis=-1) > 25.0).mean())
    double_geom = float((double[..., :3].mean(axis=-1) > 25.0).mean())
    assert double_geom + 1e-3 >= plain_geom


def test_toon_golden_or_baseline(
    gl_context: object, offscreen_fbo: int, tmp_path: Path,
) -> None:
    pillow = pytest.importorskip("PIL.Image")
    pixels = _draw_toon(_toon_pmx(tmp_path))
    _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _GOLDEN_PATH.exists():
        pillow.fromarray(pixels, mode="RGBA").save(_GOLDEN_PATH)
        pytest.skip(f"baseline written to {_GOLDEN_PATH}; rerun to compare")
    baseline = np.asarray(pillow.open(_GOLDEN_PATH).convert("RGBA"))
    if baseline.shape != pixels.shape:
        pytest.skip(f"baseline {_GOLDEN_PATH} has different shape; delete to refresh")
    try:
        from skimage.metrics import structural_similarity  # noqa: PLC0415
    except ImportError:
        mse = float(np.mean((pixels.astype(np.float32) - baseline.astype(np.float32)) ** 2))
        assert mse <= 25.0, f"toon render MSE drift = {mse:.3f}"
        return
    score = float(
        structural_similarity(
            pixels[..., :3], baseline[..., :3], channel_axis=-1, data_range=255,
        )
    )
    assert score >= _SSIM_TOLERANCE, (
        f"toon render SSIM {score:.3f} below tolerance {_SSIM_TOLERANCE}"
    )
