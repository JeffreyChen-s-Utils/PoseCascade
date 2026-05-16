"""Offscreen render smoke test.

Renders a tiny cube to an FBO via the real :class:`~posecascade.render.renderer.Renderer`
and checks the output is non-trivial and roughly matches the stored baseline.

The first run saves ``tests/golden/cube.png`` and skips the comparison; later
runs assert the rendered pixels stay within an MSE tolerance. Delete the file
to refresh the baseline.

Skips cleanly when the offscreen GL context cannot be created (``gl_context``
fixture handles that).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from posecascade.assets.types import Mesh
from posecascade.render.camera import Camera
from posecascade.render.renderer import Renderer
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.utils.math3d import quat_identity, vec3

VIEWPORT_SIZE = (256, 256)
GOLDEN_PATH = Path(__file__).parent / "golden" / "cube.png"
MSE_TOLERANCE = 5.0  # mean squared error per channel; pixels are 0–255


def _cube_mesh() -> Mesh:
    positions = np.array(
        [
            # +X
            [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5],
            # -X
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5],
            # +Y
            [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5],
            # -Y
            [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5],
            # +Z
            [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5],
            # -Z
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
        ],
        dtype=np.float32,
    )
    normals = np.array(
        [
            [1, 0, 0]] * 4 + [[-1, 0, 0]] * 4
            + [[0, 1, 0]] * 4 + [[0, -1, 0]] * 4
            + [[0, 0, 1]] * 4 + [[0, 0, -1]] * 4,
        dtype=np.float32,
    )
    indices: list[int] = []
    for face in range(6):
        base = face * 4
        indices.extend((base + 0, base + 1, base + 2, base + 0, base + 2, base + 3))
    return Mesh(
        name="cube",
        positions=positions,
        indices=np.asarray(indices, dtype=np.uint32),
        normals=normals,
    )


def _shaders_root() -> Path:
    return Path(__file__).resolve().parent.parent / "shaders"


@pytest.fixture
def offscreen_fbo(gl_context: object) -> int:
    """Create a 256x256 RGBA FBO with a depth renderbuffer; bound on entry, deleted on exit."""
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
    # Bind to GL_FRAMEBUFFER so both DRAW and READ targets point at our FBO —
    # otherwise glReadPixels reads from the default framebuffer, not our texture.
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


def test_renders_cube(gl_context: object, offscreen_fbo: int) -> None:
    pillow = pytest.importorskip("PIL.Image")
    renderer = Renderer(shaders_root=_shaders_root())
    renderer.initialize()
    # Cube test pins the cube-vs-background pixel ratio; every new
    # render pass (ground, projected shadow, self-shadow, sky,
    # sRGB-aware output) would change the baseline. All default-on for
    # the editor — the smoke test opts out so it keeps measuring what
    # it always did.
    renderer.set_ground_enabled(False)
    renderer.set_self_shadow_enabled(False)
    renderer.set_sky_enabled(False)
    renderer.set_srgb_output_enabled(False)

    cube = _cube_mesh()
    scene = Scene(name="cube_scene")
    node = Node(name="cube")
    node.transform.set_translation(vec3(0.0, 0.0, 0.0))
    node.transform.set_rotation(quat_identity())
    scene.root.add_child(node)
    renderer.attach_mesh(node, cube)

    camera = Camera(position=vec3(2.0, 1.5, 2.0), target=vec3(0.0, 0.0, 0.0))
    renderer.draw(scene, camera, VIEWPORT_SIZE)

    pixels = _read_color_pixels()
    # The clear color is dark grey (~20, 23, 26). The cube uses a light base
    # colour (~204) modulated by lambert ≥ 0.1, so cube pixels are well above
    # 25 in luminance and the alpha channel is 1.0 everywhere.
    rgb = pixels[..., :3].astype(np.float32)
    alpha = pixels[..., 3]
    assert int(alpha.min()) == 255, "alpha channel not 1.0 — readback hit the wrong FBO"
    luminance = rgb.mean(axis=-1)
    cube_pixel_fraction = float((luminance > 60.0).mean())
    assert cube_pixel_fraction > 0.05, (
        f"only {cube_pixel_fraction:.3%} of pixels look like the cube — render failed"
    )

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not GOLDEN_PATH.exists():
        pillow.fromarray(pixels, mode="RGBA").save(GOLDEN_PATH)
        pytest.skip(f"baseline written to {GOLDEN_PATH}; rerun to compare")
    baseline = np.asarray(pillow.open(GOLDEN_PATH).convert("RGBA"))
    if baseline.shape != pixels.shape:
        pytest.skip(f"baseline {GOLDEN_PATH} has different shape; delete to refresh")
    mse = float(np.mean((pixels.astype(np.float32) - baseline.astype(np.float32)) ** 2))
    assert mse <= MSE_TOLERANCE, f"render diverged from baseline (MSE={mse:.3f})"
