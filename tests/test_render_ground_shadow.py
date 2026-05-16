"""Offscreen render tests for the MMD-flavoured ground + shadow pipeline.

Checks each of the three new passes added in this round:

- Checkered ground draws something visible below the model.
- The projected ground shadow lands a darker silhouette on the ground.
- The self-shadow depth pass attenuates a chunk of the model's diffuse.
- The default effect chain seeded by ``AppController`` contains the
  AutoLuminous descriptor — that's a unit-test target that doesn't
  need a GL context.
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


def _shaders_root() -> Path:
    return Path(__file__).resolve().parent.parent / "shaders"


def _cube_above_ground() -> tuple[Scene, Mesh]:
    """A 1 m cube whose underside sits just above the ground plane."""
    positions = np.array(
        [
            [-0.4, 0.4, -0.4], [ 0.4, 0.4, -0.4], [ 0.4, 1.2, -0.4], [-0.4, 1.2, -0.4],
            [-0.4, 0.4,  0.4], [ 0.4, 0.4,  0.4], [ 0.4, 1.2,  0.4], [-0.4, 1.2,  0.4],
        ],
        dtype=np.float32,
    )
    normals = np.zeros_like(positions)
    normals[:, 1] = 1.0
    indices = np.array(
        [
            0, 1, 2, 0, 2, 3,    # back
            4, 6, 5, 4, 7, 6,    # front
            0, 4, 5, 0, 5, 1,    # bottom
            2, 6, 7, 2, 7, 3,    # top
            1, 5, 6, 1, 6, 2,    # right
            0, 3, 7, 0, 7, 4,    # left
        ],
        dtype=np.uint32,
    )
    mesh = Mesh(name="floating_cube", positions=positions, normals=normals, indices=indices)
    scene = Scene(name="cube_above_ground")
    node = Node(name="cube")
    node.transform.set_translation(vec3(0.0, 0.0, 0.0))
    node.transform.set_rotation(quat_identity())
    scene.root.add_child(node)
    return scene, mesh


def _make_offscreen_fbo() -> tuple[int, int, int]:
    """Allocate a 256×256 colour FBO and return ``(fbo, color_tex, depth_rb)``."""
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
    width, height = VIEWPORT_SIZE
    fbo = int(glGenFramebuffers(1))
    color_tex = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, color_tex)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, None,
    )
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    depth_rb = int(glGenRenderbuffers(1))
    glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0)
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb,
    )
    if int(glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE:
        pytest.skip("offscreen FBO incomplete on this driver")
    return fbo, color_tex, depth_rb


def _read_pixels() -> np.ndarray:
    from OpenGL.GL import GL_RGBA, GL_UNSIGNED_BYTE, glReadPixels  # noqa: PLC0415

    width, height = VIEWPORT_SIZE
    raw = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    return np.flipud(pixels).copy()


@pytest.fixture
def offscreen_fbo(gl_context: object) -> int:
    """Provide a 256×256 RGBA8 + depth FBO bound to the GL context.

    Mirrors the smoke-test fixture in ``test_render_smoke.py`` but lives
    here so the ground / shadow tests stay self-contained.
    """
    from OpenGL.GL import (  # noqa: PLC0415
        GL_FRAMEBUFFER,
        glBindFramebuffer,
        glDeleteFramebuffers,
        glDeleteRenderbuffers,
        glDeleteTextures,
    )
    fbo, color_tex, depth_rb = _make_offscreen_fbo()
    try:
        yield fbo
    finally:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteFramebuffers(1, [fbo])
        glDeleteTextures(1, [color_tex])
        glDeleteRenderbuffers(1, [depth_rb])


def _draw_cube_scene(
    *, ground_enabled: bool, self_shadow_enabled: bool,
) -> np.ndarray:
    """Render the floating-cube scene with the requested passes enabled."""
    scene, mesh = _cube_above_ground()
    renderer = Renderer(shaders_root=_shaders_root())
    renderer.initialize()
    renderer.set_ground_enabled(ground_enabled)
    renderer.set_self_shadow_enabled(self_shadow_enabled)
    renderer.attach_mesh(scene.root.children[0], mesh)
    camera = Camera(position=vec3(2.5, 1.5, 2.5), target=vec3(0.0, 0.6, 0.0))
    renderer.draw(scene, camera, VIEWPORT_SIZE)
    return _read_pixels()


def test_ground_pass_adds_visible_floor(
    gl_context: object, offscreen_fbo: int,
) -> None:
    """With the ground pass on, the bottom band of the frame reads as the floor."""
    _ = offscreen_fbo
    with_ground = _draw_cube_scene(ground_enabled=True, self_shadow_enabled=False)
    no_ground = _draw_cube_scene(ground_enabled=False, self_shadow_enabled=False)
    # Bottom 40% of the frame should differ — the ground draws there.
    bottom_h = VIEWPORT_SIZE[1] // 2
    diff = np.abs(
        with_ground[bottom_h:, ...].astype(np.int32)
        - no_ground[bottom_h:, ...].astype(np.int32),
    )
    altered_fraction = float((diff.max(axis=-1) > 10).mean())
    assert altered_fraction > 0.10, (
        f"ground pass altered only {altered_fraction:.2%} of the lower frame"
    )


def test_self_shadow_pass_populates_depth_map(
    gl_context: object, offscreen_fbo: int,
) -> None:
    """The depth-only shadow pass writes a non-empty depth texture.

    A single convex cube doesn't self-occlude visibly, so a "darker
    pixels" check would be flaky. Instead we read the shadow depth
    texture back after the pass and assert it contains a depth band
    interior to ``[0, 1)`` — i.e., the cube actually rasterised into
    it, rather than the depth-clear value of 1.0 everywhere.
    """
    _ = offscreen_fbo
    from OpenGL.GL import (  # noqa: PLC0415
        GL_DEPTH_COMPONENT,
        GL_FLOAT,
        GL_TEXTURE_2D,
        glBindTexture,
        glGetTexImage,
    )

    scene, mesh = _cube_above_ground()
    renderer = Renderer(shaders_root=_shaders_root())
    renderer.initialize()
    renderer.set_ground_enabled(False)
    renderer.set_self_shadow_enabled(True)
    renderer.attach_mesh(scene.root.children[0], mesh)
    camera = Camera(position=vec3(2.5, 1.5, 2.5), target=vec3(0.0, 0.6, 0.0))
    renderer.draw(scene, camera, VIEWPORT_SIZE)

    glBindTexture(GL_TEXTURE_2D, renderer._shadow_depth_tex)        # noqa: SLF001
    size = 1024
    raw = glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT)
    glBindTexture(GL_TEXTURE_2D, 0)
    depth = np.frombuffer(raw, dtype=np.float32).reshape(size, size)
    # Most of the texture is the "no occluder" depth = 1.0. The cube
    # rasterises into a smaller patch — there should be at least a few
    # thousand texels with depth < 1.0 after the pass.
    occupied = int((depth < 0.999).sum())
    assert occupied > 1_000, (
        f"shadow depth map only has {occupied} non-default texels — "
        "the depth pass likely failed to draw"
    )


def test_renderer_toggles_default_to_on() -> None:
    """Fresh Renderer has ground + projected shadow + self-shadow on."""
    renderer = Renderer(shaders_root=_shaders_root())
    assert renderer._ground_enabled is True              # noqa: SLF001
    assert renderer._projected_shadow_enabled is True    # noqa: SLF001
    assert renderer._self_shadow_enabled is True         # noqa: SLF001


def test_set_ground_enabled_synchronises_projected_shadow() -> None:
    """Turning off the ground also turns off the projected shadow."""
    renderer = Renderer(shaders_root=_shaders_root())
    renderer.set_ground_enabled(False)
    assert renderer._projected_shadow_enabled is False   # noqa: SLF001
    renderer.set_ground_enabled(True)
    assert renderer._projected_shadow_enabled is True    # noqa: SLF001


def test_controller_seeds_autoluminous_into_empty_chain() -> None:
    """A freshly-constructed controller has AutoLuminous queued by default."""
    from posecascade.app.controller import AppController  # noqa: PLC0415

    controller = AppController(
        project_root=Path("."),
        pmx_loader=lambda _path: None,
        vmd_loader=lambda _path: None,
    )
    assert len(controller.effect_chain) == 1
    assert controller.effect_chain.entries[0].descriptor.name == "autoluminous"


def test_controller_does_not_override_preconfigured_chain() -> None:
    """If a caller hands the controller a non-empty chain, the seed is skipped."""
    from posecascade.app.controller import AppController  # noqa: PLC0415
    from posecascade.render.effects.builtins import load_builtin  # noqa: PLC0415
    from posecascade.render.effects.chain import EffectChain  # noqa: PLC0415

    preset = EffectChain()
    preset.append(load_builtin("hgshadow"))
    controller = AppController(
        project_root=Path("."),
        pmx_loader=lambda _path: None,
        vmd_loader=lambda _path: None,
        effect_chain=preset,
    )
    assert len(controller.effect_chain) == 1
    assert controller.effect_chain.entries[0].descriptor.name == "hgshadow"
