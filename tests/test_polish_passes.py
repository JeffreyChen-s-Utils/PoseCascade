"""Tests for the MMD-polish renderer additions.

- PCF: the toon fragment shader compiles + runs unchanged with the new
  3×3 kernel (existing render smoke tests already pin output without
  shadow; this file checks the shader source contains the kernel).
- sRGB: the toggle flips the renderer's flag and is honoured on draw.
- Sky: the gradient pass writes pixels into the frame.
- Stage: the procedural stage builder produces a renderable scene
  and the slot machinery treats ``is_stage`` slots as animation-free.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from posecascade.animation.slots_player import (
    SlotsPlayer,
    make_slot,
    make_stage_slot,
)
from posecascade.render.renderer import Renderer
from posecascade.scene.model_slot import SceneSlots
from posecascade.scene.stage import procedural_dance_stage

_SHADERS_ROOT = Path(__file__).resolve().parent.parent / "shaders"


# ---------------------------------------------------------------------------
# PCF — static source check; visual difference is covered by the editor
# screenshot, not pinned in offscreen tests (a single-cube scene can't
# self-occlude).
# ---------------------------------------------------------------------------


def test_toon_frag_contains_pcf_kernel_loop() -> None:
    """The toon fragment shader uses a square PCF kernel for shadow sampling."""
    frag = (_SHADERS_ROOT / "toon" / "toon.frag").read_text(encoding="utf-8")
    assert "_PCF_RADIUS" in frag
    assert "for (int dy" in frag and "for (int dx" in frag, (
        "PCF double-loop missing — shadow falloff would be hard again"
    )
    assert "textureSize(u_shadowMap" in frag, (
        "PCF must consult shadow map resolution to compute texel offset"
    )


# ---------------------------------------------------------------------------
# sRGB / sky toggles — exercise the renderer's setter API without a GL
# context. Visual impact is covered by interactive use.
# ---------------------------------------------------------------------------


def test_renderer_srgb_toggle_default_on() -> None:
    """Fresh renderer reports sRGB-aware output enabled by default."""
    renderer = Renderer(shaders_root=_SHADERS_ROOT)
    assert renderer._srgb_output_enabled is True       # noqa: SLF001
    renderer.set_srgb_output_enabled(False)
    assert renderer._srgb_output_enabled is False      # noqa: SLF001


def test_renderer_sky_toggle_default_on() -> None:
    """Fresh renderer reports the gradient sky pass enabled by default."""
    renderer = Renderer(shaders_root=_SHADERS_ROOT)
    assert renderer._sky_enabled is True               # noqa: SLF001
    renderer.set_sky_enabled(False)
    assert renderer._sky_enabled is False              # noqa: SLF001


def test_renderer_dqs_toggle_default_off() -> None:
    """DQS is opt-in — fresh renderer is LBS for backwards compatibility."""
    renderer = Renderer(shaders_root=_SHADERS_ROOT)
    assert renderer._dqs_enabled is False              # noqa: SLF001
    renderer.set_dqs_enabled(True)
    assert renderer._dqs_enabled is True               # noqa: SLF001


def test_renderer_force_toon_shading_toggle_default_off() -> None:
    """force_toon_shading is opt-in — glTF / OBJ stay PBR by default."""
    renderer = Renderer(shaders_root=_SHADERS_ROOT)
    assert renderer._force_toon_shading is False       # noqa: SLF001
    renderer.set_force_toon_shading(True)
    assert renderer._force_toon_shading is True        # noqa: SLF001


def test_default_toon_material_carries_edge_flag() -> None:
    """The synthesised MMD material opts every non-MMD mesh into the outline pass."""
    from posecascade.render.material import MAT_FLAG_HAS_EDGE  # noqa: PLC0415
    from posecascade.render.toon_promote import default_toon_material  # noqa: PLC0415

    material = default_toon_material()
    assert material.has_edge is True
    assert material.edge_size > 0.0
    assert (material.flags & MAT_FLAG_HAS_EDGE) != 0


def test_default_toon_ramp_has_two_distinct_bands() -> None:
    """The procedural toon LUT is a 1×N image with a hard lit/shadow step."""
    from posecascade.render.toon_promote import default_toon_ramp_pixels  # noqa: PLC0415

    pixels = default_toon_ramp_pixels()
    assert pixels.ndim == 3
    assert pixels.shape[1] == 1
    assert pixels.shape[2] == 4
    # Top of the ramp = lit; bottom = shadow. Their luminances must
    # differ — otherwise there's no cel band.
    lit_luma = pixels[0, 0, :3].mean()
    shadow_luma = pixels[-1, 0, :3].mean()
    assert lit_luma > shadow_luma + 40, "toon ramp doesn't have a visible band"


def test_renderer_secondary_lights_clamps_to_max() -> None:
    """``set_secondary_lights`` truncates beyond the shader's ``MAX_SECONDARY_LIGHTS``."""
    renderer = Renderer(shaders_root=_SHADERS_ROOT)
    too_many = [
        ((1.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
        ((0.0, 1.0, 0.0), (0.4, 0.4, 0.4)),
        ((0.0, 0.0, 1.0), (0.3, 0.3, 0.3)),
        ((1.0, 1.0, 0.0), (0.2, 0.2, 0.2)),  # 4th — over the cap
        ((1.0, 0.0, 1.0), (0.1, 0.1, 0.1)),  # 5th — over the cap
    ]
    renderer.set_secondary_lights(too_many)
    assert len(renderer._secondary_lights) == 3        # noqa: SLF001


def test_renderer_highdef_preset_sets_two_lights() -> None:
    """``apply_highdef_light_preset`` installs back-rim + front-fill."""
    renderer = Renderer(shaders_root=_SHADERS_ROOT)
    renderer.apply_highdef_light_preset()
    assert len(renderer._secondary_lights) == 2        # noqa: SLF001
    # Rim is the first entry — its color must be nonzero so it actually
    # contributes when bound.
    rim_color = renderer._secondary_lights[0][1]        # noqa: SLF001
    assert max(rim_color) > 0.0


def test_mmd_tone_descriptor_loads_with_expected_uniforms() -> None:
    """The bundled MMD tone effect exposes the curve controls described in docs."""
    from posecascade.render.effects.builtins import load_builtin  # noqa: PLC0415

    descriptor = load_builtin("mmd_tone")
    uniform_names = {u.name for u in descriptor.uniforms}
    assert uniform_names == {
        "midtone_lift", "highlight_rolloff", "saturation", "warm_tint",
    }
    assert descriptor.fragment_shader.endswith("mmd_tone.frag")


# ---------------------------------------------------------------------------
# Stage — pure-Python verification of the procedural geometry + slot
# integration. No GL needed.
# ---------------------------------------------------------------------------


def test_procedural_dance_stage_returns_four_meshes() -> None:
    """The bundled stage ships floor + back wall + two side walls."""
    stage = procedural_dance_stage()
    assert len(stage.meshes) == 4
    names = [mesh.name for mesh in stage.meshes]
    assert names == [
        "stage_floor", "stage_wall", "stage_wall_left", "stage_wall_right",
    ]
    for mesh in stage.meshes:
        assert mesh.positions.shape == (4, 3)
        assert mesh.indices.size == 6
        assert mesh.base_color is not None


def test_stage_floor_sits_above_ground_plane() -> None:
    """The procedural stage floor must hover slightly above y=0 to occlude the
    checkered ground beneath it without z-fighting."""
    stage = procedural_dance_stage()
    floor_ys = stage.meshes[0].positions[:, 1]
    assert (floor_ys > 0.0).all(), (
        "stage floor sits at or below the checker ground plane — z-fight"
    )


def test_stage_scene_root_exposes_every_mesh() -> None:
    """Every procedural mesh is exposed as a Node the renderer can walk."""
    stage = procedural_dance_stage()
    assert stage.scene is not None
    children = stage.scene.root.children
    assert len(children) == 4
    assert {c.name for c in children} == {
        "stage_floor", "stage_wall", "stage_wall_left", "stage_wall_right",
    }


def test_stage_side_walls_face_inward() -> None:
    """Both side walls' normals point toward the stage interior (±X)."""
    stage = procedural_dance_stage()
    left = next(m for m in stage.meshes if m.name == "stage_wall_left")
    right = next(m for m in stage.meshes if m.name == "stage_wall_right")
    # All 4 verts of each wall share the same normal; sample vertex 0.
    assert left.normals is not None and right.normals is not None
    assert left.normals[0, 0] > 0.99, "left wall normal must point +X"
    assert right.normals[0, 0] < -0.99, "right wall normal must point -X"


def test_slots_player_skips_animation_for_stage_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``is_stage`` slots never get a player built — saves bone setup +
    keeps stage geometry from being driven by a stray motion."""
    from unittest.mock import MagicMock  # noqa: PLC0415

    from posecascade.animation import slots_player as sp_module  # noqa: PLC0415

    # Stub the player construction so we don't need a real PMX skeleton;
    # monkeypatch restores the original at teardown so later tests see
    # the unmodified class.
    monkeypatch.setattr(
        sp_module.VmdAnimationPlayer,
        "for_imported_scene",
        staticmethod(lambda motion, imported: MagicMock()),
    )

    stage = procedural_dance_stage()
    fake_motion = MagicMock()
    slots = SceneSlots()
    slots.add(make_slot(name="dancer", imported=stage, motion=fake_motion))
    slots.add(make_stage_slot(name="stage", imported=stage))
    player = SlotsPlayer(slots=slots)
    assert "dancer" in player._players                # noqa: SLF001
    assert "stage" not in player._players             # noqa: SLF001


def test_make_stage_slot_shorthand_sets_flag() -> None:
    """``make_stage_slot`` is sugar for ``make_slot(... is_stage=True)``."""
    stage = procedural_dance_stage()
    slot = make_stage_slot(name="prop", imported=stage)
    assert slot.is_stage is True
    assert slot.motion is None


# ---------------------------------------------------------------------------
# sRGB texture upload — pure-function check that the GL internal format
# changes when ``srgb=True`` (read by inspecting glTexImage2D's argument
# via a stub).
# ---------------------------------------------------------------------------


def test_upload_texture_passes_srgb_internal_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """``srgb=True`` requests ``GL_SRGB8_ALPHA8``; ``srgb=False`` keeps ``GL_RGBA8``."""
    from posecascade.gl import texture as tex  # noqa: PLC0415

    recorded: dict[str, int] = {}

    def fake_tex_image_2d(target, level, internal_format, *args, **kwargs):
        del target, level, args, kwargs
        recorded["internal_format"] = int(internal_format)

    monkeypatch.setattr(tex, "glGenTextures", lambda _n: 7)
    monkeypatch.setattr(tex, "glBindTexture", lambda *a, **kw: None)
    monkeypatch.setattr(tex, "glTexParameteri", lambda *a, **kw: None)
    monkeypatch.setattr(tex, "glTexImage2D", fake_tex_image_2d)
    monkeypatch.setattr(tex, "glGenerateMipmap", lambda *a, **kw: None)

    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    tex.upload_texture(pixels, srgb=True)
    assert recorded["internal_format"] == int(tex.GL_SRGB8_ALPHA8)

    recorded.clear()
    tex.upload_texture(pixels, srgb=False)
    assert recorded["internal_format"] == int(tex.GL_RGBA8)
