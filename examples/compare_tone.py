"""Side-by-side comparison: render with vs without the ``mmd_tone`` effect.

The MMD tone effect lifts mid-tones a touch, soft-knees the highlights,
boosts saturation slightly, and applies a warm tint. Stacked after
AutoLuminous it gives a sRGB-correct render a hint of the warmer,
slightly colour-amplified feel MMD's non-standard pipeline produces.

This script renders the same scene twice — bare sRGB output on the
left, ``mmd_tone`` applied on the right — and saves a labelled PNG.

Usage::

    py examples/compare_tone.py [--output PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _demo_lib import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    load_character,
    make_renderer,
    read_pixels,
    save_side_by_side,
    setup_offscreen_gl,
)


def _build_tone_chain() -> object:
    """Single-entry chain: ``mmd_tone`` at descriptor defaults."""
    from posecascade.render.effects.builtins import load_builtin  # noqa: PLC0415
    from posecascade.render.effects.chain import EffectChain  # noqa: PLC0415

    chain = EffectChain()
    chain.append(load_builtin("mmd_tone"))
    return chain


def _apply_chain(
    renderer: object, chain: object, scene_color_tex: int,
    out_fbo: int, width: int, height: int,
) -> None:
    from posecascade.render.effects.executor import EffectChainExecutor  # noqa: PLC0415

    executor = EffectChainExecutor(project_root=Path(__file__).resolve().parents[1])
    executor.compile_chain(chain)
    renderer.apply_effect_chain(
        executor, chain, (width, height),
        main_color_texture=scene_color_tex,
        default_framebuffer=out_fbo,
    )


def _draw_pane(
    renderer: object, scene: object, camera: object,
    scene_fbo: int, scene_color: int, width: int, height: int,
    *, apply_tone: bool,
):
    from OpenGL.GL import GL_FRAMEBUFFER, glBindFramebuffer  # noqa: PLC0415

    glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo)
    renderer.draw(scene.scene, camera, (width, height))
    if apply_tone:
        _apply_chain(
            renderer, _build_tone_chain(), scene_color,
            scene_fbo, width, height,
        )
        glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo)
    return read_pixels(width, height)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compare_tone")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "compare_tone.png",
    )
    args = parser.parse_args(argv)

    from posecascade.render.camera import Camera  # noqa: PLC0415
    from posecascade.utils.math3d import vec3  # noqa: PLC0415

    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    _surface, _context, scene_fbo, scene_color = setup_offscreen_gl(width, height)
    renderer = make_renderer()
    imported = load_character()
    renderer.populate_from_scene(imported)
    camera = Camera(position=vec3(0.6, 0.9, 1.8), target=vec3(0.0, 0.55, 0.0))

    pixels_plain = _draw_pane(
        renderer, imported, camera, scene_fbo, scene_color, width, height,
        apply_tone=False,
    )
    pixels_tone = _draw_pane(
        renderer, imported, camera, scene_fbo, scene_color, width, height,
        apply_tone=True,
    )

    save_side_by_side(
        pixels_plain, pixels_tone,
        label_left="sRGB only",
        label_right="+ mmd_tone",
        output=args.output,
    )
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
