"""Side-by-side comparison: scene with vs without AutoLuminous bloom.

The bundled character carries no emission map, so the default
AutoLuminous threshold (0.85 luma) gates the bloom off. This script
overrides the threshold to 0.40 in the chain entry so the bright
hair / vest highlights actually trigger the bloom, then renders the
same scene twice — once with the chain skipped, once with the chain
applied — and writes a labelled side-by-side PNG.

Usage::

    py examples/compare_bloom.py [--output PATH]
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


def _build_bloom_chain() -> object:
    """Single-entry chain: AutoLuminous with a lowered threshold so the
    bundled character's bright hair / vest highlights actually trigger
    bloom instead of being gated off."""
    from posecascade.render.effects.builtins import load_builtin  # noqa: PLC0415
    from posecascade.render.effects.chain import EffectChain  # noqa: PLC0415

    chain = EffectChain()
    entry = chain.append(load_builtin("autoluminous"))
    # Default threshold is 0.85 (gates almost everything off when no
    # emission map is present). 0.40 lets the brightest model pixels
    # — hair highlights, white shirt — drive bloom.
    entry.uniform_overrides["threshold"] = 0.40
    entry.uniform_overrides["intensity"] = 1.6
    return chain


def _render_with_chain(
    renderer: object, chain: object, scene_color_tex: int,
    out_fbo: int, width: int, height: int,
) -> None:
    """Drive ``renderer.apply_effect_chain`` so the chain output lands in ``out_fbo``."""
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
    *, apply_bloom: bool,
):
    """Render the scene; optionally route through AutoLuminous; readback pixels."""
    from OpenGL.GL import GL_FRAMEBUFFER, glBindFramebuffer  # noqa: PLC0415

    glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo)
    renderer.draw(scene.scene, camera, (width, height))
    if apply_bloom:
        _render_with_chain(
            renderer, _build_bloom_chain(), scene_color,
            scene_fbo, width, height,
        )
        glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo)
    return read_pixels(width, height)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compare_bloom")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "compare_bloom.png",
    )
    args = parser.parse_args(argv)

    from posecascade.render.camera import Camera  # noqa: PLC0415
    from posecascade.utils.math3d import vec3  # noqa: PLC0415

    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    _surface, _context, scene_fbo, scene_color = setup_offscreen_gl(width, height)
    renderer = make_renderer()
    # IMPORTANT: load the character ONCE and reuse for both populate +
    # draw. The renderer's mesh registry keys off ``id(node)``, so a
    # second ``load_character()`` call would yield fresh Node objects
    # whose ids don't match the registered meshes — the model would
    # silently disappear from the render.
    imported = load_character()
    renderer.populate_from_scene(imported)
    camera = Camera(position=vec3(0.6, 0.9, 1.8), target=vec3(0.0, 0.55, 0.0))

    pixels_no_bloom = _draw_pane(
        renderer, imported, camera, scene_fbo, scene_color, width, height,
        apply_bloom=False,
    )
    pixels_with_bloom = _draw_pane(
        renderer, imported, camera, scene_fbo, scene_color, width, height,
        apply_bloom=True,
    )

    save_side_by_side(
        pixels_no_bloom, pixels_with_bloom,
        label_left="bloom OFF",
        label_right="AutoLuminous bloom",
        output=args.output,
    )
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
