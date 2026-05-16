"""Side-by-side comparison: single key light vs HighDef multi-light setup.

The default light setup is one directional source — fine for a hero
shot but it leaves the silhouette flat against a same-toned
background. The HighDef preset (``apply_highdef_light_preset``) adds:

- A cool back-rim light tracing the silhouette from above-back.
- A faint warm front-fill bouncing into the under-jaw / inner skirt.

This script renders the character at a 3/4 back camera angle so the
rim light traces the visible edge, then compares "primary only" vs
"primary + back rim + front fill". The HighDef pane should show a
brighter edge along the back of the hair / shoulder / arm.

Usage::

    py examples/compare_lights.py [--output PATH]
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


def _draw_pane(
    renderer: object, scene: object, camera: object,
    scene_fbo: int, width: int, height: int, *, highdef: bool,
):
    """Render one pane with the requested lighting setup."""
    from OpenGL.GL import GL_FRAMEBUFFER, glBindFramebuffer  # noqa: PLC0415

    if highdef:
        renderer.apply_highdef_light_preset()
    else:
        renderer.set_secondary_lights([])
    glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo)
    renderer.draw(scene.scene, camera, (width, height))
    return read_pixels(width, height)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compare_lights")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "compare_lights.png",
    )
    args = parser.parse_args(argv)

    from posecascade.render.camera import Camera  # noqa: PLC0415
    from posecascade.utils.math3d import vec3  # noqa: PLC0415

    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    _surface, _context, scene_fbo, _scene_color = setup_offscreen_gl(width, height)
    # Start with HighDef OFF so the first pane is the "primary only"
    # baseline. _draw_pane flips the setting per call.
    renderer = make_renderer(highdef_lights=False)
    imported = load_character()
    renderer.populate_from_scene(imported)

    # 3/4 BACK camera so the HighDef preset's back-rim light traces a
    # visible edge along the hair + shoulder silhouette. A front-3/4
    # framing would only show the front-fill, which is much subtler.
    camera = Camera(
        position=vec3(-1.2, 0.95, -1.6),
        target=vec3(0.0, 0.85, 0.0),
    )

    pixels_primary = _draw_pane(
        renderer, imported, camera, scene_fbo, width, height, highdef=False,
    )
    pixels_highdef = _draw_pane(
        renderer, imported, camera, scene_fbo, width, height, highdef=True,
    )

    save_side_by_side(
        pixels_primary, pixels_highdef,
        label_left="primary light only",
        label_right="+ HighDef rim + fill",
        output=args.output,
    )
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
