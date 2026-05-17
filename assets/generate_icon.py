"""Regenerate the desktop app icon from a single source of truth.

Run with the project venv active:

.. code-block:: bash

   py assets/generate_icon.py

Outputs:

* ``assets/PoseCascade.ico``  — multi-resolution Windows icon
  (16 / 24 / 32 / 48 / 64 / 128 / 256). The Nuitka build in
  ``.github/workflows/wheels.yml`` references this file via
  ``--windows-icon-from-ico``.
* ``assets/PoseCascade.png``  — 256×256 PNG, useful for the macOS
  ``--macos-app-icon`` flag (Nuitka converts to ``.icns`` on its own)
  and for any web / README usage.

The design is deliberately reproducible in code rather than checked-in
as a hand-drawn binary so a future visual refresh is a code review
(``git diff assets/generate_icon.py``) rather than an opaque blob diff.

Design notes:

* Background — vertical gradient from deep purple ``#5E35B1`` (top) to
  blue ``#1E88E5`` (bottom), clipped to a rounded square. Reads as
  "creative tool" (purple) + "tech tool" (blue) without colliding with
  Blender (orange), C4D (deep blue) or Maya (grey-cyan).
* Foreground — three white horizontal bars of decreasing width,
  right-aligned, stepping down across the canvas. The cascade reads as
  the project name (PoseCascade) without needing the wordmark.
* The bars are sized in canvas-relative units so the same drawing
  function renders cleanly at every ICO size.

Adding a new size: extend ``ICO_SIZES``. Pillow handles the multi-res
ICO packing automatically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ---- canvas ------------------------------------------------------------------

CANVAS = 256
CORNER_RADIUS = 48          # rounded-square corner; ~19 % of canvas
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)

# ---- colour palette ----------------------------------------------------------

COLOR_TOP = (94, 53, 177, 255)      # #5E35B1 deep purple
COLOR_BOTTOM = (30, 136, 229, 255)  # #1E88E5 blue
BAR_COLOR = (255, 255, 255, 240)    # near-white with slight translucency

# ---- foreground geometry (proportional to CANVAS) ----------------------------

# Margin from canvas edge to the widest bar.
MARGIN_X = 40
# Cascade vertical bounds.
BAR_TOP_Y = 70
BAR_BOTTOM_Y = CANVAS - 70
# Bar visual.
BAR_HEIGHT = 30
BAR_CORNER = 14
# Three bars cascade right-aligned, each shorter than the last.
BAR_WIDTH_FRACTIONS = (1.0, 0.72, 0.44)


def _gradient_background() -> Image.Image:
    """Vertical purple→blue gradient filling the full canvas."""
    # Vectorised across the y axis — putpixel in a loop is 100× slower
    # at 256² and not worth the readability hit.
    arr = np.empty((CANVAS, CANVAS, 4), dtype=np.uint8)
    ts = np.linspace(0.0, 1.0, CANVAS, dtype=np.float32)
    for channel in range(3):
        column = (
            COLOR_TOP[channel] * (1.0 - ts) + COLOR_BOTTOM[channel] * ts
        ).astype(np.uint8)
        arr[:, :, channel] = column[:, None]
    arr[:, :, 3] = 255
    return Image.fromarray(arr, mode="RGBA")


def _rounded_mask() -> Image.Image:
    """Alpha mask the gradient is clipped against — gives the rounded square."""
    mask = Image.new("L", (CANVAS, CANVAS), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, CANVAS - 1, CANVAS - 1), CORNER_RADIUS, fill=255,
    )
    return mask


def _draw_cascade_bars(image: Image.Image) -> None:
    """Three white horizontal bars cascading right-aligned + downward."""
    draw = ImageDraw.Draw(image, "RGBA")
    max_bar_width = CANVAS - 2 * MARGIN_X
    bar_step = (BAR_BOTTOM_Y - BAR_TOP_Y - BAR_HEIGHT) / (len(BAR_WIDTH_FRACTIONS) - 1)
    for index, fraction in enumerate(BAR_WIDTH_FRACTIONS):
        width = max_bar_width * fraction
        right = CANVAS - MARGIN_X
        left = right - width
        top = BAR_TOP_Y + bar_step * index
        bottom = top + BAR_HEIGHT
        draw.rounded_rectangle(
            (left, top, right, bottom), BAR_CORNER, fill=BAR_COLOR,
        )


def render_icon() -> Image.Image:
    """Compose the full 256×256 RGBA icon."""
    image = _gradient_background()
    image.putalpha(_rounded_mask())
    _draw_cascade_bars(image)
    return image


def write_outputs(icon: Image.Image, *, ico_path: Path, png_path: Path) -> None:
    """Save the multi-res ICO and the 256×256 PNG."""
    # Pillow's ICO writer downscales the source image to every requested
    # size. Quality is acceptable for non-pixel-art icons because the
    # design is bold + low-detail; if a future revision wants per-size
    # tuning (e.g. dropping the smallest bar at 16×16) move to manually
    # rendering at each size and packing with ``append_images``.
    icon.save(ico_path, format="ICO", sizes=[(s, s) for s in ICO_SIZES])
    icon.save(png_path, format="PNG", optimize=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    icon = render_icon()
    write_outputs(
        icon,
        ico_path=here / "PoseCascade.ico",
        png_path=here / "PoseCascade.png",
    )
    print(f"wrote {here / 'PoseCascade.ico'} ({', '.join(f'{s}px' for s in ICO_SIZES)})")
    print(f"wrote {here / 'PoseCascade.png'} (256×256)")


if __name__ == "__main__":
    main()
