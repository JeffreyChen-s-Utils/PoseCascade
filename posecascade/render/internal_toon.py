"""Built-in MikuMikuDance-style toon ramps.

PMX materials may reference one of ten "internal" toon ramps (``toon01``
through ``toon10``) instead of an external texture. We ship procedural
analogues under ``posecascade/assets/toon/`` and load them on demand into
:class:`~posecascade.assets.types.Texture` instances so they slot into the
importer's normal texture table.

The shipped ramps are not bit-for-bit copies of MMD's; they are a smooth
gradient from light (top, ``v ≈ 0``) to dark (bottom, ``v ≈ 1``) with
progressively sharper transitions per index — visually equivalent for
shading without copying anyone's binary assets.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np

INTERNAL_TOON_COUNT = 10
_RAMP_HEIGHT = 256
_LIGHT_INTENSITY = 1.0
_DARK_INTENSITY = 0.4


def _assets_root() -> Path:
    """Return the absolute path of ``posecascade/assets``."""
    return Path(__file__).resolve().parent.parent / "assets"


def internal_toon_path(index: int) -> Path:
    """Path to the ``toonNN.png`` file for ramp ``index`` (``0 ≤ index < 10``)."""
    if not 0 <= index < INTERNAL_TOON_COUNT:
        raise ValueError(f"internal toon index out of range: {index}")
    return _assets_root() / "toon" / f"toon{index + 1:02d}.png"


def generate_ramp_pixels(index: int) -> np.ndarray:
    """Procedurally synthesise the ``index``-th toon ramp as ``(H, 1, 4)`` uint8.

    Ramp ``index = 0`` is the softest gradient (almost continuous shading);
    higher indices snap closer to a hard two-tone cel boundary.
    """
    if not 0 <= index < INTERNAL_TOON_COUNT:
        raise ValueError(f"internal toon index out of range: {index}")
    transition = 0.55 - index * 0.025
    width = 0.5 / (1.0 + index)
    pixels = np.empty((_RAMP_HEIGHT, 1, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    for row in range(_RAMP_HEIGHT):
        t = row / (_RAMP_HEIGHT - 1)
        shade = _smoothstep_shade(t, transition, width)
        intensity = _LIGHT_INTENSITY * (1.0 - shade) + _DARK_INTENSITY * shade
        gray = int(round(intensity * 255))
        pixels[row, 0, :3] = gray
    return pixels


def _smoothstep_shade(t: float, transition: float, width: float) -> float:
    """Smooth ``0 → 1`` step centred on ``transition`` with ``width`` falloff."""
    half = width * 0.5
    if t < transition - half:
        return 0.0
    if t > transition + half:
        return 1.0
    local = (t - (transition - half)) / width
    return local * local * (3.0 - 2.0 * local)


def _texture_class():     # noqa: ANN202 — late-bound to avoid import cycle
    from posecascade.assets.types import Texture  # noqa: PLC0415
    return Texture


def load_internal_toon(index: int):                 # noqa: ANN201 — late-bound Texture
    """Load the ``index``-th internal ramp as a :class:`Texture`.

    Reads the on-disk PNG when it exists and falls back to procedural
    generation otherwise — keeps tests green even if the bundled asset is
    accidentally absent and lets the build script regenerate the ramps
    from the same source of truth.
    """
    pixels = _read_or_generate(index)
    return _texture_class()(name=f"toon{index + 1:02d}", pixels=pixels, srgb=False)


def _read_or_generate(index: int) -> np.ndarray:
    path = internal_toon_path(index)
    if not path.is_file():
        return generate_ramp_pixels(index)
    try:
        from PIL import Image, UnidentifiedImageError  # noqa: PLC0415
        with Image.open(BytesIO(path.read_bytes())) as opened:
            pil = opened.convert("RGBA")
            return np.array(pil, dtype=np.uint8)
    except (OSError, UnidentifiedImageError):
        return generate_ramp_pixels(index)


def write_all_ramps_to_disk() -> list[Path]:
    """Regenerate every internal-toon PNG; used by the bundled build script."""
    from PIL import Image  # noqa: PLC0415
    out: list[Path] = []
    target_dir = _assets_root() / "toon"
    target_dir.mkdir(parents=True, exist_ok=True)
    for index in range(INTERNAL_TOON_COUNT):
        pixels = generate_ramp_pixels(index)
        path = internal_toon_path(index)
        Image.fromarray(pixels, mode="RGBA").save(path)
        out.append(path)
    return out
