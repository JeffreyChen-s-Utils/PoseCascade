"""Side-by-side comparison: LBS vs DQS skinning at an extreme arm twist.

Linear-blend skinning collapses the volume of a joint when two bones'
rotations differ sharply — the classic "candy-wrapper" pinch at
shoulders / elbows / wrists. Dual-quaternion skinning replaces the
matrix interpolation with a screw-motion blend that preserves
volume.

This script poses the character's right upper arm with a 2.2-radian
(~126°) forward swing — far enough that LBS shows visible pinching at
the shoulder skin — then renders the same pose twice, once with LBS
and once with DQS, and writes a labelled PNG.

Usage::

    py examples/compare_dqs.py [--output PATH]
"""
from __future__ import annotations

import argparse
import math
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

_TWIST_BONE = "J_Bip_R_UpperArm"
# Rotate around the bone's local X axis. For a VRoid upper-arm bone
# whose rest pose is roughly aligned with its parent's frame, local-X
# rotation swings the arm up/down through ~90° — large enough that
# LBS visibly pinches at the shoulder skin while DQS keeps it round.
# Sign is positive so the right arm raises upward (the more dramatic
# direction for this character's rig).
_TWIST_ANGLE_RADIANS = 2.0


def _find_bone(scene: object, name: str) -> object:
    """Walk the scene to find a Node by name; raises if missing."""
    for node in scene.scene.root.traverse():
        if node.name == name:
            return node
    raise RuntimeError(f"bone {name!r} not found in scene")


def _apply_extreme_twist(scene: object) -> None:
    """Compose an in-place twist on top of the bone's existing local matrix.

    glTF bones carry their rest-pose matrix as ``matrix_override`` so
    decomposition is lossy; ``set_rotation`` would clear the override
    and snap the bone to ``(0, 0, 0)`` with our quaternion, losing the
    arm's outward position. Instead we multiply the existing local
    matrix by a Z-axis rotation matrix and stash it back as the new
    override — pose-preserving + still gives us the dramatic twist.
    """
    import numpy as np  # noqa: PLC0415

    from posecascade.scene.transform import Transform  # noqa: PLC0415

    bone = _find_bone(scene, _TWIST_BONE)
    cos_a = math.cos(_TWIST_ANGLE_RADIANS)
    sin_a = math.sin(_TWIST_ANGLE_RADIANS)
    x_rotation = np.array(
        [
            [1.0, 0.0,    0.0,    0.0],
            [0.0, cos_a, -sin_a,  0.0],
            [0.0, sin_a,  cos_a,  0.0],
            [0.0, 0.0,    0.0,    1.0],
        ],
        dtype=np.float32,
    )
    existing = bone.transform.to_matrix()
    twisted = existing @ x_rotation
    bone.transform = Transform.from_raw_matrix(twisted)


def _draw_pane(
    renderer: object, scene: object, camera: object,
    scene_fbo: int, width: int, height: int, *, dqs: bool,
):
    from OpenGL.GL import GL_FRAMEBUFFER, glBindFramebuffer  # noqa: PLC0415

    renderer.set_dqs_enabled(dqs)
    glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo)
    renderer.draw(scene.scene, camera, (width, height))
    return read_pixels(width, height)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compare_dqs")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "compare_dqs.png",
    )
    args = parser.parse_args(argv)

    from posecascade.render.camera import Camera  # noqa: PLC0415
    from posecascade.utils.math3d import vec3  # noqa: PLC0415

    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    _surface, _context, scene_fbo, _scene_color = setup_offscreen_gl(width, height)
    renderer = make_renderer()
    imported = load_character()
    renderer.populate_from_scene(imported)

    # Pose the right shoulder before either render so both panes see
    # the same input — the only variable between them is the skinning
    # method.
    _apply_extreme_twist(imported)

    # Frame the upper body so the raised arm + shoulder seam fill the
    # pane. With the arm bent ~115°, the shoulder skin under LBS pinches
    # visibly; DQS keeps the volume round.
    camera = Camera(
        position=vec3(0.95, 1.10, 1.6),
        target=vec3(0.0, 0.85, 0.0),
    )

    pixels_lbs = _draw_pane(
        renderer, imported, camera, scene_fbo, width, height, dqs=False,
    )
    pixels_dqs = _draw_pane(
        renderer, imported, camera, scene_fbo, width, height, dqs=True,
    )
    _ = math  # ruff F401 guard — kept for future trig-driven poses

    save_side_by_side(
        pixels_lbs, pixels_dqs,
        label_left="LBS (candy-wrapper)",
        label_right="DQS (volume preserved)",
        output=args.output,
    )
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
