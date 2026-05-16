"""End-to-end render of March 7th via the native MMD (PMX) code path.

Unlike ``mmd_demo.py`` — which exercises the ``force_toon_shading``
fallback on a glTF asset — this script loads a real PMX
(``examples/assets/march7th/march7th.pmx``) so the renderer's
PMX-native code path actually fires: per-mesh ``MMDMaterial`` from the
PMX, sphere-texture composite, edge flag honoured from the file,
toon ramp indices read from the model rather than synthesised.

The PMX is a Gregman-uploaded version of the March 7th character,
converted via Blender's ``mmd_tools`` add-on; it ships with five
baseColor PNGs under ``examples/assets/march7th/textures/``. See the
matching ``NOTICE.md`` for license + IP attribution (CC-BY 4.0 via
Sketchfab uploader Gregman; character © HoYoverse, Fan Content
Guidelines). The default glTF demo character is ``herta.glb`` —
this script intentionally exercises the PMX-native path instead.

Default behaviour: a single static frame at the bundled hero camera.
``--frames N`` (with ``N > 1``) drives a simple breathing + body-sway
animation across N evenly-spaced timestamps and writes the resulting
frames as a horizontal strip PNG so the motion reads in a single
image — useful for visual smoke checks where opening a video player
isn't practical.

Usage::

    py examples/march7th_pmx_demo.py                  # one frame, static pose
    py examples/march7th_pmx_demo.py --frames 6       # 6-frame breathing strip
    py examples/march7th_pmx_demo.py --output D:/Temp/out.png --frames 4

If the PMX is missing for any reason (license takedown, fresh clone
with the asset stripped, …) the script prints a helpful message
pointing at the NOTICE file and exits cleanly with status 1.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _demo_lib import (  # noqa: E402  — path injection above is the whole point
    REPO_ROOT,
    make_renderer,
    read_pixels,
    setup_offscreen_gl,
)

_MARCH7TH_PMX = REPO_ROOT / "examples" / "assets" / "march7th" / "march7th.pmx"
_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "march7th_pmx_demo.png"
_DEFAULT_WIDTH = 768
_DEFAULT_HEIGHT = 1024
# Per-frame strip dimensions when rendering an animated sequence. The
# strip's height matches a single render; width is N frames wide. We
# scale each frame down so the strip stays under typical preview size
# even with 6+ frames.
_STRIP_FRAME_WIDTH = 320
_STRIP_FRAME_HEIGHT = 480
# Animation profile — sin-driven amplitudes the loop sweeps each
# bone through. Tuned to look like idle breathing + a gentle sway,
# nothing dramatic. Bone names are the FBX-style identifiers preserved
# by mmd_tools' GLB→PMX conversion (Sketchfab's original FBX rig).
_UPPER_BODY_BONE = "Chest_M_049"     # March 7th's chest bone (between Spine2 and Neck)
_HEAD_BONE = "Head_M_055"            # head bone
_SHOULDER_L_BONE = "Shoulder_L_0183"  # left shoulder
_SHOULDER_R_BONE = "Shoulder_R_0233"  # right shoulder
_SWAY_AMPLITUDE_RADIANS = 0.30       # body yaw left-right (~17°)
_BREATH_AMPLITUDE_RADIANS = 0.15     # chest tilt forward-back (~9°)
_HEAD_LOOK_AMPLITUDE_RADIANS = 0.50  # head turn left-right (~29°)
_ARM_SWING_AMPLITUDE_RADIANS = 0.25  # arms swing forward-back (~14°)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="march7th_pmx_demo")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=_DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=_DEFAULT_HEIGHT)
    parser.add_argument(
        "--frames", type=int, default=1,
        help="Render N frames into a horizontal strip (1 = single static render).",
    )
    args = parser.parse_args(argv)

    if not _MARCH7TH_PMX.is_file():
        notice = _MARCH7TH_PMX.parent / "NOTICE.md"
        sys.stderr.write(
            f"March 7th PMX not found at {_MARCH7TH_PMX}\n"
            f"See {notice} for the upstream URL + license; the asset must be\n"
            "present locally for this demo (run mmd_demo.py with herta.glb\n"
            "instead if you just want the glTF + force_toon path).\n",
        )
        return 1

    from OpenGL.GL import GL_FRAMEBUFFER, glBindFramebuffer  # noqa: PLC0415

    from posecascade.assets.importer_manager import (  # noqa: PLC0415
        ImporterManager,
    )
    from posecascade.render.camera import Camera  # noqa: PLC0415
    from posecascade.utils.math3d import vec3  # noqa: PLC0415

    print(f"booting GL context @ {args.width}x{args.height} …")
    _surface, _context, fbo, _color = setup_offscreen_gl(args.width, args.height)

    print("building renderer (force_toon=False, HighDef lights) …")
    renderer = make_renderer(force_toon=False, highdef_lights=True)

    print(f"loading model: {_MARCH7TH_PMX.relative_to(REPO_ROOT)}")
    manager = ImporterManager(importers_root=REPO_ROOT / "importers")
    manager.discover()
    scene = manager.load(_MARCH7TH_PMX)
    print(
        f"  meshes={len(scene.meshes)}, textures={len(scene.textures)}, "
        f"bones={len(scene.skins[0].joints) if scene.skins else 0}",
    )
    renderer.populate_from_scene(scene)

    # mmd_tools scales meshes by ~12.5× on PMX export (Blender metre →
    # MMD unit); the model spans Y ≈ [-7, 15] with hips around Y=4 and
    # the head crown near Y=14. The camera frames the upper body at
    # eye-level so the toon shading on the face reads clearly.
    camera = Camera(
        position=vec3(6.0, 8.0, -22.0),
        target=vec3(0.0, 5.0, 0.0),
    )

    if args.frames <= 1:
        print(f"rendering single frame → {args.output} …")
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        renderer.draw(scene.scene, camera, (args.width, args.height))
        pixels = read_pixels(args.width, args.height)
        _ = np  # numpy lives in _demo_lib's helpers, kept imported here for clarity
        Image.fromarray(pixels, "RGBA").save(args.output)
        print(f"saved {args.output}")
        return 0

    print(f"rendering {args.frames}-frame breathing + sway animation …")
    base_matrices = _capture_base_matrices(
        scene,
        (_UPPER_BODY_BONE, _HEAD_BONE, _SHOULDER_L_BONE, _SHOULDER_R_BONE),
    )
    if not base_matrices:
        sys.stderr.write(
            "no animated bones found on this model — the strip would be N "
            "copies of the same frame. Falling back to a single static render.\n",
        )
        args.frames = 1
        return main([
            "--output", str(args.output),
            "--width", str(args.width),
            "--height", str(args.height),
            "--frames", "1",
        ])

    frames = []
    for index in range(args.frames):
        phase = (index / max(args.frames, 1)) * math.tau
        _apply_animation_frame(base_matrices, phase)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        renderer.draw(scene.scene, camera, (args.width, args.height))
        frames.append(read_pixels(args.width, args.height))

    _save_strip(frames, args.output)
    print(f"saved {args.output} ({args.frames} frames @ "
          f"{_STRIP_FRAME_WIDTH}x{_STRIP_FRAME_HEIGHT} each)")
    return 0


def _capture_base_matrices(
    scene: object, bone_names: tuple[str, ...],
) -> dict[str, tuple[object, np.ndarray]]:
    """Return ``{name → (node, rest_matrix)}`` for every requested bone we find.

    Animating bones means *modulating* their rest pose each frame
    rather than replacing it; capturing the resting local matrix once
    here lets the per-frame callback apply a fresh rotation against
    a stable baseline instead of accumulating drift.
    """
    found: dict[str, tuple[object, np.ndarray]] = {}
    for node in scene.scene.root.traverse():
        if node.name in bone_names and node.name not in found:
            found[node.name] = (node, node.transform.to_matrix().copy())
    return found


def _apply_animation_frame(
    base_matrices: dict[str, tuple[object, np.ndarray]], phase: float,
) -> None:
    """Drive each captured bone with a phase-dependent rotation.

    The phase parameter is the loop progress in radians ``[0, 2π)``;
    different bones use different multiples of it so head, chest, and
    body don't all swing in lockstep.
    """
    from posecascade.scene.transform import Transform  # noqa: PLC0415

    sway = _SWAY_AMPLITUDE_RADIANS * math.sin(phase)
    breath = _BREATH_AMPLITUDE_RADIANS * math.sin(phase * 2.0)
    head_turn = _HEAD_LOOK_AMPLITUDE_RADIANS * math.sin(phase * 0.7)
    arm_swing = _ARM_SWING_AMPLITUDE_RADIANS * math.sin(phase)

    if _UPPER_BODY_BONE in base_matrices:
        node, base = base_matrices[_UPPER_BODY_BONE]
        node.transform = Transform.from_raw_matrix(
            base @ _y_rotation(sway) @ _x_rotation(breath),
        )
    if _HEAD_BONE in base_matrices:
        node, base = base_matrices[_HEAD_BONE]
        node.transform = Transform.from_raw_matrix(base @ _y_rotation(head_turn))
    # Arms swing in opposite phase to body sway — natural counter-motion.
    if _SHOULDER_L_BONE in base_matrices:
        node, base = base_matrices[_SHOULDER_L_BONE]
        node.transform = Transform.from_raw_matrix(base @ _x_rotation(-arm_swing))
    if _SHOULDER_R_BONE in base_matrices:
        node, base = base_matrices[_SHOULDER_R_BONE]
        node.transform = Transform.from_raw_matrix(base @ _x_rotation(arm_swing))


def _y_rotation(angle: float) -> np.ndarray:
    """Right-handed rotation around the Y axis (yaw)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[ c, 0, s, 0],
         [ 0, 1, 0, 0],
         [-s, 0, c, 0],
         [ 0, 0, 0, 1]],
        dtype=np.float32,
    )


def _x_rotation(angle: float) -> np.ndarray:
    """Right-handed rotation around the X axis (pitch)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[1, 0,  0, 0],
         [0, c, -s, 0],
         [0, s,  c, 0],
         [0, 0,  0, 1]],
        dtype=np.float32,
    )


def _save_strip(frames: list[np.ndarray], output: Path) -> None:
    """Compose ``frames`` as a horizontal strip; each frame downscales to
    ``_STRIP_FRAME_WIDTH × _STRIP_FRAME_HEIGHT`` so the result stays
    preview-friendly even with many frames."""
    scaled = [
        Image.fromarray(frame, "RGBA").resize(
            (_STRIP_FRAME_WIDTH, _STRIP_FRAME_HEIGHT), Image.LANCZOS,
        )
        for frame in frames
    ]
    canvas = Image.new(
        "RGBA",
        (_STRIP_FRAME_WIDTH * len(scaled), _STRIP_FRAME_HEIGHT),
        (24, 28, 34, 255),
    )
    for index, image in enumerate(scaled):
        canvas.paste(image, (index * _STRIP_FRAME_WIDTH, 0))
    canvas.save(output)


if __name__ == "__main__":
    sys.exit(main())
