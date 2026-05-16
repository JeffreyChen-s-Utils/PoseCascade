"""Micro-benchmark — measure renderer.draw + cloth tick over N frames.

Run with::

    .venv\\Scripts\\python.exe tools\\bench_renderer.py [--frames 60]

Loads the bundled Herta scene plus the showcase declarative animation,
drives the engine for ``--frames`` simulated frames at 60 FPS, and
prints the per-frame breakdown from ``frame_section`` plus total
wall-clock. Useful for spotting regressions before pushing renderer
changes. Not part of the test suite — runs against a live GL context
and prints to stdout, not a pytest fixture.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "importers"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bench_renderer")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument(
        "--script", type=Path,
        default=_REPO_ROOT / "examples" / "scripts" / "showcase.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from examples._demo_lib import (  # noqa: PLC0415
        load_character,
        make_renderer,
        setup_offscreen_gl,
    )
    from posecascade.animation.cloth_host import ClothHost  # noqa: PLC0415
    from posecascade.render.camera import Camera  # noqa: PLC0415
    from posecascade.scripting.declarative import (  # noqa: PLC0415
        DeclarativeRuntime,
        parse_animation,
    )
    from posecascade.utils.math3d import vec3  # noqa: PLC0415
    from posecascade.utils.profiling import current_stats, frame_section  # noqa: PLC0415

    print(f"booting GL context @ {args.width}x{args.height}")
    _surface, _context, _fbo, _color = setup_offscreen_gl(args.width, args.height)
    scene = load_character()
    renderer = make_renderer()
    renderer.populate_from_scene(scene)
    renderer.set_self_shadow_enabled(True)

    cloth_host = ClothHost()
    cloth_host.install_default_forces()
    cloth_host.register_imported_scene(scene)

    # Drive the scene with the showcase.json declarative animation so we
    # get a realistic per-frame load on the JSON evaluator too.
    import json  # noqa: PLC0415
    sim_clock = {"t": 0.0}
    runtime: DeclarativeRuntime | None = None
    if args.script.is_file():
        document = json.loads(args.script.read_text(encoding="utf-8"))
        parsed = parse_animation(document)
        runtime = DeclarativeRuntime(
            animation=parsed,
            scene=scene.scene,
            time=lambda: sim_clock["t"],
            cloth_host=cloth_host,
            source_dir=args.script.parent,
        )
        hooks = runtime.hooks()
        hooks["start"]()
    camera = Camera(position=vec3(0.6, 0.9, 1.8), target=vec3(0.0, 0.55, 0.0))

    dt = 1.0 / 60.0
    totals: dict[str, float] = defaultdict(float)
    # Warm-up — first frame triggers a lot of lazy allocation (shader compile,
    # bone caches, GPU SSBO allocation). Skip it for a fair average.
    cloth_host.tick(dt)
    renderer.apply_cloth_state(cloth_host)
    renderer.draw(scene.scene, camera, (args.width, args.height))

    wall_start = time.perf_counter_ns()
    for i in range(args.frames):
        current_stats().reset()
        if runtime is not None:
            sim_clock["t"] = (i + 1) * dt
            with frame_section("declarative.update"):
                runtime.hooks()["update"](dt)
        cloth_host.tick(dt)
        renderer.apply_cloth_state(cloth_host)
        renderer.draw(scene.scene, camera, (args.width, args.height))
        stats = current_stats().sections
        for k, v in stats.items():
            totals[k] += v
    wall_ms = (time.perf_counter_ns() - wall_start) / 1.0e6

    print(f"\n{args.frames} frames, wall = {wall_ms:.1f} ms "
          f"({wall_ms / args.frames:.2f} ms/frame, "
          f"{1000.0 * args.frames / wall_ms:.1f} FPS)\n")
    rows = sorted(totals.items(), key=lambda x: -x[1])
    name_w = max(len(k) for k in totals)
    for name, total in rows:
        avg = total / args.frames
        print(f"  {name:<{name_w}}  {avg:6.2f} ms/frame  (sum {total:8.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
