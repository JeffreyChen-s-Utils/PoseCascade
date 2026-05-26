"""CLI: bake a per-pose drape snapshot from a scene + declarative script.

Usage::

    py tools/bake_drape_snapshot.py \\
        --scene examples/assets/herta/herta.glb \\
        --script examples/scripts/dog_crawl.json \\
        --pose dog_crawl \\
        --frames 360 \\
        --out examples/scripts/dog_crawl.drape.json

The script loads the scene, runs the declarative animation for
``--frames`` ticks (1/60 s each) to settle the springs / PBD into the
target pose, then writes a versioned JSON snapshot.

At runtime, reference the file from a phase's ``drape_snapshot`` key:

.. code-block:: json

    {"phases": [{
        "name": "settle_into_crawl",
        "drape_snapshot": "dog_crawl.drape.json"
    }]}

The path is resolved against the script's own directory (via
:func:`posecascade.assets.path_safety.resolve_safe`), so absolute paths
and ``..`` traversal are rejected — keep snapshots co-located with the
script JSON or in a subdirectory beneath it.

This is the same workflow HoYoverse uses to ship reliable, deterministic
hair / cloth drape for *Genshin* / *Honkai: Star Rail* extreme poses.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "importers"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path,
                        help="Path to the GLB / FBX / USD scene")
    parser.add_argument("--script", required=True, type=Path,
                        help="Path to the declarative animation JSON")
    parser.add_argument("--pose", required=True,
                        help="Pose name embedded in the snapshot")
    parser.add_argument("--frames", type=int, default=360,
                        help="Frames at 1/60 s to settle (default: 360 = 6 s)")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0,
                        help="Per-frame dt seconds (default: 1/60)")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSON path (parent dir must exist)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from posecascade.animation import drape_snapshot as ds  # noqa: PLC0415
    from posecascade.animation.cloth_host import ClothHost  # noqa: PLC0415
    from posecascade.animation.physics_host import PhysicsHost  # noqa: PLC0415
    from posecascade.assets.importer_manager import ImporterManager  # noqa: PLC0415
    from posecascade.scripting.declarative import load_animation  # noqa: PLC0415
    from posecascade.scripting.sandbox import build_api  # noqa: PLC0415

    mgr = ImporterManager(importers_root=ROOT / "importers")
    mgr.discover()
    imp = mgr.load(args.scene)

    physics = PhysicsHost()
    cloth = ClothHost()
    physics.share_colliders_with(cloth)
    physics.register_scene(imp.scene)
    physics.install_default_forces()
    physics.register_imported_scene(imp)
    cloth.register_imported_scene(imp)

    api = build_api(
        scene=imp.scene,
        time_provider=lambda: 0.0,
        physics_host=physics,
        cloth_host=cloth,
        foot_planter=None,
        skins=imp.skins,
        meshes=imp.meshes,
    )
    hooks = load_animation(
        args.script.read_text(encoding="utf-8"), str(args.script), api,
    )
    hooks["start"]()
    for _ in range(args.frames):
        hooks["update"](args.dt)
        physics.tick(args.dt)
        cloth.tick(args.dt)

    snap = ds.capture(
        physics, cloth, name=args.pose,
        settled_at_seconds=args.frames * args.dt,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.save(snap, args.out)
    print(
        f"saved {args.out}  "
        f"({len(snap.chain_states)} chains, {len(snap.cloth_states)} cloth pieces)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
