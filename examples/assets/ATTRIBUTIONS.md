# Asset attributions

The bundled demo assets in this directory are licensed and credited as
follows. New entries SHOULD be added when a third-party asset replaces
or augments anything here.

## herta/herta.glb (default glTF demo character)

- **Source**: "Honkai: Star Rail - The Herta" by **X9_YT** on Sketchfab.
- **URL**: <https://sketchfab.com/3d-models/honkai-star-rail-the-herta-c30f5dfa6ec04dd9ab2f9c8f2f9f6418>
- **Sketchfab license**: [CC Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Underlying IP**: Character © HoYoverse, used under their
  [Fan Content Guidelines](https://hoyoverse.com/en-us/about/fan-content-guidelines).
  Non-commercial use only — see `examples/assets/herta/NOTICE.md`.
- **Use here**: Default model for `examples/mmd_demo.py`, the
  comparison scripts (`examples/compare_*.py`), and the marquee
  showcase in `examples/scene_compose.py`.
- **Attribution required**: "Honkai: Star Rail - The Herta" by X9_YT,
  CC-BY 4.0. Character © HoYoverse.

## march7th/march7th.pmx (MMD PMX demo character)

- **Source**: "March 7th - Honkai StarRail" by **Gregman** on Sketchfab.
- **URL**: <https://sketchfab.com/3d-models/march-7th-honkai-starrail-c4071fa4e5f1462fa478417d1aea1130>
- **Sketchfab license**: [CC Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Underlying IP**: Character © HoYoverse, used under their
  [Fan Content Guidelines](https://hoyoverse.com/en-us/about/fan-content-guidelines).
  Non-commercial use only — see `examples/assets/march7th/NOTICE.md`.
- **Use here**: PMX importer / MMD-style rendering demo
  (`examples/march7th_pmx_demo.py`).
- **Attribution required**: "March 7th - Honkai StarRail" by Gregman,
  CC-BY 4.0. Character © HoYoverse.

All script-driven demos under `examples/scripts/` resolve bone names
through the canonical alias layer (`posecascade/animation/bone_aliasing.py`
— `head`, `upper_arm_L`, `upper_leg_L`, …) so they target each rig's
skeleton without per-asset edits.
