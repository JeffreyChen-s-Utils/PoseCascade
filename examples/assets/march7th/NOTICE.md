# March 7th (Honkai: Star Rail) PMX — Demo Asset Notice

`march7th.pmx` is the MMD (MikuMikuDance) `.pmx`-format version of
the March 7th character. It is bundled with this repository to exercise
the PMX importer + MMD-style rendering path (sphere maps, toon ramps,
edge outlines) on a real asset, as a counterpart to the glTF demo
character in `examples/assets/herta/`.

The matching textures live under `examples/assets/march7th/textures/`.

## Source

- **Title**: "March 7th - Honkai StarRail"
- **Uploader**: Gregman (Sketchfab)
- **URL**: <https://sketchfab.com/3d-models/march-7th-honkai-starrail-c4071fa4e5f1462fa478417d1aea1130>
- **Sketchfab license**: [CC Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Format here**: PMX (MikuMikuDance binary), converted from the
  original FBX via Blender + MMD Tools. Texture references rewritten to
  point at the sibling `textures/` directory.

## Underlying IP — important

The 3D model is a derivative work of HoYoverse's character "March 7th"
from the game *Honkai: Star Rail*. The Sketchfab uploader's CC-BY tag
covers the uploader's contribution; it does **not** transfer the
underlying character / brand IP, which remains with HoYoverse.

Use of this asset in this repository is conditioned on HoYoverse's
[Fan Content Guidelines](https://hoyoverse.com/en-us/about/fan-content-guidelines):

- **Permitted here**: non-commercial demonstration of the renderer in
  an open-source engineering project, with prominent attribution
  (this file + `examples/assets/ATTRIBUTIONS.md`).
- **NOT permitted**: redistribution as a commercial product, sale,
  use in NFTs, or use in any application that monetises HoYoverse's
  characters or brand without an explicit licence from HoYoverse.

If you fork this repository and your fork becomes commercial (sold,
SaaS, etc.), **delete `examples/assets/march7th/` from your fork**
before publishing and supply a substitute MMD model.

## Attribution required (per CC-BY)

When reusing the asset elsewhere, the credit line is:

> "March 7th - Honkai StarRail" by Gregman, licensed CC-BY 4.0 via
> Sketchfab. Character © HoYoverse.

## Removal

If HoYoverse or the original uploader requests removal, delete this
folder entirely. With the folder absent, `examples/march7th_pmx_demo.py`
exits cleanly with a message pointing at this NOTICE.
