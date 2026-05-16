# The Herta (Honkai: Star Rail) — Demo Asset Notice

`herta.glb` is the default demo character for the renderer. It exists
to give the renderer a visually rich anime model — long flowing dress
rigged into the body mesh, witch hat, hair — so the skinning,
toon / outline, and walk-cycle pipelines have something interesting to
work on without forcing every clone to download a model out-of-band.

## Source

- **Title**: "Honkai: Star Rail - The Herta"
- **Uploader**: X9_YT (Sketchfab)
- **URL**: <https://sketchfab.com/3d-models/honkai-star-rail-the-herta-c30f5dfa6ec04dd9ab2f9c8f2f9f6418>
- **Sketchfab license**: [CC Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Format here**: glTF binary (`.glb`), exported via Blender's glTF
  exporter from the original FBX Sketchfab serves. The Sketchfab
  promotional billboard mesh and wrapper empties have been stripped,
  the armature was rotated upright (Z-up in Blender → Y-up in glTF)
  and the transform baked, so the character stands along world Y when
  loaded. The original skinning weights and bone hierarchy are
  preserved.

## Underlying IP — important

The 3D model is a derivative work of HoYoverse's character "The Herta"
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
SaaS, etc.), **delete `examples/assets/herta/` from your fork**
before publishing and supply a substitute model (any CC-BY humanoid
glTF whose joint names match the canonical aliases — VRoid, Mixamo,
MMD all work).

## Attribution required (per CC-BY)

When reusing the asset elsewhere, the credit line is:

> "Honkai: Star Rail - The Herta" by X9_YT, licensed CC-BY 4.0 via
> Sketchfab. Character © HoYoverse.

## Removal

If HoYoverse or the original uploader requests removal, delete this
folder entirely. The demos require `herta.glb` to load; with the
folder absent, `examples/mmd_demo.py` raises a clear file-not-found at
startup pointing at this NOTICE.
