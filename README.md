# PoseCascade

> **Languages**: **English** · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md)
> **Documentation**: <https://posecascade.readthedocs.io/en/latest/>

A PySide6 + OpenGL desktop engine for importing 3D models and driving
them with sandboxed scripts. The visual target is MMD: toon shading
with crisp inverted-hull outlines, PMX materials with sphere maps,
VMD-style animation curves, IK + foot planting, morph targets, and a
PBD cloth solver fast enough to drape several pieces in real time.

## Features

- **Multi-format import**: glTF / GLB, OBJ + MTL, FBX, STL, PLY,
  USD / USDZ, COLLADA, and the MMD pair (PMX models + VMD motions +
  VPD poses), each behind its own plugin adapter so adding a format
  doesn't touch the renderer.
- **MMD-flavoured forward renderer**: toon ramp (NEAREST + clamp for
  crisp cel bands), sphere-map composite, inverted-hull outlines,
  procedural checkered ground, projected ground shadow, depth-mapped
  PCF-softened self-shadow, sRGB-aware output, gradient-sky pass,
  multi-light HighDef setup (1 primary + up to 3 secondary), opt-in
  dual-quaternion skinning for joint-volume preservation, default
  AutoLuminous bloom, MMD tone-curve post-process effect, procedural
  dance-stage abstraction (floor + back wall + side walls), and a
  selection-overlay pass that re-outlines the picked top-level holder
  in a bright contrast colour. See
  [`docs/rendering_pipeline.md`](https://posecascade.readthedocs.io/en/latest/rendering_pipeline.html)
  for the full pass order + per-pass toggles.
- **VMD-driven animation**: per-bone / per-morph / per-camera tracks
  with the four-control-point bezier interpolation MMD uses, IK
  solver, foot planter, external-parent bindings between slots,
  display-frame groups, and physics chains.
- **Cloth solver**: position-based dynamics with structural + bend
  constraints, sphere / capsule colliders (with continuous-collision
  sweep), and a Cython kernel that puts the per-step cost on a
  480-vert skirt around **0.35 ms** — ~9× faster than the pure-NumPy
  fallback the same kernel transparently degrades to when the
  compiled extension isn't built.
- **Declarative animation runtime**: drive a character from a JSON
  document instead of Python — phases, gaits, body trajectories,
  morph timelines, an inline expression DSL, plus an `extends`
  profile-inheritance mechanism and array shorthands (`[from, to]`
  curves, `[x, y, z]` translation, `x` / `y` / `z` bone axis aliases)
  so a typical authoring file is a third the size of one written
  out longhand. See
  [`docs/declarative_animation.md`](https://posecascade.readthedocs.io/en/latest/declarative_animation.html).
- **In-editor animation editor** (new): two right-column docks
  bound to one shared document — a JSON editor with syntax
  highlighting, line-number gutter, inline error marks, format
  button, and dirty indicator; and a phase-blocks dock with a
  horizontal timeline (drag-to-reorder + drag-edge-to-resize),
  vertical phase cards, and an inline form covering every common
  field (name / duration / blends / pose / gait / body translation /
  bones / morphs). Ctrl+Z / Ctrl+Y undo/redo across both views.
  See
  [`docs/animation_editor.md`](https://posecascade.readthedocs.io/en/latest/animation_editor.html).
- **GPU compute skinning** (new): an OpenGL 4.3 compute-shader path
  for `passive_skin_deform` cloth pieces does LBS + collider push +
  world-to-local on the GPU, writing directly into the mesh's
  position + normal VBOs. Drops per-frame cloth + apply_cloth from
  ~9 ms to under 0.05 ms on a 30 k-vert body mesh. Transparently
  falls back to the CPU LBS path on contexts that lack 4.3 / compute.
- **Sandboxed Python scripts**: a restricted namespace (no `open`,
  `os`, `subprocess`, `__import__`, …) exposes `scene`, `nodes`,
  `time`, `input`, and a curated math layer so users can pose,
  keyframe, and animate without touching engine internals.
- **MCP server**: a Model Context Protocol server lets any MCP-aware
  LLM agent drive the engine — list and validate declarative animation
  scripts, inspect models, and benchmark the cloth solver. See
  [`docs/mcp.md`](https://posecascade.readthedocs.io/en/latest/mcp.html).

## Quick start

```bash
# Clone + create a virtualenv
git clone https://github.com/JeffreyChen-s-Utils/PoseCascade.git
cd PoseCascade
python -m venv .venv
.venv\Scripts\activate.ps1            # Windows PowerShell
# source .venv/bin/activate           # Linux / macOS

# Install with the AI extras so the MCP server is available too
pip install -e .[dev,ai]
```

The editable install compiles the Cython cloth kernel in place. If
you don't have a C compiler (Microsoft Build Tools on Windows, gcc
or clang on Linux / macOS), the install prints a warning and the
engine transparently uses the NumPy fallback path.

Launch the editor against a bundled example:

```bash
# Classic 3D-model demo reel (30s): intro idle → 360° turntable →
# walk-in-place → wave → V-pose → hip-pop → bow → return to neutral.
python -m posecascade --scene examples/assets/herta/herta.glb \
                       --script examples/scripts/showcase.json

# In-place walking + arm-swing loop (4s).
python -m posecascade --scene examples/assets/herta/herta.glb \
                       --script examples/scripts/walk.json

# Minimal breathing-idle loop (4s).
python -m posecascade --scene examples/assets/herta/herta.glb \
                       --script examples/scripts/idle.json
```

Or render the MMD-style hero frame headlessly — no editor window
required, useful as a smoke check for the full visual stack:

```bash
# Loads herta.glb (glTF), enables every MMD-fluence toggle on the
# force_toon path, writes mmd_demo.png next to the script.
python examples/mmd_demo.py

# Loads the bundled March 7th PMX via the renderer's PMX-native path
# (per-mesh MMDMaterial + sphere textures + edge flag straight from the
# file). The asset is third-party — see examples/assets/march7th/NOTICE.md
# for attribution. Writes march7th_pmx_demo.png next to the script.
python examples/march7th_pmx_demo.py

# Also works through the interactive editor:
python -m posecascade --scene examples/assets/march7th/march7th.pmx
```

A set of side-by-side comparison scripts demonstrates each
MMD-fluence feature against its baseline — each writes a labelled
PNG so the visual difference is reproducible:

```bash
python examples/compare_bloom.py    # bloom OFF vs AutoLuminous applied
python examples/compare_tone.py     # sRGB only vs + mmd_tone
python examples/compare_dqs.py      # LBS candy-wrapper vs DQS at extreme twist
python examples/compare_lights.py   # primary only vs + HighDef rim + fill
```

## Documentation

The Sphinx tree at <https://posecascade.readthedocs.io/en/latest/>
covers:

- [Rendering pipeline](https://posecascade.readthedocs.io/en/latest/rendering_pipeline.html)
  — what each render pass does and the toggles that turn it on / off.
- [Declarative animation](https://posecascade.readthedocs.io/en/latest/declarative_animation.html)
  — authoring JSON animations: phases, gaits, curves, the expression
  DSL, `extends` profile inheritance, and the array shorthands.
- [Animation editor](https://posecascade.readthedocs.io/en/latest/animation_editor.html)
  — using the in-editor JSON dock + phase blocks dock with timeline
  + curve editor + undo/redo.
- [MCP server](https://posecascade.readthedocs.io/en/latest/mcp.html)
  — the Model Context Protocol server surface (tools, parameters,
  return shapes).
- [Packaging with PyInstaller](https://posecascade.readthedocs.io/en/latest/packaging_pyinstaller.html)
  — shipping PoseCascade as a standalone executable with PyInstaller.
- [Packaging with Nuitka](https://posecascade.readthedocs.io/en/latest/packaging_nuitka.html)
  — same again with Nuitka (compiled binary, smaller / faster,
  longer build).

For contributor / maintainer notes (dev workflow, CI workflows,
release pipeline, performance numbers), see
[`DEVELOPMENT.md`](DEVELOPMENT.md).

## License

See [`LICENSE`](LICENSE) for the project's MIT-style terms. Bundled
assets carry their own licenses — `examples/assets/herta/herta.glb`
ships under CC-BY 4.0 (uploader X9_YT on Sketchfab; character
"The Herta" © HoYoverse, used under their Fan Content Guidelines —
see `examples/assets/herta/NOTICE.md` for the full notice).
The MMD demo `examples/assets/march7th/march7th.pmx` is separately
licensed CC-BY 4.0 (uploader Gregman; character "March 7th" ©
HoYoverse) — see `examples/assets/march7th/NOTICE.md`.
