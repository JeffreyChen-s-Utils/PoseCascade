# PoseCascade

> **Languages**: **English** · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md)
> **Documentation**: [Read the Docs source](docs/) (Sphinx)

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
  [`docs/rendering_pipeline.md`](docs/rendering_pipeline.md) for the
  full pass order + per-pass toggles.
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
  morph timelines, and an inline expression DSL.
  See [`docs/declarative_animation.md`](docs/declarative_animation.md).
- **Sandboxed Python scripts**: a restricted namespace (no `open`,
  `os`, `subprocess`, `__import__`, …) exposes `scene`, `nodes`,
  `time`, `input`, and a curated math layer so users can pose,
  keyframe, and animate without touching engine internals.
- **MCP server**: a Model Context Protocol server lets any MCP-aware
  LLM agent drive the engine — list and validate declarative animation
  scripts, inspect models, and benchmark the cloth solver. See
  [`docs/mcp.md`](docs/mcp.md).

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

## Project layout

```
PoseCascade/
├── posecascade/                  # main package
│   ├── animation/                # cloth, skin, morphs, IK, VMD tracks
│   │   ├── cloth.py              # PBD solver (Python orchestration)
│   │   └── _cloth_kernels.pyx    # Cython inner loop (built by setup.py)
│   ├── app/                      # QApplication bootstrap, main window
│   ├── assets/                   # cache, path safety, importer manager
│   ├── gl/                       # GL context, shaders, framebuffers
│   ├── mcp/                      # Model Context Protocol server
│   ├── render/                   # render graph, materials, lights
│   ├── scene/                    # scene graph, transforms, components
│   ├── scripting/                # sandboxed script host + declarative runtime
│   └── ui/                       # viewport, outliner, inspector, timeline
├── importers/<format>/           # per-format importer plugins
├── shaders/                      # GLSL by render pass
├── examples/                     # bundled models + animation scripts
├── tests/                        # pytest suite mirroring the package
├── docs/                         # design + integration docs
├── schemas/                      # JSON schemas (declarative animation)
├── setup.py                      # cythonize build hook
└── pyproject.toml                # project metadata + ruff / bandit config
```

## Development

The Definition of Done (see [`CLAUDE.md`](CLAUDE.md)) requires every
change to satisfy three gates before commit. Reproduce locally:

```bash
.venv/Scripts/python.exe -m pytest tests/             # unit + offscreen-GL tests
.venv/Scripts/python.exe -m ruff check .              # lint + style
.venv/Scripts/python.exe -m bandit -c pyproject.toml -r posecascade/
```

The cloth Cython kernel needs an in-place build whenever the `.pyx`
source changes:

```bash
.venv/Scripts/python.exe setup.py build_ext --inplace
```

For distribution, the `[tool.cibuildwheel]` section in
`pyproject.toml` produces pre-built wheels across Win / macOS / Linux
× supported Python versions; `.github/workflows/wheels.yml` drives
that on tag pushes.

## Visual pipeline

The forward renderer runs six passes per frame in fixed order — depth-
map shadow pass, scene, ground, projected shadow, selection overlay,
post-process effect chain. Each pass has a toggle (`set_ground_enabled`,
`set_self_shadow_enabled`, `set_projected_shadow_enabled`,
`set_selected_holder`) so smoke tests and headless renders can opt
out without losing pixel fidelity. The full breakdown — pass order,
shader files, light-space math, texture units, MMD-fluence gaps —
lives in [`docs/rendering_pipeline.md`](docs/rendering_pipeline.md).

## Performance notes

The cloth solver has been the recent focus. On the 480-vert skirt
benchmark in `posecascade.mcp.server.cloth_benchmark`:

| Stage                                  | ms/step (best) | vs baseline |
|----------------------------------------|---------------:|------------:|
| Baseline (pre-tuning)                  |          3.225 |           — |
| NumPy: einsum + combined-bincount      |          2.085 |        −35% |
| Cython kernel                          |          0.356 |    **−89%** |
| Cython + broad-phase + bin culling     |          0.36–0.38 (single-bin colliders save 30%) | — |

Per-frame timing for the renderer hot path is wrapped in
`posecascade.utils.profiling.frame_section` so a UI overlay (or a
custom test) can pull a per-frame breakdown out of
`current_stats().sections`.

## License

See [`LICENSE`](LICENSE) for the project's MIT-style terms. Bundled
assets carry their own licenses — `examples/assets/herta/herta.glb`
ships under CC-BY 4.0 (uploader X9_YT on Sketchfab; character
"The Herta" © HoYoverse, used under their Fan Content Guidelines —
see `examples/assets/herta/NOTICE.md` for the full notice).
The MMD demo `examples/assets/march7th/march7th.pmx` is separately
licensed CC-BY 4.0 (uploader Gregman; character "March 7th" ©
HoYoverse) — see `examples/assets/march7th/NOTICE.md`.
