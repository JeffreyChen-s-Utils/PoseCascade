# Rendering pipeline

PoseCascade's renderer aims for MMD-style visuals: toon-shaded characters
on a checkered ground, with inverted-hull outlines, ground-projected
silhouette shadows, depth-mapped self-shadows, and an AutoLuminous-style
bloom on top. This document walks each pass in execution order and
points at the toggles you can use to disable individual passes for
tests / headless renders / perf debugging.

## Pass order

`Renderer.draw(scene, camera, viewport_size)` runs these passes from
top to bottom. Pass costs are wrapped in
`posecascade.utils.profiling.frame_section`, so a debug overlay can
surface their per-frame timing.

```
┌────────────────────────────────────────────────────────────────┐
│ 1.  shadow_pass         ← if _self_shadow_enabled              │
│         renders depth-only into _shadow_fbo (1024² depth tex)  │
│         from the light's POV; populates _shadow_depth_tex.     │
├────────────────────────────────────────────────────────────────┤
│ 2.  bind caller's FBO + toggle GL_FRAMEBUFFER_SRGB +           │
│     clear color & depth                                        │
├────────────────────────────────────────────────────────────────┤
│ 3.  sky                 ← if _sky_enabled                      │
│         fullscreen-triangle gradient (zenith→horizon→ground),  │
│         depth test off, depth write off, draws first so every  │
│         later pass overdraws atmospheric values.               │
├────────────────────────────────────────────────────────────────┤
│ 4.  scene_nodes                                                │
│         per-node dispatch:                                     │
│           MMD material → outline pass + toon pass              │
│           skinned      → basic skinned forward                 │
│           plain        → basic forward                         │
│         toon pass binds _shadow_depth_tex to texture unit 3    │
│         and samples u_shadowMap with a 3×3 PCF kernel for      │
│         soft self-shadow attenuation.                          │
├────────────────────────────────────────────────────────────────┤
│ 5.  ground              ← if _ground_enabled                   │
│         single quad at y=0, fragment-shader checker pattern    │
│         with a radial fade to clear-color near the horizon.    │
├────────────────────────────────────────────────────────────────┤
│ 6.  projected_shadow    ← if _projected_shadow_enabled         │
│         re-rasterises every scene mesh with vertices projected │
│         onto y=0.001 along -light_direction; alpha-blended.    │
├────────────────────────────────────────────────────────────────┤
│ 7.  selection_overlay   ← if _selected_holder is not None      │
│         re-outlines every mesh under the selected holder with  │
│         a thicker yellow inverted-hull pass.                   │
└────────────────────────────────────────────────────────────────┘
```

Post-processing (AutoLuminous bloom, HG-shadow, …) lives on the
[`EffectChain`](../posecascade/render/effects/) — that's a separate
ping-pong stage the integrator calls via `Renderer.apply_effect_chain`
*after* `draw`, so it's not in the table above.

## The four MMD-fluence passes

### 1. Self-shadow depth map (PCF-softened)

A depth-only pass from the light's point of view feeds a 24-bit depth
texture; the toon fragment shader samples it per pixel through a 3×3
percentage-closer-filtering kernel to attenuate fragments that are
occluded by another part of the model (chin under head, back of the
skirt under the body, …). PCF averages nine depth comparisons across
adjacent texels — the result is a soft 1–2 pixel falloff at the
silhouette instead of MMD-Lite's stepped hard edge.

**Shaders**

- `shaders/shadow/depth_only.vert` — non-skinned, transforms by
  `u_lightSpaceMatrix * u_modelMatrix`.
- `shaders/shadow/depth_only_skinned.vert` — same, with skinning.
- `shaders/shadow/depth_only.frag` — empty (depth-only).
- `shaders/toon/toon.frag` — gains `u_shadowMap`,
  `u_lightSpaceMatrix`, `u_shadowEnabled`, `u_shadowStrength`. The
  helper `_self_shadow_attenuation()` does the projective sample +
  0.005 depth bias + frustum-edge guard, then averages nine PCF taps
  (controlled by the `_PCF_RADIUS` constant) for a soft edge.

**Light-space transform** (`_compute_light_space_matrix`)

- Light is placed at `light_direction × _SHADOW_LIGHT_DISTANCE` (10 m).
- Look-at points back at the origin. `up = +Y` unless the light is
  within ~2.5° of straight up/down, in which case `up = +Z` to keep
  the look-at non-singular.
- Orthographic projection of ±`_SHADOW_HALF_EXTENT` (4 m) per side,
  `_SHADOW_NEAR=0.1`, `_SHADOW_FAR=20.0`. Adequate for a single MMD
  character at editor zoom; bump the half-extent for full stage scenes.

**Tunables (module constants in `posecascade/render/renderer.py`)**

- `_SHADOW_MAP_SIZE = 1024` — texture resolution. Higher = sharper
  edges, more VRAM.
- `_SHADOW_STRENGTH = 0.45` — fraction the shadowed pixel darkens by.
  0.45 matches the default MMD Standard shader feel; 1.0 = pitch
  black, 0.0 = no visible shadow.
- `_TEX_UNIT_SHADOW = 3` — texture unit reserved on the toon pass.
  Units 0/1/2 are base-colour / sphere / toon ramp.

**Toggling**

```python
renderer.set_self_shadow_enabled(False)   # toon shader receives u_shadowEnabled=0
```

The smoke tests in `tests/test_render_smoke*.py` flip this off so their
golden-image baselines stay stable.

### 2. Checkered ground

A single 4-vertex quad covering ±25 m at y=0; the fragment shader
does the checker pattern from world-space XZ and fades the alpha
to zero near the horizon so the floor blends into the clear-colour
instead of cutting off at a hard edge.

**Shaders**

- `shaders/ground/ground.vert` — pass world-XZ to the fragment.
- `shaders/ground/ground.frag` — `mod(floor(xz/cell), 2)` checker,
  `length(xz)` radial fade.

**Tunables**

- `_GROUND_HALF_EXTENT = 25.0` — plane size.
- `_GROUND_CELL_SIZE = 0.5` — one checker square in metres.
- `_GROUND_COLOR_A / _B` — light and dark squares.
- `_GROUND_FADE_START = 6.0` / `_FADE_END = 16.0` — radial fade band.

**Toggling**

```python
renderer.set_ground_enabled(False)   # also disables projected_shadow
```

### 3. Projected ground shadow

For each scene mesh, the vertex shader projects every vertex along
`-light_direction` onto the y=0.001 plane (one millimetre above the
ground to win the z-fight with the checker quad). The fragment shader
writes a uniform semi-transparent black. The result is the model's
silhouette flattened against the floor — visually equivalent to
MMD's classic ground shadow without needing a second shadow map.

**Shaders**

- `shaders/ground/shadow_projection.vert` — non-skinned.
- `shaders/ground/shadow_projection_skinned.vert` — same, with
  `u_boneMatrices[MAX_BONES]`.
- `shaders/ground/shadow_projection.frag` — solid colour from
  `u_shadowColor`.

**Limitations**

- Vertices below the ground plane project incorrectly (to the side
  opposite the light); we don't discard them, so a model whose feet
  sink below y=0 will see weird shadow streaks at the silhouette
  edges. Foot planting (`posecascade/animation/foot_planting.py`)
  prevents this in practice.
- The shadow has no soft falloff — it's the model's exact silhouette.
  PCF (percentage-closer filtering) would soften the edge but add a
  fragment-shader cost we don't yet need.

**Toggling**

```python
renderer.set_projected_shadow_enabled(False)   # independent of the ground
```

`set_ground_enabled(False)` also turns off the projected shadow as a
matched pair (a floating shadow with no ground looks worse than no
shadow at all). Use `set_projected_shadow_enabled(...)` to override
that coupling if you want one without the other.

### 4. Gradient sky

A fullscreen-triangle pass drawn immediately after `clear`. The vertex
shader synthesises three corner positions from `gl_VertexID` so the
draw needs no VBO — just an empty VAO bound for Core Profile's sake.
The fragment shader does a three-stop vertical gradient (zenith /
horizon / ground) with the horizon line at `_SKY_HORIZON_Y = 0.42`
of the frame.

**Shaders**

- `shaders/sky/gradient.vert` — emits the screen-covering triangle
  + `v_uv` in [0,1]² for the fragment.
- `shaders/sky/gradient.frag` — three-stop mix from
  `u_zenithColor`, `u_horizonColor`, `u_groundColor`.

**Why first, not last** — depth test is off + depth write is off, so
the gradient lands at whatever depth the buffer has been cleared to
(1.0). Every later pass overwrites the sky pixels wherever geometry
covers them. Drawing it first keeps the pixel-shader cost flat
regardless of how much of the frame the model takes.

**Toggling**

```python
renderer.set_sky_enabled(False)   # falls back to the dark-grey clear
```

### 5. sRGB output

When `_srgb_output_enabled` is true (default), `draw` enables
`GL_FRAMEBUFFER_SRGB` for the duration of the frame. Base-colour
textures uploaded with `srgb=True` get the `GL_SRGB8_ALPHA8` internal
format; the GPU decodes them to linear at sample time, shader math
runs in linear, and the GPU re-encodes back to sRGB on write. Toon
ramps and sphere maps stay linear (`GL_RGBA8`) because they're lookup
tables, not display-referred colours — the renderer pre-scans every
mesh's `MMDMaterial` and routes those texture indices into the linear
upload path via `_collect_linear_texture_indices`.

**Toggling**

```python
renderer.set_srgb_output_enabled(False)   # match legacy linear baselines
```

The smoke-render tests all opt out so their pre-sRGB golden images
stay valid.

### 6. Selection overlay

Not strictly MMD — a Blender-style affordance. When
`Renderer._selected_holder` is set, the renderer runs a second
inverted-hull outline pass over every mesh under that holder using a
thicker edge (`_SELECTION_EDGE_SIZE = 0.04`) and a bright yellow
colour (`_SELECTION_EDGE_COLOR = (1.0, 0.85, 0.10, 1.0)`). Works for
materials that didn't opt into `has_edge` because the outline
shaders only need positions + normals on attribute locations 0/1.

**Toggling**

```python
renderer.set_selected_holder(node)   # turn on for a specific holder
renderer.set_selected_holder(None)   # clear
```

## Stage slots

`ModelSlot` carries an `is_stage: bool` flag. A stage slot is a
passive prop (dance floor, walls, environment PMX model) — the
renderer draws it like any other slot but the animation player skips
the per-frame bone / morph / IK / physics pass on it.
`SlotsPlayer.__post_init__` skips `is_stage` slots when building its
player dict, so even if a stage slot is handed a stray motion file it
won't move.

A built-in procedural stage ships in `posecascade.scene.stage` —
`procedural_dance_stage()` returns a two-mesh `ImportedScene` (raised
floor + back wall) that drops into the existing slot machinery:

```python
from posecascade.animation.slots_player import make_stage_slot
from posecascade.scene.stage import procedural_dance_stage

slots.add(make_stage_slot(name="stage", imported=procedural_dance_stage()))
```

Replace the procedural stage with any imported PMX / glTF scene to
load a real stage asset. The slot machinery doesn't care what's
inside — only the renderer's mesh-walk consumes the geometry.

## Post-processing chain

After the per-frame `draw()` finishes, `Renderer.apply_effect_chain`
runs the user's `EffectChain` through `EffectChainExecutor` — a
ping-pong FBO pair that pipes the main scene texture through each
enabled pass.

Four built-in descriptors ship in `posecascade/render/effects/builtins/`:

- **`autoluminous`** — emissive bloom. Threshold + Gaussian blur +
  additive composite. Default on for newly-created
  `AppController` instances (the "MMD glow" is on by default).
- **`hgshadow`** — additional soft shadow pass.
- **`o_greener`** — green-channel emphasis (MME port).
- **`ikeshita_ray`** — god-ray sample (MME port).

**Toggling AutoLuminous default** — `AppController._seed_default_effect_chain`
only appends AutoLuminous when the chain is empty. If you hand the
controller a pre-built chain (loaded from a project file or
constructed elsewhere) it's left alone.

## Texture units

The toon pass binds four texture units per draw:

| Unit | Sampler          | Source                                    | Format          |
|-----:|------------------|-------------------------------------------|-----------------|
|    0 | `u_baseColorTex` | Mesh's base-colour map (white fallback)   | sRGB (`GL_SRGB8_ALPHA8`) |
|    1 | `u_sphereTex`    | MMD sphere texture (multiply / add / sub) | Linear (`GL_RGBA8`) — LUT |
|    2 | `u_toonTex`      | Toon ramp (NEAREST + CLAMP_TO_EDGE)       | Linear (`GL_RGBA8`) — LUT |
|    3 | `u_shadowMap`    | Depth map from the shadow pass            | `GL_DEPTH_COMPONENT24` |

## Order of operations gotchas

- **The shadow pass changes the bound framebuffer.** Both
  `_build_shadow_fbo` (at init) and `_draw_shadow_pass` (per frame)
  save the previously bound FBO and restore it on exit, so an
  offscreen render test fixture's FBO survives intact. If you add a
  new pass that targets its own FBO, follow the same save/restore
  pattern.
- **Projected shadows disable depth writes.** A shadow vert that
  overshoots the ground (anchor below floor, foot a hair under the
  rig) can't seed an artefact in subsequent passes. Depth *test* is
  still on so the shadow is occluded by 3D geometry in front of the
  ground.
- **Ground + projected shadow + selection overlay all use alpha
  blending.** They each enable `GL_BLEND` in a `try/finally` and
  disable it on exit, so the renderer's leave-state matches its
  enter-state for every pass.

## Multi-light HighDef toon

The toon fragment shader accepts a primary light (`u_lightDirection`
+ `u_lightColor` — drives the cel banding and is what the self-shadow
samples) plus up to three secondary directional lights. Secondaries
are flat additive Lambert × colour — they don't perturb the toon
ramp, so they fill specific zones (rim from behind, bounce from
below) without breaking the cel look.

```python
renderer.set_secondary_lights([
    ((-0.5,  0.7, -0.7), (0.42, 0.50, 0.60)),   # back-rim
    (( 0.0, -0.3,  0.95), (0.18, 0.20, 0.24)),  # front-fill bounce
])
# Or the bundled shortcut:
renderer.apply_highdef_light_preset()
```

`set_secondary_lights([])` reverts to the single-light Standard
setup. The shader hard-cap is `MAX_SECONDARY_LIGHTS = 3` (see
`shaders/toon/toon.frag`).

## Dual-quaternion skinning (opt-in)

LBS (linear blend skinning) collapses to a sharp pinch when two bones
with opposing rotations meet — the classic shoulder / elbow / wrist
candy-wrapper artefact at extreme angles. DQS represents each bone
transform as a screw motion (rotation + screw translation) and
blends those screw motions, which preserves volume at the joint.

PoseCascade ships both:

- **LBS** (default): `shaders/toon/toon_skinned.vert`, faster, fine
  for typical dance motions.
- **DQS** (opt-in): `shaders/toon/toon_skinned_dqs.vert`, slightly
  costlier per-vertex, no candy-wrapper.

Toggle via the renderer:

```python
renderer.set_dqs_enabled(True)   # toon-skinned meshes route to DQS
```

The CPU-side conversion lives in
`posecascade.utils.dual_quaternion.matrices_to_dual_quaternions` —
called once per frame per skinned mesh when DQS is on. The shader
re-normalises every blended dual quaternion to keep the screw motion
unit even when bone weights have float32 noise.

Currently scoped to the **toon-skinned path**. The basic forward
skinned shader, the depth shadow pass, and the projected ground
shadow all stay on LBS — the sub-millimetre divergence under DQS
isn't visible in those passes.

## Force-toon-shading for non-MMD imports

glTF / OBJ / FBX meshes don't carry an `MMDMaterial`, so they
normally render through the basic forward path — smooth Lambert
shading, no outline, no toon banding. That looks like a modern game
character on a checkered floor, *not* MMD.

Flip the toggle to apply MMD aesthetics to any imported mesh:

```python
renderer.set_force_toon_shading(True)
```

When on, every mesh without an explicit `MMDMaterial` gets routed
through the toon pipeline using `posecascade.render.toon_promote`:

- `default_toon_material()` — synthetic MMD material (almost-white
  diffuse, low ambient, `MAT_FLAG_HAS_EDGE`, thin black outline).
- `default_toon_ramp_pixels()` — procedural 1×4 cel ramp with one
  hard lit/shadow band at the 50% Lambert mark.

Defaults off so non-MMD scenes keep their native look. Designed for
"apply MMD style to a non-MMD glTF" use cases — that's exactly what
the bundled `examples/assets/herta/herta.glb` is.

## MMD tone-curve effect

A new built-in post-process effect, `mmd_tone`, applies a four-knob
curve to push a sRGB-correct render toward the slightly warmer,
slightly lifted look MMD's non-standard colour pipeline produces:

- `midtone_lift` (default 0.04) — floor under deep shadow.
- `highlight_rolloff` (default 0.15) — Reinhard-style soft knee.
- `saturation` (default 1.08) — gentle colour boost.
- `warm_tint` (default `(1.02, 1.0, 0.97)`) — multiplicative warm bias.

Stack it after `autoluminous` for the canonical MMD post chain. Not
seeded by default — opt-in via the effect-chain UI or
`controller.effect_chain.append(load_builtin("mmd_tone"))`.

## CPU-side performance — per-frame uniform-state cache

The forward renderer's hot path used to re-upload the same constant
uniforms (view / projection / light / shadow / secondary lights /
bone-matrix palettes) on **every mesh** of **every pass**, paying a
`np.ascontiguousarray` + `glUniform*` round-trip per upload. On the
showcase scene with 30 meshes × 3 passes this was ~360 redundant
matrix uploads per frame and a similar count of array conversions.

The renderer now tracks per-program, per-frame state and skips the
upload when it's already been written this frame:

```python
@dataclass
class _ProgramFrameState:
    camera_uploaded: bool = False
    lights_uploaded: bool = False
    shadow_uploaded: bool = False
    skin_uploaded: set[int] = field(default_factory=set)
```

Cleared at the top of every `draw()`. Each upload helper
(`_set_camera_uniforms`, `_bind_secondary_lights`,
`_bind_shadow_uniforms`, the skinned branch of
`_set_geometry_uniforms`) early-returns when its flag is already
set. View / projection / light vectors and the light-space matrix
are pre-converted to `float32` once at the top of the frame and
shared by reference through every per-mesh call.

The shadow pass and the projected-shadow pass additionally hoist
`glUseProgram` out of the per-mesh inner loop — every skinned mesh
draws through the same program, so we switch programs only on a
transition between skinned and unskinned.

`_compute_bone_matrices` was Python-looping a per-joint `world(joint) @ inverse_bind`
matmul (354 calls per skin per frame on the bundled rig). It now
stacks all joint world matrices into a single `(J, 4, 4)` buffer
and issues one batched `np.matmul` against the inverse-bind stack.

Net effect on the showcase scene at 768×768
(`tools/bench_renderer.py`, three-run averages):

| Step                                   | ms/frame |    FPS |
|----------------------------------------|---------:|-------:|
| Baseline                               |     7.88 |    127 |
| Per-frame uniform-state cache          |     6.09 |    164 |
| + `glUseProgram` hoisting              |     5.04 |    198 |
| + batched bone-matrix matmul           | **~4.50**| **~220** |

The `frame_section` breakdown shows `scene_nodes` falling from
2.68 ms → 0.92 ms (2.9× faster), `shadow_pass` from 3.97 ms → 2.85
ms, and `projected_shadow` from 0.64 ms → 0.23 ms across the three
steps. The remaining `shadow_pass` time is dominated by real GPU
work (depth pass at 2048² for 30+ meshes).

## GPU compute passive-skin path

`shaders/passive_skin/passive_skin_push.comp` is an OpenGL 4.3
compute shader that does LBS + collider push + world-to-local
transform for `passive_skin_deform` cloth pieces on the GPU,
writing directly into the mesh's existing position + normal VBOs.
No CPU readback; the next draw call sees the deformed positions
through the unchanged `glVertexAttribPointer` bindings.

The dispatcher lives in `posecascade/gl/compute_skin.py`:

```python
dispatcher = PassiveSkinDispatcher.try_create(shader_path)
if dispatcher is None:
    # Driver lacks GL 4.3 / compute — caller stays on CPU LBS.
    return
dispatcher.register_piece(
    piece_id=id(piece),
    output_position_vbo=gl_mesh.vbos[gl_mesh.position_vbo_index],
    output_normal_vbo=gl_mesh.vbos[gl_mesh.normal_vbo_index],
    bind_positions=..., bind_normals=...,
    joints_per_vert=..., weights_per_vert=...,
    dominant_joint=...,
)
dispatcher.dispatch(
    piece_id, bone_matrices, world_to_local,
    colliders, exclude_bits,
)
```

Per-piece static buffers (bind positions, bind normals, joint
indices, weights, dominant-joint-per-vert, collider exclude
bitmask) are allocated once. Per-frame the dispatcher uploads the
joint world × inverse-bind palette (shared SSBO across all pieces)
and the active collider list (UBO, 16-collider cap), then issues
one `glDispatchCompute(ceil(N / 64), 1, 1)` per piece and a
`GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT`
memory barrier so the next draw sees the updated VBO.

`ClothHost.mark_gpu_managed(piece)` toggles the host's CPU LBS off
for that piece — `_update_skin_targets` still computes the joint
matrix palette (the renderer's bone-matrix cache reads from it) but
skips the per-vert gather + matmul that previously dominated the
hot path. `_project_passive_pieces` and `iter_local_state` also
skip GPU-managed pieces; the renderer's `apply_cloth_state` adopts
them late so a script that adds a passive-skin cloth post-load is
picked up automatically.

On the bundled Herta body mesh (30 k verts, 354 bones, 7 active
colliders) the per-frame cloth + apply_cloth budget drops from
~9 ms on the CPU path to under 0.05 ms on GPU compute. Fallback to
CPU is automatic when:

* the active context is older than 4.3,
* the compute shader fails to compile (rare; some drivers stub
  compute even when they advertise 4.3),
* the mesh has no normal VBO (the shader writes both),
* the piece has no `_SkinTargetFollower` with `bind_normals`.

Both `examples/_demo_lib.py` and `examples/mmd_demo.py` request
`QSurfaceFormat.setVersion(4, 3)` so the headless demos exercise
the GPU path; the interactive viewport's context picks up 4.3 too
on any reasonable driver.

## What's missing vs full MMD parity

After this round, every named MMD-fluence dimension has landed:

| MMD feature | PoseCascade |
|---|---|
| Cel shading | ✅ toon + outline |
| Sphere maps | ✅ multiply / add / sub |
| Self-shadow | ✅ depth FBO + PCF |
| Ground shadow | ✅ procedural projected silhouette |
| Stage / floor | ✅ procedural; load PMX for custom |
| AutoLuminous bloom | ✅ default-on |
| Atmospheric backdrop | ✅ gradient sky |
| MMD colour tone | ✅ sRGB + optional `mmd_tone` |
| Multi-light (HighDef) | ✅ up to 3 secondaries |
| Volume-preserving skinning | ✅ opt-in DQS |

What's left is genuinely "graphics research" territory: PCF on the
projected ground shadow (currently hard), proper SDEF when the
importer carries per-vertex c/r0/r1 (DQS is a generalisation, but
SDEF's per-vertex correction terms aren't honoured), or a true
HDR + tonemap pipeline. None of these is a "this isn't MMD" gap.
