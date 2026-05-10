# Declarative animation

PoseCascade can drive a character demo from a JSON document instead
of a Python script. The runtime ships a fixed vocabulary of phase
timing, body trajectories, gait primitives, ground geometry,
physics tunings, morph curves, and an inline expression DSL — so an
authoring tool or a non-programmer user can describe the same
animations the sandbox-Python scripts cover, without writing code.

The schema is `schemas/animation_v1.json` (JSON Schema 2020-12). The
loader is `posecascade.scripting.declarative.load_animation`; the
`bootstrap.attach_script` path branches on file extension, so a
`.json` file goes through this runtime while `.py` keeps the existing
sandbox flow.

## Document shape

```json
{
  "schema_version": 1,
  "name": "stair_walk",
  "loop_sec": 16.0,
  "bpm": 120,
  "rig": { ... },
  "ground": { ... },
  "physics_chains": { ... },
  "wind": { ... },
  "phases": [ ... ]
}
```

`schema_version` is required and currently must be `1`. The runtime
rejects unknown versions to give us room to evolve the format.

`bpm` is optional. When set to a positive number, two things change:

- Each phase may use `duration_beats` instead of `duration_sec` —
  authors think in beats, the runtime resolves to seconds at parse
  time. Mixing both in one phase is rejected.
- The expression DSL gains two variables: `beat` (= `elapsed * bpm /
  60`) and `phase_beat` (= `phase_elapsed * bpm / 60`). Drop a
  `"sin(beat * tau / 4)"` into a curve to phase-lock motion to a
  4-beat cycle.

Without `bpm`, `beat` and `phase_beat` evaluate to `0` so legacy
documents keep working.

## Rig bindings

```json
"rig": {
  "character_root": "Sketchfab_model",
  "leg_chain_l": ["upper_leg_L", "lower_leg_L", "foot_L"],
  "leg_chain_r": ["upper_leg_R", "lower_leg_R", "foot_R"],
  "knee_limit_min": [-2.4, 0, 0],
  "knee_limit_max": [0.1, 0, 0],
  "body_bones": {
    "upper_arm_L": "upper_arm_L",
    "upper_arm_R": "upper_arm_R"
  }
}
```

`character_root` is the scene node that gets the body's yaw +
translation each frame. Leg chains drive the engine foot planter
(see `posecascade/animation/foot_planting.py`) and the stride gait's
lock-target IK. `body_bones` is an alias map — the runtime references
bones by logical names like `"upper_leg_L"` internally; if the rig
uses different names, point the alias at the actual bone.

## Ground

```json
"ground": {"kind": "stairs", "base_z": -0.20,
           "step_depth": 0.04, "step_rise": 0.02, "count": 5,
           "forward_sign": -1}
```

Two kinds today:

- `flat` (`y` defaults to 0).
- `stairs` (regular flight; same parameters as
  `posecascade.animation.foot_planting.stair_ground`).

If `ground` is set and the rig has both leg chains, the runtime auto-
binds the engine foot planter at start so feet stay above the surface
without scripting.

## Phases

```json
"phases": [
  {
    "name": "walk_forward",
    "duration_sec": 4.0,
    "body": { ... },
    "gait": { ... },
    "morphs": { ... }
  }
]
```

Phases run sequentially; the runtime selects one per tick from the
elapsed time modulo `loop_sec`.

### body

```json
"body": {
  "yaw_rad": "pi",
  "lean_x_rad": 0.10,
  "translation": {
    "x": 0,
    "y": 0,
    "z": {"kind": "linear", "from": 0.0, "to": -0.20}
  }
}
```

Each field accepts a *value curve* (see below). The `translation`
field also supports a stair shortcut:

```json
"translation": {
  "stair": {
    "base_z": -0.20, "rise": 0.10, "forward": 0.20,
    "step_count": 5, "ascending": true,
    "rise_window": [0.40, 0.75], "forward_sign": -1
  }
}
```

…which sweeps Y up step-by-step and Z forward over the phase, with a
cosine-eased rise inside `rise_window` of each per-stride
normalised step time.

### gait

Two kinds:

- `walking` — sinusoidal alternating leg / arm swing. Cross-body
  coordination (R leg back when L arm forward) comes free from
  the L / R amplitude flip.
- `stride` — N discrete strides with bell-windowed knee + forward
  swings, leading / trailing parity flip per step. Setting
  `lock_trailing_foot: true` (default) snapshots the trailing
  foot's world position at each step boundary and runs analytical
  2-bone IK to keep it pinned.

```json
"gait": {
  "kind": "stride",
  "step_count": 5,
  "leading_lift_rad": -0.70,
  "trailing_back_rad": 0.25,
  "knee_bend_rad": 0.30,
  "arm_swing_amplitude_rad": 0.40,
  "arm_hang_rad": -1.45,
  "knee_bell": [0.10, 0.65],
  "forward_bell": [0.10, 0.65]
}
```

Both gait kinds also accept `arm_hang_rad` — the world-Z rotation that
flips the arms from a rig-rest T-pose down to a vertical hang. Mirrored
per side, so a single value drives both arms. Defaults to `-1.45` (~-83°)
which suits VRoid / Galaxia rigs whose upper arms rest along ±X. Set to
`0.0` if your rig's arms already rest at the sides.

Body-frame deltas in either gait are conjugated by the current frame's
body yaw, so a single `+0.50` `leg_swing_amplitude` reads as "body-forward"
both when the character faces -Z (yaw=π) and when it faces +Z (yaw=0).
The runtime handles the cross-yaw sign flip — authors only ever write the
amplitude they want in body-frame.

### morphs

```json
"morphs": {
  "smile": {"kind": "linear", "from": 0.0, "to": 1.0},
  "blink": "0.5 * sin(elapsed * tau)"
}
```

Per-phase morph-name → value curve. Each frame's resolved weight is
pushed into the sandbox `morphs` API; the renderer / morph applier
reads from there. Phases that don't declare `morphs` leave the
weight map alone, so a previous phase's last weight persists.

### Lyrics overlay (optional)

```json
"lyrics": [
  { "at_beat": 0, "text": "Now's the time to dance", "duration_beats": 4 },
  { "at_beat": 8, "text": "Feel the rhythm flow",    "duration_beats": 4 }
]
```

Karaoke-style lyric lines drawn as a 2D text overlay on the viewport.
Each entry needs `text` plus exactly one of `at_sec` / `at_beat` for
the start and at most one of `duration_sec` / `duration_beats`
(defaults to a 1-second flash). The runtime finds the active line per
frame and pushes its text through `viewport.set_overlay_text(...)`;
between lines the overlay clears.

The `Viewport.paintEvent` override draws the text via `QPainter` after
the GL pass — a single line, bottom-centred, white bold, 22pt by
default. Lines may overlap; the first matching entry in array order
wins per frame (predictable). Documents without a `lyrics` field
never touch the overlay machinery.

### Audio playback (optional)

```json
"audio": {
  "path": "song.wav",
  "offset_sec": 0.0,
  "sync_clock": true
}
```

Optional document-root audio attachment. `path` is a WAV file
resolved relative to the .json document's directory (absolute paths
also accepted). When present, the runtime loads the clip and starts
playback at start. With `sync_clock: true`, the runtime's wall-clock
time provider is replaced by the audio player's playback position —
the entire animation drifts with the music's actual rate, so retiming
the music retimes the dance. `offset_sec` shifts the audio clock by
a constant.

Gating: missing `audio` → no audio module loaded, no behaviour change.
A present `audio` block with a missing file or unavailable Qt audio
backend logs a warning and falls back silently (the dance continues
with the wall-clock time).

### Camera animation

```json
"camera": [
  { "at_sec": 0.0, "position": [0, 1.4, 2.5], "target": [0, 1.2, 0], "fov": 50 },
  { "at_beat": 16, "position": [1.5, 1.4, 2.0], "target": [0, 1.2, 0], "fov": 35 }
]
```

Document-root keyframe array (sorted by time at parse time). Each
keyframe carries `position` (3-vector), `target` (3-vector), optional
`fov` in degrees, and either `at_sec` or `at_beat` (the beat form
requires the document-level `bpm` to be > 0). The runtime lerps
between bracketing keyframes per frame and writes onto the viewport's
Camera. Times before the first keyframe / after the last hold to that
keyframe — useful for "establishing shot held for N seconds, then
animate". Documents that omit `camera` leave the Camera untouched
(legacy / character-only demos).

### Pose presets

```json
{
  "name": "finale",
  "duration_beats": 4,
  "pose": "v_arms_up",
  "bones": {
    "head": {"y_rad": "0.10 * sin(phase_beat * tau)"}
  }
}
```

Or with a per-frame weight curve so the preset eases in:

```json
"pose": {"name": "v_arms_up", "weight": {"kind": "ease", "from": 0.0, "to": 1.0}}
```

The phase loads the named preset's bone rotations as a starting
silhouette, then `phase.bones` overrides on a **per-axis** basis —
the preset's `head` `x_rad` is kept while the phase's `head` `y_rad`
is added on top. Built-in presets: `v_arms_up`, `arms_to_chest`,
`hip_pop_L`, `hip_pop_R`, `point_L`, `point_R`, `hands_clasp`.

The document may declare its own `pose_library` to override built-ins
or add new presets:

```json
"pose_library": {
  "my_finale": {
    "upper_arm_L": {"x_rad": -1.40, "z_rad": 0.60},
    "upper_arm_R": {"x_rad": -1.40, "z_rad": -0.60}
  }
}
```

User entries with the same name as a built-in win wholesale.

### Hand / finger presets

```json
{
  "name": "wave",
  "duration_beats": 4,
  "hand_L": "open_palm_L",
  "hand_R": "peace_R"
}
```

Per-phase `hand_L` / `hand_R` fields name finger presets from the
document's hand library. Built-ins: `peace_L/R`, `fist_L/R`,
`point_L/R`, `open_palm_L/R`, `thumbs_up_L/R` — each one writes the
five-finger group on the named side using VRoid bone names
(`J_Bip_{L,R}_{Index,Middle,Ring,Little,Thumb}{1,2,3}`). Documents
may extend or override via a root-level `hand_library`:

```json
"hand_library": {
  "rock_horns_L": {
    "J_Bip_L_Index1": {"x_rad": 0.0},
    "J_Bip_L_Little1": {"x_rad": 0.0},
    "J_Bip_L_Middle1": {"x_rad": 1.4},
    "J_Bip_L_Ring1": {"x_rad": 1.4},
    "J_Bip_L_Thumb1": {"x_rad": 0.6}
  }
}
```

Composition order (low → high precedence): body `pose` → `hand_L` →
`hand_R` → phase `bones`. So a phase can use a stock body pose, a
stock hand pose, AND override a single finger axis without
duplicating the rest of the rotations.

### Cross-fade between phases

```json
{
  "name": "reach_right",
  "duration_beats": 4,
  "blend_out_sec": 0.3,
  "body": { ... }
},
{
  "name": "reach_left",
  "duration_beats": 4,
  "blend_in_sec": 0.3,
  "body": { ... }
}
```

When BOTH the current phase's `blend_out_sec` and the next phase's
`blend_in_sec` are > 0, the runtime evaluates both phases in the
overlap window (`min(prev.blend_out_sec, next.blend_in_sec)` seconds
before the boundary) and lerps between their outputs. Body fields use
scalar lerp; bone rotations use quaternion slerp; morph weights use
scalar lerp. Bones present in only one phase blend against rest pose;
morphs present in only one phase blend against weight 0. **Gait does
not blend** — the current phase's gait runs continuously up to the
boundary, then the next phase's takes over. Setting either field to 0
suppresses blending — the boundary becomes a hard cut.

### bones

```json
"bones": {
  "head": {
    "y_rad": "0.30 * sin(elapsed * tau / 2.0)",
    "x_rad": 0.10
  },
  "chest": {
    "x_rad": {"kind": "ease", "from": 0.0, "to": 0.18}
  },
  "upper_arm_L": {
    "x_rad": 1.20
  }
}
```

Per-phase **arbitrary bone driver**. Each entry is `{bone_name: {x_rad?:
curve, y_rad?: curve, z_rad?: curve}}`; missing axes default to zero.
The runtime composes `Rz · Ry · Rx` (extrinsic XYZ) in body frame,
yaw-conjugates to world, then parent-local-conjugates so the same
authored curve produces the same visible motion regardless of root yaw.

Composition order each frame:

1. `_apply_root` — body translation / yaw / lean.
2. `_reset_idle_bones` — cached bones snapped back to rest.
3. `_apply_gait` — walking / stride writes legs / arms.
4. **`_apply_bones`** — explicit per-bone curves overwrite whatever the
   gait wrote on the same bone. Lets a phase hold an arm overhead while
   the underlying walking gait would otherwise swing it.
5. `_apply_morphs` — morph weights pushed to the API.

Bone names go through the rig's `body_bones` alias map, so a single
JSON document can target rigs with different naming conventions
(VRoid `J_Bip_*` vs Sketchfab `upper_arm_L` etc.) by remapping in one
place.

## Value curves

A curve is one of:

- A scalar (number, symbolic constant `"pi"` / `"tau"` / etc., or an
  expression string).
- `{"kind": "constant", "value": ...}`.
- `{"kind": "linear", "from": ..., "to": ...}` — interpolates
  uniformly over `phase_t`.
- `{"kind": "ease", "from": ..., "to": ...}` — cosine ease (smooth
  start AND smooth end).
- `{"kind": "expression", "source": "..."}` — evaluated each frame
  through the safe AST DSL.

**Snappy curve kinds** (added for MMD-style accent hits — the soft
sine ease is too gentle for "snap into pose on the beat" choreography):

- `{"kind": "step", "from": ..., "to": ..., "at": 0.5}` — discrete
  jump at `at` (default 0.5). Below the threshold the value is `from`;
  at or above it, `to`. The cymbal-crash of curve kinds.
- `{"kind": "quad-in" | "quad-out", "from": ..., "to": ...}` — `t²`
  (slow start, abrupt end) and `1 − (1−t)²` (abrupt start, slow end).
  Quad-out is the pose-hit standard: snap into the pose, then settle.
- `{"kind": "cubic-in" | "cubic-out", "from": ..., "to": ...}` — same
  shape as quad but sharper (`t³` / `1 − (1−t)³`). Use when quad still
  feels mushy.
- `{"kind": "back-out", "from": ..., "to": ..., "overshoot": 1.70158}`
  — Penner ease-out-back. Snaps past `to` then settles back exactly on
  it. The "land hard then rebound" feel of MMD pose hits. Default
  overshoot matches MMD / After Effects; raise to 4–5 for a cartoonish
  kick.
- `{"kind": "pulse", "from": ..., "to": ..., "center": 0.5, "width": 0.5}`
  — bell-shaped excursion: returns `from` outside the
  `[center − width/2, center + width/2]` window, peaks at `to` at
  `center`. One declaration replaces a "rise then fall" pair of ease
  curves. Drop on a beat for a "thump" in/out without manual chaining.

All snappy curves accept `from` / `to` (defaults 0). Step also takes
`at`; back-out takes `overshoot`; pulse takes `center` / `width`. All
parameter values go through the same scalar resolver as `from` / `to`,
so symbolic constants and inline expressions work everywhere.

## Expression DSL

Inline expressions can use:

- arithmetic operators: `+ - * / // % ** unary-`
- variables from scope: `elapsed`, `phase_t`, `phase_elapsed`
- math constants: `pi`, `tau`, `e`, `inf`
- math functions: `sin / cos / tan / asin / acos / atan / atan2 /
  exp / log / log10 / sqrt / abs / min / max / floor / ceil / round /
  pow / clamp / lerp / sign`

The evaluator denies attribute access, subscripting, lambdas,
strings / dicts / lists, comparisons, and indirect function calls,
so a malformed or hostile expression raises `ExpressionError`
instead of executing arbitrary Python.

## Physics chains and wind

```json
"physics_chains": {
  "hair_C": {"stiffness": 14.0, "damping": 0.45, "inertia": 0.015}
},
"wind": {
  "direction": [1, 0, 0],
  "speed": 0.4,
  "turbulence_amplitude": 0.10,
  "turbulence_frequency_hz": 1.0
}
```

Applied once at `start()`. Chain names match those discovered by
`physics_host.register_imported_scene`. Wind is registered as an
ambient force on the spring solver.

## Validating a document

```python
from posecascade.scripting.declarative import parse_animation
import json
parse_animation(json.loads(Path("walk.json").read_text("utf-8")))
```

Errors are raised as `DeclarativeAnimationError` with a message that
names the offending field. The full schema lives at
`schemas/animation_v1.json` for IDE / editor integrations.
