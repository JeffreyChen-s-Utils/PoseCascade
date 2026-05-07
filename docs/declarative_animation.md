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
  "rig": { ... },
  "ground": { ... },
  "physics_chains": { ... },
  "wind": { ... },
  "phases": [ ... ]
}
```

`schema_version` is required and currently must be `1`. The runtime
rejects unknown versions to give us room to evolve the format.

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
  "knee_bell": [0.10, 0.65],
  "forward_bell": [0.10, 0.65]
}
```

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

## Value curves

A curve is one of:

- A scalar (number, symbolic constant `"pi"` / `"tau"` / etc., or an
  expression string).
- `{"kind": "constant", "value": ...}`.
- `{"kind": "linear", "from": ..., "to": ...}` — interpolates
  uniformly over `phase_t`.
- `{"kind": "ease", "from": ..., "to": ...}` — cosine ease.
- `{"kind": "expression", "source": "..."}` — evaluated each frame
  through the safe AST DSL.

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
