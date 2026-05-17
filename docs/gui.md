# GUI walkthrough

A single page covering the editor's user-facing surface: window layout,
every menu, every dock, the toolbar, the status bar, the per-frame loop,
the language menu, and how widgets scale with DPI / system font size. For
the underlying architecture (scene graph, render passes, sandbox), follow
the cross-references at the end of each section.

## Window layout

The editor is a single `QMainWindow`:

```text
┌─────────────────────────────────────────────────────────────────────┐
│  File   View   Settings   Help                          ▢ ─ ✕     │  ← menu bar
├─────────────────────────────────────────────────────────────────────┤
│ ⏸ Pause   ↻ Reset Physics                                         │  ← Playback toolbar
├──────────────┬─────────────────────────────┬────────────────────────┤
│              │                             │                        │
│   Outliner   │                             │      Inspector         │
│              │                             │   (tabbed with:        │
│   ──────     │                             │    Effects             │
│              │     ┌────────────────┐      │    Animation JSON      │
│    Slots     │     │                │      │    Phase blocks)       │
│              │     │   3D Viewport  │      │                        │
│              │     │                │      │                        │
│              │     └────────────────┘      │                        │
├──────────────┴─────────────────────────────┴────────────────────────┤
│  Timeline (tabbed with: Tracks)                                     │
├─────────────────────────────────────────────────────────────────────┤
│  scene: herta.glb · script: walk.json · selected: head    FPS: 60.0 │  ← status bar
└─────────────────────────────────────────────────────────────────────┘
```

The 3D viewport sits as the central widget; everything else is a
`QDockWidget` that can be dragged, tabbed, floated, or hidden. Toggle any
dock from `View → <dock name>`. The dock layout itself is **not** persisted
across sessions in this cut — restart and you get the default arrangement.

### Startup window size

The window opens at 75 % of the primary screen's available geometry. On a
1366×768 laptop that's 1024×576; on a 4K panel that's 2880×1620. See
[Internationalisation & responsive UI](internationalization.md) — the
*Responsive UI sizing* section there covers the rationale and the
fallback path.

## Menu bar

### File

| Entry                | Shortcut       | What it does                                           |
|----------------------|----------------|--------------------------------------------------------|
| Open Scene…          | Ctrl+O         | File dialog filtered to `.glb .gltf .obj .stl .ply .fbx .pmx .pmd`. The importer manager dispatches to the right adapter; failures show as a `QMessageBox` with the underlying `AssetError` message. |
| Open Script…         | (none)         | File dialog filtered to `.py`. Compiles through the sandbox loader. Warning dialog if the scene is empty (most scripts look up nodes by name in `start()` and silently no-op). |
| Open Project…        | (none)         | Loads a `.posecascade` / `.json` project file via the `AppController`. |
| Save Project As…     | Ctrl+Shift+S   | Saves the active project + scene + script bindings. |
| Export…              | Ctrl+E         | Opens the **Export dialog** described below.            |
| Quit                 | Ctrl+Q (platform-specific) | `window.close()` — the per-frame timer stops in `closeEvent`. |

### View

Reset Camera plus a checkable toggle for every dock:

* Outliner
* Slots
* Inspector
* Effects
* Animation JSON
* Phase blocks
* Timeline
* Tracks

The toggle reflects the dock's current visibility — re-clicking an unchecked
entry brings the dock back at its last position.

### Settings → Language

Auto-populated from `available_languages()`. Every shipped locale gets a
checkable entry showing its display name (English, 繁體中文, 简体中文, …);
the currently active locale carries the check mark. Selecting a different
language writes it to `QSettings("PoseCascade", "PoseCascade")` and shows a
`QMessageBox.information` asking the user to restart for the change to
apply to every panel.

> A locale with no display-name registered in
> `posecascade/i18n/__init__.py::_DISPLAY_NAMES` falls back to the raw
> code in the menu — see the *Adding a new language* section in
> [Internationalisation & responsive UI](internationalization.md).

### Help

* **About PoseCascade** — modal with a one-paragraph engine summary.

## Playback toolbar

Two actions:

* **⏸ Pause / ⏵ Play** — flips a `_paused` flag. While paused, the per-frame
  timer keeps firing but the `_tick_simulations` call is short-circuited, so
  the camera still orbits but physics / cloth / scripts freeze.
* **↻ Reset Physics** — calls `reset()` on every registered spring chain and
  cloth piece. Useful when a script has driven the simulation into a wedged
  state and you want a clean restart without reloading the scene.

## Status bar

Two segments:

* **Left (stretching)** — `scene: <name> · script: <name> · selected: <node>`.
  Updated by `_update_status_text` whenever the scene, script, or selection
  changes. The selection part is omitted when nothing is selected; the script
  part is omitted when no script is bound.
* **Right (permanent)** — `FPS: 60.0`. Driven by a rolling 30-frame average
  so the readout doesn't twitch on individual frame-time spikes. The label's
  minimum width is computed as `digit_w × 8` to prevent layout jitter as the
  digit count changes.

## Docks

### Outliner

Left dock, shows the active scene's node tree:

* **Click a row** to select. `node_selected(node)` propagates to the
  Inspector (binds the editor) and to the Viewport (highlights the node).
* **Right-click** a row to open a context menu with **Delete**. Right-click
  also clears the current selection so the highlight disappears immediately.
* **Delete key** deletes the selected node when the tree has focus. The
  `node_deleted` signal carries the deleted node so `MainWindow` can clean
  up any spring chains or cloth pieces that were rooted there.
* Each row shows `<node name>  (N)` where N is the number of attached
  components (visible only when ≥ 1).
* Unnamed nodes render as `<unnamed>` (translated in non-English locales).

### Slots

Left dock under the Outliner — shows every loaded model slot from the
project's `SceneSlots`. One row per slot:

* **Visibility checkbox** — toggling hides / shows the slot in the
  viewport. Hidden slots **still tick** their physics + animation; only the
  draw is skipped. This matters when a paused / hidden slot is driving a
  shared cloth piece.
* **Slot name** label.
* **Three XYZ spin boxes** — world-space translation. Edits the
  `slot.transform.translation` directly and emits
  `slot_translation_changed(name, x, y, z)`.

### Inspector

Right dock. Bound to the currently selected node:

```text
┌────────────────────────────────────────┐
│  Head_M_055                            │  ← name (bold)
│  Translation:  [   0.123 ][ 1.450 ][...]│
│  Rotation (°): [   0.000 ][-15.00 ][...]│
│  Scale:        [   1.000 ][ 1.000 ][...]│
│                                        │
│  Components:                           │
│  ┌──────────────────────────────────┐  │
│  │ SpringChainComponent             │  │
│  │ ClothComponent                   │  │
│  └──────────────────────────────────┘  │
│                                        │
│  ┌─ SpringChain: hair_back ─────────┐  │
│  │  Joints:      [ 12 ]              │  │
│  │  Stiffness:   [ 10.000 ]          │  │
│  │  Damping:     [  0.600 ]          │  │
│  │  Inertia:     [  0.025 ]          │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

Three Vec3 rows for transform editing (rotation is presented as Tait-Bryan
ZYX Euler degrees; round-tripping through quaternion may shift the
displayed values when the same node is reselected, the standard editor
trade-off).

For each attached `SpringChainComponent` or `ClothComponent`, an inline
form lets you tune the live simulation parameters. Edits hit the registered
chain / piece directly (so changes are visible the next frame); they also
write back to the component for offline scenes that haven't been registered
with a host yet.

The components list is capped at five text rows tall (`fm.height() × 5`)
so the parameter forms below stay visible without scrolling on nodes with
many components.

### Effects

Right dock, tabbed with Inspector. Edits the live `EffectChain`:

* **List of chain entries** with a per-entry enable checkbox. Clicking the
  checkbox toggles the entry's enabled state and re-emits `chain_changed`
  so the integrator rebuilds the GL pipeline.
* **↑ / ↓ / Remove** buttons reorder / drop the selected entry.
* **Per-entry uniform editor** below the list — scalar uniforms get a
  spin box, booleans get a checkbox, vector-color uniforms get one spin
  box per component with axis-aware tooltips (`R / X`, `G / Y`, …).

Changes write through the live `EffectChain`, so the viewport repaints
with the new chain immediately. The list of available effects is
populated externally; the dock just edits whatever's on the chain.

### Timeline (basic)

Bottom dock, the simple transport:

```text
┌──────────────────────────────────────────────────────────────────┐
│  ◀━━━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶  │  ← scrub slider
│                                                       [  120 ]    │  ← current frame
│  [ Play ]    Range: [   0 ] → [ 1000 ]                  ☑ Loop   │
└──────────────────────────────────────────────────────────────────┘
```

* **Slider** — drag to scrub. Releasing settles on the dropped frame.
* **Frame spin box** — type a number to jump.
* **Play / Pause** — toggles the internal `QTimer` at 30 FPS (VMD-canonical).
* **Range Start / End spin boxes** — clamp the playable range. Playback
  wraps to Start when Loop is checked; otherwise it stops at End.
* **Loop checkbox** — End → Start wraparound vs. stop-at-end.

When an `AudioPlayer` is attached via `attach_audio`, the timer pulls the
playhead from the audio clock instead of advancing one frame per tick.
Detach with `detach_audio` to revert to wall-clock ticking.

### Tracks

Bottom dock, tabbed with Timeline. The multi-track keyframe editor backed
by an `AnimationDocument`:

* **Tree view** grouped by PMX display frame. Each leaf is a track
  (bone or morph) showing its current keyframe count.
* **Insert / Delete buttons** push commands onto the `CommandStack` —
  Insert adds a default keyframe at the playhead frame on the selected
  track; Delete removes the keyframe at the playhead on the selected
  bone track.
* **Undo / Redo buttons** drive the same stack.
* **Frame label** on the right mirrors the current playhead frame.

Cross-selection between the Tracks tree and the Timeline transport is
not yet plumbed; the playhead drives both via the
`current_frame_changed` signal.

### Animation JSON

Right dock, tabbed with Inspector. A code editor for declarative animation
documents:

* **Plain text editor** (`CodeEditor` subclass of `QPlainTextEdit`) with a
  `JsonHighlighter` painting keys / strings / numbers / literals / punct.
  The line-number gutter paints the parser-error row red.
* **Format button** pretty-prints the buffer through `json.dumps`. Only
  fires on a valid buffer — malformed JSON keeps the user's text intact
  and the status strip surfaces the parse error.
* **Save button** writes the buffer to disk (the path is bound when the
  file was opened). Ctrl+S is the keyboard shortcut.
* **Reload into runtime** re-attaches the live text through the script
  host without restarting the editor. Validation runs first; an
  unparseable buffer refuses to reload so the runtime never sees broken
  state.
* **Status strip** at the bottom shows the validator's verdict in colour
  (green OK, red error, grey "no document").
* **Dirty indicator** — the dock title gets a trailing `*` when the
  buffer diverges from disk. Cleared by Save or by a document-driven sync.

Ctrl+Z / Ctrl+Y route through a shared `AnimationCommandStack` so undo
works across both this dock and the Phase blocks dock. The built-in
per-keystroke undo is disabled — snapshots happen once per typing session
and once per discrete UI action, which matches author expectations better.

### Phase blocks

Right dock, tabbed with the other right docks. The visual editor for
declarative-animation phases:

* **Horizontal timeline strip** at the top — one bar per phase, width
  proportional to `duration_sec`. Click to select; drag to reorder; drag
  the right edge to resize.
* **Vertical card list** below the strip, each card a summary of the
  phase (name, duration, pose preset, gait kind, bone / morph counts).
* **Inline form** below the list, revealing the common editable fields
  for the selected phase. Sub-editors:
  * **Gait** — kind picker (none / walking / stride) with kind-aware
    field reveal.
  * **Body translation** — XYZ value curves or a stair block.
  * **Bones** — table of bone × (x / y / z) curve cells; clicking a cell
    opens a `CurveEditor` covering all 11 supported curve kinds.
  * **Morphs** — same table pattern keyed on morph name.
* **+ Add / Duplicate / Delete buttons** for the phase list.
* **Reload into runtime button** mirrors the JSON dock's button.

Both this dock and the JSON dock share one `AnimationJsonDocument` so an
edit through either view propagates instantly. Both feed the same
command stack so Ctrl+Z spans both. See
[Declarative animation](declarative_animation.md) for the document
schema.

### Export dialog

Modal dialog (`File → Export…` or Ctrl+E). Three tabs:

| Tab            | Output                                                              |
|----------------|---------------------------------------------------------------------|
| VMD            | Binary MMD motion file containing every visible track's keyframes.  |
| Image sequence | One PNG per frame in a directory, zero-padded filenames.            |
| Video          | Encoded video file via libav / FFmpeg, with a codec picker.         |

Common to every tab:

* **Start / End frame** spin boxes.
* **Include post-effect chain** checkbox — apply the viewport's
  post-effect chain (bloom, tonemap, …) to each rendered frame. Uncheck
  to export the raw scene render.

Submitting the dialog emits `export_requested(ExportSpec)`; the integrator
wires that to the actual exporter call. The dialog itself is free of disk
I/O — it just gathers parameters.

## Per-frame loop

`MainWindow` owns a `QTimer` configured for ~60 Hz (`_TARGET_FRAME_INTERVAL_MS
= 16`, `PreciseTimer` type). Each tick:

1. `_clock.tick()` returns the wall-clock dt since the last tick.
2. If unpaused, `_tick_simulations(dt)` runs:
   * `physics_host.tick(dt)` — every registered spring chain steps.
   * `cloth_host.tick(dt)` — PBD cloth solver steps.
   * `script_host.tick(dt)` — every attached script's `update(dt)` runs
     under sandbox protection.
   * `foot_planter.apply()` — last so it sees the final pose; lifts feet
     out of any ground / stair surface they've clipped into.
   Each call is wrapped in its own try / except boundary — one bad
   simulator or script can't freeze the timeline.
3. `_update_fps(dt)` appends to the rolling history + updates the label.
4. `viewport.update()` requests a repaint.

Pausing the toolbar short-circuits step 2 but still runs the FPS update
and viewport repaint, so the camera stays interactive while the
simulation is frozen.

## 3D viewport

The central widget is a `QOpenGLWidget` (`posecascade/ui/viewport.py`)
that owns the GL context. Mouse controls:

| Gesture                                    | Effect                                                   |
|--------------------------------------------|----------------------------------------------------------|
| Middle- / Right-button drag                | Orbit the camera around the look-at point.               |
| Shift + Middle- / Right-button drag        | Pan camera + look-at together (parallel to screen).      |
| Mouse wheel                                | Zoom (changes camera ↔ look-at distance).                |
| Left-button click on a model               | Pick that top-level holder; subsequent drag translates the holder in screen space at its current depth (follows the cursor). |
| Press F (when the viewport has focus)      | Frame the selected node.                                 |

Pitch is clamped to ±89° and orbit distance to `[0.1, 10_000]` to avoid
gimbal flip and runaway zoom. The orbit state re-syncs from the camera's
current pose on every mouse press, so external code (CLI flags, scripts)
that moves the camera between drags doesn't desync the state.

Selection is mirrored bidirectionally with the Outliner — picking in the
viewport selects the row, selecting in the Outliner highlights the
viewport node.

For the render-pass breakdown that draws every frame, see
[Rendering pipeline](rendering_pipeline.md).

## DPI + font-metric sizing

Qt6 high-DPI scaling is enabled with `HighDpiScaleFactorRoundingPolicy
.PassThrough` in `bootstrap._configure_high_dpi`, so every widget reads
its DPI from the OS exactly and renders at the requested point sizes
rather than the rounded multiples Qt would otherwise pick.

Every user-visible widget that needs a floor size derives it from
`QFontMetrics` instead of pixel literals:

```python
fm = self.fontMetrics()
self.setMinimumWidth(fm.horizontalAdvance("0") * 12)   # 12 digits wide
self.setMinimumHeight(fm.height() * 4)                 # 4 text rows tall
```

That gives a layout that scales linearly with the user's font size and
the OS DPI scale — important because CJK glyphs are typically taller
than Latin glyphs at the same point size, and a 640×480 minimum
viewport reads as a postage stamp on a 200 % HiDPI display.

For the precise constants used + the rationale, see the *Responsive UI
sizing* section in
[Internationalisation & responsive UI](internationalization.md).

## Threading model

Reading state from the GUI thread:

* The `QOpenGLWidget` (Viewport) owns the GL context and is the **only**
  thread allowed to call `gl*` functions.
* The `QTimer` runs on the GUI thread and dispatches the per-frame tick
  there.
* Scripts run **on the GUI thread inside the sandbox** — they cannot
  touch Qt or GL handles, but they can mutate the scene graph freely
  because the render thread reads an immutable snapshot at the start of
  each frame.

Background work (asset import, texture decode, mesh tangent generation)
runs on `QThread` workers and emits signals carrying CPU-side data
(numpy arrays, decoded pixel buffers). The GUI thread converts those to
GL objects on receipt. See the threading rules in `CLAUDE.md` for the
full constraint list.

## Cross-references

| Topic                                 | See                                                |
|---------------------------------------|----------------------------------------------------|
| Catalog format, adding a language     | [Internationalisation & responsive UI](internationalization.md) |
| Render pass breakdown                 | [Rendering pipeline](rendering_pipeline.md)        |
| Declarative animation document schema | [Declarative animation](declarative_animation.md)  |
| Phase blocks / JSON dock deep dive    | [Animation editor](animation_editor.md)            |
| Sandboxed scripting API               | English / 繁中 / 简中 User Guide → *Sandboxed Python scripts* |
| MCP server tools (headless)           | [MCP server](mcp.md)                               |
| Packaging the desktop app             | [PyInstaller](packaging_pyinstaller.md) · [Nuitka](packaging_nuitka.md) |
