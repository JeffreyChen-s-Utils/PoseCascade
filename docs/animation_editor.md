# Animation editor

PoseCascade ships an in-editor authoring surface for declarative
animations: two right-column docks that share one in-memory document
plus an undo / redo command stack so an author can pick whichever
view fits the moment (drag a card vs. type into the JSON) and switch
freely.

The docks live alongside Inspector + Effect chain in the right
column — tabified so they don't take screen real-estate from the
viewport on first open. Toggle them from **View → Animation JSON**
and **View → Phase blocks**.

## Opening a document

`File → Open Script…` (or the `--script path/to/anim.json` CLI flag)
loads any `.json` declarative animation into both docks at once. The
JSON dock's path label updates to the file's location; the phase
blocks dock rebuilds its timeline + card list from the document.

After loading, the runtime is also re-attached — the viewport plays
the animation immediately so author edits show up on the next
**Reload into runtime** click.

## JSON dock

A code editor over the raw JSON, plus a status strip that runs the
JSON-Schema check and the parser end-to-end on every keystroke
(debounced 250 ms). Surface:

* **Syntax highlighting** — keys / strings / numbers / literals /
  punctuation, regex-based per line.
* **Line-number gutter** — keeps the gutter width sized to the
  current line count.
* **Inline error mark** — the gutter row for a parser-error line
  paints red and the row background tints; the status strip below
  shows the message.
* **Format button** — pretty-prints the buffer through
  `json.dumps(indent=2, ensure_ascii=False)`. Skipped if the buffer
  doesn't parse (the status strip explains).
* **Save button + Ctrl+S** — writes the buffer to the bound path.
  No-ops with a status warning if no file is loaded.
* **Reload button** — re-attaches the script through the runtime so
  the viewport picks up the changes without a window restart.
  Blocked on a parse error.
* **Dirty indicator** — the dock title shows a trailing `*` when
  the buffer differs from the last save / sync.

Keyboard:

| Shortcut | Action                                       |
|----------|----------------------------------------------|
| Ctrl+S   | Save buffer to the bound path                |
| Ctrl+Z   | Undo (shared with phase-blocks dock)         |
| Ctrl+Y   | Redo                                         |

The editor's own per-keystroke undo is disabled — undo goes through
the shared command stack so it stays consistent with edits made on
the blocks side.

## Phase blocks dock

Visual editor for the phase timeline:

1. **Horizontal timeline strip** at the top — one coloured bar per
   phase, width proportional to `duration_sec`. Click to select;
   drag to reorder; drag the right edge to resize duration live.
2. **Phase card list** below — one row per phase summarising name,
   duration, pose, gait kind, and bones/morphs counts. Drag rows to
   reorder; the timeline mirrors the change.
3. **Toolbar** — Add (`+`) appends a placeholder phase, Duplicate
   clones the selection with a `_copy` name suffix, Delete removes
   the selection.
4. **Inline form** — when a card is selected, expands a scrollable
   form below the list covering every common field:

   * **Basic** — name, duration, blend in/out, pose preset combo,
     hand L/R preset combos, body yaw, body lean X.
   * **Gait** — kind picker (none / walking / stride) with kind-aware
     field reveal: walking shows step cycle / leg swing / knee bend /
     arm swing / arm hang; stride shows step count + the four
     stride-specific angles plus the shared arm hang.
   * **Body translation** — kind picker between `xyz` (three
     `CurveEditor` cells) and `stair` (base Z / rise / forward /
     step count).
   * **Bones** — table of bone name × (x / y / z) curve cells. Click
     an axis cell to reveal a `CurveEditor` below the table; the
     cell text summarises the current curve.
   * **Morphs** — same table pattern keyed on morph name.

The curve editor handles all 11 supported kinds (`constant`,
`linear`, `expression`, `ease`, `quad-in/out`, `cubic-in/out`,
`back-out`, `pulse`, `step`). The kind combo reveals only the
fields that kind needs; round-trips between scalar / `[from, to]` /
expression string / dict shapes are lossless.

The **Reload** button on this dock matches the JSON dock's — same
re-attach flow into the running runtime.

## Undo / redo (shared)

`AnimationCommandStack` records a whole-document snapshot for every
discrete user action: add / duplicate / delete a phase, drag-reorder
on the timeline, edit a form field, or type into the JSON pane (one
snapshot per "typing session" — the next keystroke after a sync
re-takes a fresh pre-edit snapshot, then subsequent keystrokes don't
pile up). History is capped at 200 entries.

The stack lives on `MainWindow._animation_stack`; both docks
subscribe to the same instance, so Ctrl+Z works the same regardless
of which dock has keyboard focus when the user presses it.

`begin_transaction(label)` / `end_transaction()` collapse a batch of
mutations into one undoable step — used by the form when a single
user action (changing the gait kind, say) needs to write multiple
fields atomically.

## Authoring loop

A typical iteration:

1. **Type in the JSON** for the long-tail (expressions, custom
   curve kinds, pose-library overrides).
2. **Drag on the timeline** to retime phases / reorder the reel.
3. **Form-edit** the selected card for everything else (name,
   duration, pose, gait, body, bones, morphs).
4. **Ctrl+S** to write to disk, then **Reload** to see it in the
   viewport.

Save + Reload are deliberately distinct: an author who's halfway
through a malformed edit can Save the buffer for later without
risking the runtime seeing a parse error.
