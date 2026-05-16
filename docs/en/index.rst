PoseCascade User Guide
======================

A PySide6 + OpenGL desktop engine for importing 3D models and driving
them with sandboxed scripts. The visual target is **MMD**: toon shading
with crisp inverted-hull outlines, PMX materials with sphere maps,
VMD-style animation curves, IK + foot planting, morph targets, and a
PBD cloth solver fast enough to drape several pieces in real time.

.. contents:: Table of contents
   :depth: 2
   :local:

----

Installation
------------

PoseCascade targets Python **3.14** on Windows, macOS, and Linux. A
project-local virtualenv is recommended so the editable install can
compile the Cython cloth kernel against the same interpreter:

.. code-block:: bash

   git clone https://github.com/JeffreyChen-s-Utils/PoseCascade.git
   cd PoseCascade
   python -m venv .venv
   .venv\Scripts\Activate.ps1            # Windows PowerShell
   # source .venv/bin/activate           # Linux / macOS

   pip install -e .[dev,ai]

The ``dev`` extra pulls in ``pytest``, ``pytest-qt``, ``ruff``,
``bandit``, and ``scikit-image`` (the golden-image SSIM tolerance).
The ``ai`` extra installs the optional ``mcp`` server entry point.

.. note::
   The editable install compiles ``posecascade/animation/_cloth_kernels.pyx``
   in place. On Windows you need **Microsoft Build Tools**; on Linux /
   macOS gcc or clang. If the C compiler is missing, ``pip`` prints a
   warning and the engine transparently degrades to the NumPy fallback
   path — slower, but functionally identical.

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Extra
     - What it unlocks
   * - ``[ai]``
     - ``mcp`` + ``jsonschema`` — installs the ``posecascade-mcp`` CLI
       used by Claude Code, Claude Desktop, Cursor, …
   * - ``[fbx]``
     - The FBX importer plugin (separate runtime because the FBX SDK is
       Autodesk-licensed and bulky).
   * - ``[usd]``
     - Universal Scene Description plugin (Pixar's USD bindings).
   * - ``[collada]``
     - ``.dae`` importer (XML-based; uses ``defusedxml`` to avoid XXE).

----

Running the examples
--------------------

The shipped examples live in ``examples/``. They fall into three
buckets:

Interactive viewport (declarative JSON)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to see a character animated. Pass ``--scene`` for
the model and ``--script`` for a ``.json`` animation document:

.. code-block:: bash

   # In-place walking + arm swing — 4 s loop
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/walk.json

   # 30 s showcase reel — idle → turntable → wave → V-pose → bow
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/showcase.json

   # Up + down stairs (declarative phases)
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/climb_stairs.json

   # Minimal breathing-idle loop
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/idle.json

Interactive viewport (sandboxed Python)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use a ``.py`` extension to load a sandboxed Python script instead of
a JSON document. These are useful when the per-frame logic doesn't
fit cleanly into phases (e.g. a cloth-only demo or a hair-physics
showcase):

.. code-block:: bash

   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/walk.py
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/hair_sway.py
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/cape_cloth.py
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/full_demo.py
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/idle_orbit.py

Headless / smoke renders
^^^^^^^^^^^^^^^^^^^^^^^^

Standalone scripts that don't open a Qt window — they spin up an
offscreen GL context, render a single frame (or an N-frame strip),
and write the result to disk. Useful as fast smoke checks for the
whole visual stack:

.. code-block:: bash

   # MMD-style hero frame with every fluence toggle enabled
   python examples/mmd_demo.py

   # PMX-native path (per-mesh MMDMaterial + sphere texture)
   # --frames N produces an N-frame animation strip
   python examples/march7th_pmx_demo.py --frames 8

   # Side-by-side comparison renders
   python examples/compare_bloom.py     # AutoLuminous bloom OFF vs ON
   python examples/compare_dqs.py       # LBS candy-wrapper vs DQS
   python examples/compare_lights.py    # primary only vs HighDef rig
   python examples/compare_tone.py      # sRGB only vs + mmd_tone

   # 360° turntable spin (writes spin.mp4)
   python examples/spin.py

Bundled character
^^^^^^^^^^^^^^^^^

All shipped examples target ``examples/assets/herta/herta.glb``
(Honkai: Star Rail's "The Herta" — 354 joints, dress rigged into the
body mesh). The MMD demo path uses a separate PMX asset:

.. code-block:: bash

   python -m posecascade --scene examples/assets/march7th/march7th.pmx

Both assets are third-party CC-BY 4.0. The glTF is by X9_YT on
Sketchfab (character "The Herta" © HoYoverse, Fan Content Guidelines);
the PMX is by Gregman on Sketchfab (character "March 7th" © HoYoverse).
See ``examples/assets/herta/NOTICE.md`` and
``examples/assets/march7th/NOTICE.md`` for the full attributions.

----

Authoring declarative animations
--------------------------------

A declarative animation is a JSON document under ``examples/scripts/``
that wires the character into a sequence of **phases**. Each phase
specifies how long it lasts, what the body does (translation, yaw,
lean), what the limbs do (gait, IK targets, pose presets), and what
the morphs do (smile, blink, mouth-A). The runtime cross-fades
between phases, loops at the end, and serves the per-frame pose to
the renderer.

Document anatomy
^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "schema_version": 1,
     "name": "walk_in_place_demo",
     "loop_sec": 4.0,
     "rig": {
       "character_root": "Sketchfab_model",
       "body_bones": {
         "head":        "Head_M_055",
         "upper_arm_L": "Shoulder_L_0183",
         "upper_arm_R": "Shoulder_R_0233"
       }
     },
     "phases": [
       {
         "name": "walk",
         "duration_sec": 4.0,
         "body":  { "yaw_rad": 0.0 },
         "gait":  { "kind": "walking", "step_cycle_sec": 1.0,
                    "leg_swing_amplitude": 0.50,
                    "arm_swing_amplitude": 0.55 }
       }
     ]
   }

Top-level keys:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``schema_version``
     - Always ``1``. Bumped if a future change is non-backwards-compatible.
   * - ``name``
     - Stable identifier for logs and the MCP ``list_animations`` tool.
   * - ``loop_sec``
     - Total length of one loop. Must equal the sum of phase
       ``duration_sec`` (the parser checks this).
   * - ``rig``
     - Bone-name aliases — see *Bone aliasing* below.
   * - ``ground``
     - Optional flat ground plane (Y height) for foot-IK clamping.
   * - ``phases``
     - Ordered list. Each entry runs ``duration_sec`` seconds.
   * - ``hide``
     - Optional list of node names to detach at start (props, lights,
       stairs the animation doesn't want).
   * - ``cloth`` / ``colliders``
     - Optional declarative cloth setup — see *Cloth* below.
   * - ``pose_library``
     - Optional document-local pose presets that override built-ins.

Phase body
^^^^^^^^^^

Each phase declares one or more of:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Drives
   * - ``body``
     - Whole-character yaw, translation, lean. Values are
       expressions evaluated per frame (``"phase_t"`` is the time
       elapsed within the phase, in seconds).
   * - ``gait``
     - Procedural walking / running cycle. Sets leg + arm angles
       analytically. ``kind`` is one of ``"walking"``, ``"running"``,
       ``"idle"``.
   * - ``ik``
     - Foot-IK targets. Combined with the ``ground`` clamp to keep
       feet on the floor at all times.
   * - ``poses``
     - Named preset blends (``reach_R_soft``, ``wave_L``, ``T_pose``…).
       Built-in presets in ``posecascade/scripting/pose_library.py``.
   * - ``morphs``
     - Per-morph weights with expression support
       (``"smile": "0.5 + 0.5*sin(phase_t * pi)"``).
   * - ``cross_fade_sec``
     - Time to ease into this phase from the previous one. The runtime
       interpolates both pose + morph weights.

Expression DSL
^^^^^^^^^^^^^^

Numeric fields accept either a literal number or a string expression.
The DSL is a safe subset of Python — no ``import``, no attribute
access, no function definitions. Available identifiers:

* **Time**: ``phase_t`` (elapsed in current phase), ``t`` (elapsed
  in whole loop), ``loop_sec``.
* **Constants**: ``pi``, ``tau``, ``e``.
* **Math**: ``sin``, ``cos``, ``tan``, ``sqrt``, ``exp``, ``log``,
  ``abs``, ``min``, ``max``, ``clamp(x, lo, hi)``, ``lerp(a, b, t)``,
  ``smoothstep(edge0, edge1, x)``.

Anything else raises ``ExpressionError`` at parse time — typos surface
as a test failure rather than a silent NaN at frame 200.

Bone aliasing
^^^^^^^^^^^^^

Different rigs name the same anatomical bone differently:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Convention
     - Example bone names
   * - VRoid / VRM
     - ``J_Bip_C_Head``, ``J_Bip_L_UpperArm``, ``J_Bip_L_UpperLeg``
   * - HoYoverse FBX
     - ``Head_M_055``, ``Shoulder_L_0183``, ``Hip_L_02``
   * - MMD PMX / PMD
     - ``頭``, ``左腕``, ``左足``
   * - Mixamo
     - ``mixamorig:Head``, ``mixamorig:LeftArm``

``posecascade.animation.bone_aliasing.detect_humanoid_aliases`` runs
at import time and produces a ``{canonical → Node}`` map. Animation
documents reference canonical names (``head``, ``upper_arm_L``,
``upper_leg_L``) so the same JSON plays on any rig that aliases
those bones. The optional ``rig.body_bones`` block lets an animation
override the autodetected mapping.

Cloth
^^^^^

Declarative cloth attaches a PBD simulation to a named mesh:

.. code-block:: json

   {
     "cloth": [{
       "mesh_node": "skirt",
       "track_bone": "hip",
       "anchor_top_verts": true,
       "stiffness": 0.85,
       "bend_stiffness": 0.20
     }],
     "colliders": [
       { "kind": "sphere",  "follow_bone": "Hip_L_02", "radius": 0.06 },
       { "kind": "capsule", "follow_bone": "Knee_L_04",
         "radius": 0.05, "height": 0.20 }
     ],
     "wind": { "speed": 0.40, "direction": [0, 0, -1] }
   }

The ``track_bone`` keeps top-row anchor vertices glued to a moving
bone, so the cloth follows the character even when it walks across
the floor. Colliders are bone-follow capsules / spheres — the cloth
solver sweeps each edge against the collider every step.

For the full schema, see :doc:`/declarative_animation`.

Inheriting boilerplate (``extends``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A profile JSON can carry the rig / ground / physics_chains / wind /
colliders / collision_deform_meshes block once; every animation that
targets the same character drops them and just references the
profile:

.. code-block:: json

   {
     "schema_version": 1,
     "extends": "_herta_profile.json",
     "name": "my_anim",
     "loop_sec": 4.0,
     "phases": [
       { "name": "do_thing", "duration_sec": 4.0, "pose": "rest_arms" }
     ]
   }

``extends`` resolves relative to the file's directory and is
path-traversal safe. The merge is shallow at the top level — each
child key replaces the parent's value outright, with the exception
of ``pose_library`` and ``hand_library`` which merge per-preset.
``phases`` are never inherited.

The bundled ``_herta_profile.json`` is a working example —
``idle.json``, ``walk.json``, ``climb_stairs.json``, and
``showcase.json`` all extend from it.

Shorthand syntax
^^^^^^^^^^^^^^^^

Three array shapes are accepted as authoring shortcuts:

* ``[from, to]`` instead of ``{"kind": "linear", "from": …, "to": …}``
  for a linear curve.
* ``[x, y, z]`` instead of ``{"x": …, "y": …, "z": …}`` for
  ``body.translation``. Each element is itself a value curve, so
  ``[0, 0, [0.0, -2.0]]`` means "linear Z 0 → -2 over the phase,
  X and Y constant".
* ``x`` / ``y`` / ``z`` instead of ``x_rad`` / ``y_rad`` / ``z_rad``
  for axes inside a ``bones`` block.

Mixing the short and long forms on the same axis (``{"x": 0.5,
"x_rad": 0.5}``) is rejected at parse time.

----

In-editor animation editor
--------------------------

PoseCascade ships an in-editor authoring surface for declarative
animations: two right-column docks sharing one in-memory document
plus an undo / redo command stack so an author can pick whichever
view fits the moment (drag a card vs. type into the JSON) and switch
freely.

Open any ``.json`` declarative animation via ``File → Open Script…``
and both docks populate at once. Toggle them from ``View → Animation
JSON`` and ``View → Phase blocks``.

The **JSON dock** is a code editor with per-line syntax highlighting
(keys / strings / numbers / literals / punct), a line-number gutter
that paints the parser-error row red, a Format button that
pretty-prints through ``json.dumps``, a dirty indicator (``*`` in
the dock title until saved), and a Reload button that re-attaches
the script into the running runtime without a window restart.

The **Phase blocks dock** has a horizontal timeline strip (one bar
per phase, width proportional to ``duration_sec``; click to select,
drag to reorder, drag the right edge to resize) plus a vertical
card list summarising each phase. Selecting a card reveals an inline
form covering every common field:

* **Basic** — name, duration, blend in/out, pose preset, hand L/R
  presets, body yaw, body lean X.
* **Gait** — kind picker (none / walking / stride) with kind-aware
  field reveal.
* **Body translation** — XYZ value curves or a stair block.
* **Bones** — table of bone × (x / y / z) curve cells; clicking a
  cell opens a ``CurveEditor`` for any of the 11 supported curve
  kinds.
* **Morphs** — same table pattern keyed on morph name.

Ctrl+Z / Ctrl+Y route through the shared command stack so undo
works across both docks. The JSON editor's per-keystroke undo is
disabled; snapshots happen once per typing session and once per
discrete UI action.

For the full surface, see :doc:`/animation_editor`.

----

Sandboxed Python scripts
------------------------

When a ``.py`` extension is passed to ``--script``, PoseCascade loads
the file through the **sandbox** in ``posecascade/scripting/sandbox.py``.
The sandbox:

1. Reads the script source (path-traversal-checked against the
   project root).
2. Builds a restricted ``globals`` dict containing only curated API
   objects (``scene``, ``nodes``, ``time``, ``input``, ``math``,
   ``vec3``, ``quat``, ``lerp``, ``clamp``, ``noise``) and a minimal
   builtins whitelist (``len``, ``range``, ``min``, ``max``, ``abs``,
   ``round``, ``enumerate``, ``zip``, ``print`` → routed to logger).
3. Compiles + executes the script.
4. Pulls out the user's ``update(dt)`` / ``start()`` / ``on_event(...)``
   callables and stores them on the script host.

Every per-frame call is wrapped in a try/except: a user exception is
logged as a ``ScriptRuntimeError`` and the offending script is
disabled — one bad script never freezes the timeline.

What is **not** available
^^^^^^^^^^^^^^^^^^^^^^^^^

* ``open``, ``os``, ``sys``, ``subprocess`` — no filesystem / network
  access. Use the engine's asset cache if you need a texture.
* ``eval``, ``exec``, ``compile``, ``__import__`` — the sandbox loader
  is the **only** exec call in the codebase.
* ``__builtins__`` is replaced wholesale; nothing leaks through
  ``__class__.__mro__`` tricks.
* Any Qt / GL handle. Mutations go through ``scene.find(name)`` →
  ``node.translate/rotate/scale``; the render thread picks up the
  change next frame.

----

Rendering pipeline
------------------

The forward renderer runs **six passes per frame** in a fixed order:

1. **Depth-map shadow pass** — renders the scene from the primary
   light's PoV into a depth FBO. Drives self-shadow PCF.
2. **Scene pass** — the actual lit + shaded scene. Toon ramp,
   sphere-map composite, inverted-hull outline, optional DQS skinning.
3. **Ground pass** — procedural checkered floor with depth, blended
   against the gradient sky.
4. **Projected ground shadow** — quad in the ground plane projects
   the character silhouette as a soft drop shadow.
5. **Selection overlay** — re-outlines the picked top-level holder
   in a bright contrast colour.
6. **Post-process chain** — AutoLuminous bloom + MMD tone curve +
   sRGB-aware output.

Each pass has a toggle (``set_ground_enabled``,
``set_self_shadow_enabled``, ``set_projected_shadow_enabled``,
``set_selected_holder``) so smoke tests and headless renders can
opt out without losing pixel fidelity in the remaining passes.

The full breakdown — pass order, shader files, light-space math,
texture units, MMD-fluence gaps — lives in
:doc:`/rendering_pipeline`.

MMD-fluence toggles
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - What it does
   * - Toon ramp
     - 1-D toon ramp texture sampled NEAREST + clamp → crisp cel
       bands without the linear-filter smear.
   * - Sphere map (mul / add)
     - Per-pixel composite of an env-mapped sphere texture. PMX
       materials store the mode (``MUL`` / ``ADD`` / ``SUB``) per mat.
   * - Inverted-hull outline
     - Second draw pass with front faces culled and vertices pushed
       along their normal by ``outline_width`` — the classic MMD
       silhouette outline.
   * - AutoLuminous bloom
     - Per-pixel emission threshold + box-blur + additive composite.
       Driven by PMX material ``emission`` channel.
   * - MMD tone curve
     - Hue / saturation / value remap applied last in the post
       chain. Specifically tuned to match MikuMikuDance's defaults.
   * - DQS (dual-quaternion skinning)
     - Optional skinning alternative to LBS. Preserves joint volume
       at extreme twist — no "candy wrapper" pinch on the elbow.

----

Cloth solver
------------

PBD (position-based dynamics) with structural + bend constraints and
sphere / capsule colliders. The solver is written twice:

* **Python orchestrator** (``posecascade/animation/cloth.py``) —
  builds the constraint topology, runs broad-phase culling against
  colliders, integrates external forces (gravity + wind).
* **Cython kernel** (``posecascade/animation/_cloth_kernels.pyx``) —
  the hot per-vertex / per-constraint inner loop. Built in place
  by ``setup.py build_ext --inplace``.

If the Cython extension isn't compiled, the kernel transparently
degrades to a pure-NumPy fallback. Same API, ~9× slower.

A separate **GPU compute path** handles ``passive_skin_deform``
cloth pieces — large character meshes that need LBS + collider push
on every vertex but don't need full PBD. The OpenGL 4.3 compute
shader at ``shaders/passive_skin/passive_skin_push.comp`` writes
directly into the mesh's existing position + normal VBOs, dropping
the per-frame cost on a 30 k-vert body mesh from ~9 ms on the CPU
to under 0.05 ms. Falls back transparently to the CPU LBS path
when the context lacks 4.3 / compute support. See
:doc:`/rendering_pipeline` for the dispatcher API and integration
details.

Benchmarks (480-vert skirt, 8 iterations / step, 100-step warmup,
best of three runs):

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Stage
     - ms/step
     - vs baseline
   * - Baseline (pre-tuning)
     - 3.225
     - —
   * - NumPy: einsum + combined bincount
     - 2.085
     - −35%
   * - Cython kernel
     - 0.356
     - **−89%**
   * - Cython + broad-phase + bin culling
     - 0.36–0.38
     - + extra 30% for single-bin colliders

Reproduce locally with the MCP ``cloth_benchmark`` tool, or directly:

.. code-block:: python

   from posecascade.mcp.server import _cloth_benchmark_impl
   result = _cloth_benchmark_impl(rows=20, cols=24, steps=600)
   print(result["ms_per_step"], result["native_kernel"])

``native_kernel: true`` means the Cython extension loaded; ``false``
means the NumPy fallback is running.

----

MCP server
----------

PoseCascade ships a `Model Context Protocol
<https://modelcontextprotocol.io/>`_ server so any MCP-aware LLM
agent can drive the engine without going through the desktop UI.
The server is headless — it never touches Qt or the GL context —
so it runs cleanly in a subprocess over stdio.

Install with the ``ai`` extra:

.. code-block:: bash

   pip install -e .[ai]

That pulls in ``mcp`` and ``jsonschema`` and installs a
``posecascade-mcp`` console script.

The repo's ``.mcp.json`` is a project-level config Claude Code (and
other MCP-aware clients) picks up automatically when the venv is on
PATH:

.. code-block:: json

   {
     "$schema": "https://modelcontextprotocol.io/schema/server-config.json",
     "mcpServers": {
       "posecascade": {
         "command": "posecascade-mcp",
         "args": [],
         "env": {}
       }
     }
   }

Tools exposed
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Tool
     - Purpose
   * - ``list_animations``
     - List every ``.json`` declarative animation under
       ``examples/scripts/`` with its ``name``, ``loop_sec``, and
       phase count.
   * - ``read_animation``
     - Return the raw JSON text of one of those files.
   * - ``validate_animation``
     - JSON-Schema check **plus** runtime-parser check. Pass
       exactly one of ``content`` (inline) or ``path``
       (project-relative).
   * - ``inspect_model``
     - Import any supported model and return a structural summary
       (mesh / texture / skin / node / vertex / triangle counts,
       first 20 bone names, world AABB).
   * - ``cloth_benchmark``
     - Build a synthetic grid, drop gravity + a sphere collider,
       run the solver N steps. Returns ms/step (best of three),
       ``frame_section`` breakdown, and ``native_kernel`` flag.

For the full surface (signatures, path safety, schema details), see
:doc:`/mcp`.

----

Project layout
--------------

::

   PoseCascade/
   ├── posecascade/                # main package
   │   ├── animation/              # cloth, skin, morphs, IK, VMD tracks
   │   │   ├── cloth.py            # PBD solver (Python orchestration)
   │   │   └── _cloth_kernels.pyx  # Cython inner loop
   │   ├── app/                    # QApplication bootstrap, main window
   │   ├── assets/                 # cache, path safety, importer manager
   │   ├── gl/                     # GL context, shaders, framebuffers
   │   ├── mcp/                    # Model Context Protocol server
   │   ├── render/                 # render graph, materials, lights
   │   ├── scene/                  # scene graph, transforms, components
   │   ├── scripting/              # sandbox host + declarative runtime
   │   └── ui/                     # viewport, outliner, inspector, timeline
   ├── importers/<format>/         # per-format importer plugins
   ├── shaders/                    # GLSL by render pass
   ├── examples/                   # bundled models + animation scripts
   ├── tests/                      # pytest suite mirroring the package
   ├── docs/                       # this Sphinx tree
   ├── schemas/                    # JSON schemas (declarative animation)
   ├── setup.py                    # cythonize build hook
   └── pyproject.toml              # metadata + ruff / bandit config

----

Development workflow
--------------------

Definition of Done
^^^^^^^^^^^^^^^^^^

Every change must satisfy three gates before commit (see
``CLAUDE.md``):

.. code-block:: bash

   .venv/Scripts/python.exe -m pytest tests/
   .venv/Scripts/python.exe -m ruff check .
   .venv/Scripts/python.exe -m bandit -c pyproject.toml -r posecascade/

The ``-c`` flag on bandit is **required** — without it bandit ignores
the project skip config and the run will be noisy.

Rebuilding the Cython kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cloth Cython kernel must be re-built in place after any change
to ``_cloth_kernels.pyx``:

.. code-block:: bash

   .venv/Scripts/python.exe setup.py build_ext --inplace

Tests
^^^^^

Tests mirror the package layout: each production module
``posecascade/<area>/<feature>.py`` has a paired
``tests/test_<feature>.py``. Run the whole suite:

.. code-block:: bash

   .venv/Scripts/python.exe -m pytest tests/

GL-heavy tests use the ``gl_context`` fixture which spins up an
offscreen ``QOpenGLContext`` — they skip cleanly with
``pytest.skip("no GL")`` on systems where context creation fails,
so headless CI runners don't false-fail.

Golden-image tests under ``tests/render/`` diff the rendered frame
against ``tests/golden/*.png`` using SSIM with a documented per-test
tolerance.

Linting + security
^^^^^^^^^^^^^^^^^^

Ruff catches most style issues automatically. The project enforces
SonarQube / Codacy / pylint default rules (complexity ≤ 15,
function length ≤ 75 lines, file length ≤ 1000 lines, no magic
numbers, no bare ``except``, no mutable default args). Bandit
scans for security-relevant patterns (``pickle.load``, ``yaml.load``
without SafeLoader, MD5 / SHA-1 for security, ``shell=True``).

Project-wide skips live in ``.bandit`` and mirror in
``pyproject.toml`` ``[tool.bandit]``. Per-line suppressions need a
brief justification on the same line, e.g.
``# nosec B102  # restricted globals; see sandbox.py``.

----

Troubleshooting
---------------

GL context fails to create
^^^^^^^^^^^^^^^^^^^^^^^^^^

PoseCascade requires **OpenGL 3.3 core profile** or newer. On older
Intel iGPUs without a current driver, Qt may fall back to OpenGL
1.4 and crash on shader compile. Workarounds:

* Update the GPU driver (Intel HD 4000+ supports GL 3.3 on Windows).
* Force software rendering via ``QT_QPA_PLATFORM=offscreen`` for
  headless renders, or install Mesa software rasteriser
  (``libgl1-mesa-glx`` on Debian-based Linux).
* If you only need the MCP server, the ``ai`` extra works on any
  CPU — no GL context is created.

Cloth solver is slow
^^^^^^^^^^^^^^^^^^^^

Check ``cloth_benchmark`` reports ``native_kernel: true``. If
``false``, the Cython extension didn't compile. Look for warnings
from ``pip install -e .`` and verify a C compiler is on PATH:

* **Windows**: install Microsoft Build Tools 2022 with the C++
  workload. Re-run ``pip install -e .[dev]``.
* **Linux**: ``apt install build-essential python3.14-dev``.
* **macOS**: ``xcode-select --install``.

Imported model is invisible / appears as a single bone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most often the model contains a separate mesh for a weapon, prop,
or stage piece that triggers an importer warning. Open the file in
Blender, remove the offending mesh, and re-export. The bundled
March 7th GLB went through exactly this dance — two weapon meshes
were stripped before the file shipped in the repo.

If you're using the PMX path and the toon shading looks washed out,
verify the toon ramp texture (``toon01.bmp`` … ``toon10.bmp``) is
in the same folder as the ``.pmx`` file. PMX materials reference
toon textures by relative path.

----

License + attribution
---------------------

The codebase is **MIT-style**; see ``LICENSE`` for the exact terms.

Bundled assets carry their own licenses:

* ``examples/assets/herta/herta.glb`` — CC-BY 4.0 (uploader
  *X9_YT* on Sketchfab). The character "The Herta" is © HoYoverse
  and the model is used under their **Fan Content Guidelines**: no
  commercial use, attribution maintained, non-derogatory use only.
  Full attribution in ``examples/assets/herta/NOTICE.md``.
* ``examples/assets/march7th/march7th.pmx`` — CC-BY 4.0 (uploader
  *Gregman* on Sketchfab). The character "March 7th" is © HoYoverse,
  same Fan Content Guidelines. Full attribution in
  ``examples/assets/march7th/NOTICE.md``.
* Stock skybox / ground textures — public domain (CC0).

If you fork PoseCascade for a commercial product, replace
``examples/assets/herta/`` and ``examples/assets/march7th/`` with
models you have the rights to. The engine itself has no MMD /
HoYoverse-specific code paths — any PBR-skinned humanoid plugs into
the same alias layer.
