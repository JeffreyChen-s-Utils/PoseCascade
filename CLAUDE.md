# Project Guidelines

## Project Overview

PoseCascade is a PySide6 + OpenGL desktop engine that:

1. **Imports 3D models and scenes** from common formats (glTF/GLB, OBJ + MTL, FBX, STL, USD/USDZ, PLY,
   COLLADA). Each format ships behind an importer adapter so adding a new format does not touch
   the renderer or scene graph.
2. **Renders them in a Qt viewport** (`QOpenGLWidget` / `QOpenGLWindow`) using a forward + optional
   deferred path with PBR materials, skeletal animation, morph targets, and IBL.
3. **Drives them with user scripts** — a sandboxed scripting host exposes a stable API
   (`scene`, `nodes`, `time`, `input`, `physics_lite`) so users can pose, key-frame, and animate
   imported assets without touching engine internals.

The whole stack is single-process and runs on Python 3.14 with PySide6. Heavy I/O (model parsing,
texture decoding, mesh tangent/normal generation) MUST happen off the GL thread and feed into the
renderer via Qt signals or a producer/consumer queue.

### Top-level layout (target)

```
PoseCascade/
├── posecascade/                  # main package
│   ├── app/                      # QApplication bootstrap, main window, dock layouts
│   ├── gl/                       # GL context, shaders, framebuffers, render passes
│   ├── render/                   # render graph, materials, lights, cameras
│   ├── scene/                    # scene graph, transforms, components, ECS-lite
│   ├── animation/                # skeletons, clips, blend trees, morph drivers
│   ├── assets/                   # asset cache, GPU resource lifecycle, hot-reload
│   ├── scripting/                # sandboxed script host, exposed API surface
│   ├── ui/                       # Qt widgets (viewport, outliner, inspector, timeline)
│   └── utils/                    # logging, math, IO, profiling
├── importers/<format>/           # per-format importers (gltf/, obj/, fbx/, stl/, usd/, ply/, collada/)
│   └── __init__.py               # sets `importer_class`; pure-logic submodules live INSIDE the folder
├── shaders/                      # GLSL source, organised by render pass
├── examples/                     # sample scenes + scripts users can study
├── tests/                        # pytest suite, mirrors package layout
├── pyproject.toml                # ruff, bandit, build config
└── .bandit                       # canonical bandit skip list
```

## Subagents — consult BEFORE editing

This project ships specialised subagents under `.claude/agents/` that
encode load-bearing engine quirks. **READ the relevant subagent file
before touching the code it covers** — these documents exist because
each rule was paid for with a debug cycle on a real bug, and the
gotchas are not visible from the source alone.

Current subagents:

- **`.claude/agents/animation-tuning.md`** — hair / spring chains,
  leg-arm IK, `floor_align`, pose-specific drape JSON. Consult before
  any edit to: `posecascade/animation/spring.py`, `posecascade/animation/ik.py`,
  `posecascade/animation/cloth*.py`, `shaders/hair/*`, `posecascade/gl/compute_hair.py`,
  `examples/scripts/*.json`, `posecascade/scripting/declarative.py` (IK or
  pose blocks), or any task that mentions hair clipping, knees / feet,
  drape, `gravity_override`, `static_drape`, `floor_align`, `bones_local`,
  `dog_crawl` / `kneeling` / `prone` poses.

The main agent should read these files itself (they're project docs,
not spawn targets) — don't `Agent`-spawn unless the user asks.

## Definition of Done (HARD REQUIREMENT)

Every feature, bug fix, refactor, or behaviour change MUST satisfy ALL of the following before it
can be committed. No exceptions — incomplete work stays on the working copy until the gates pass.

1. **Unit tests are written and they pass.** New code without new tests is incomplete; the commit
   fails this gate. See the **Unit Tests** section below for the exact coverage expectations.
2. `py -m pytest tests/` runs clean (or only skips that already existed before the change).
3. `py -m ruff check .` reports no new errors.
4. `py -m bandit -c pyproject.toml -r posecascade/` reports `No issues identified`.
5. **Visual smoke check** for any change that touches `posecascade/gl/`, `posecascade/render/`,
   `posecascade/scene/`, or shaders: launch `py -m posecascade --scene examples/smoke.glb` and
   confirm no GL errors are logged and the frame matches the reference screenshot in
   `tests/golden/`. Headless CI runs the same check via `pytest --offscreen` (Qt's offscreen
   platform plugin); golden-image diffs use SSIM with a tolerance documented per-test.
6. The commit message contains no AI tool/model names and no `Co-Authored-By` line.

When you finish editing code, work through this list explicitly before staging. If a gate fails,
fix it — do not ship around it. Skipping tests "to come back later" is not allowed because later
never happens and the gap compounds.

## Git Commits

- NEVER add `Co-Authored-By` lines to commit messages. All commits should only contain the commit
  message itself with no co-author attribution.
- NEVER mention "Claude", "Claude Code", "AI-generated", "GPT", "Copilot", or any AI tool/model
  name anywhere — including commit messages, PR titles, PR descriptions, code comments, and
  documentation.

## Code Quality Requirements

### Design Patterns

- Apply appropriate design patterns (Strategy, Observer, Factory, Singleton, Command, Builder,
  Adapter, Decorator, Composite, Visitor) where they fit naturally. The scene graph is a textbook
  Composite; importers are Strategies behind a Factory; the undo/redo stack is Command;
  property change notifications are Observer.
- Prefer composition over inheritance. The scene-graph node is a transform + a list of components,
  not a deep class hierarchy.
- Follow SOLID principles: Single Responsibility, Open/Closed, Liskov Substitution, Interface
  Segregation, Dependency Inversion. The renderer depends on a `Mesh` / `Material` / `Skeleton`
  interface, never on a concrete importer's intermediate types.
- Apply DRY — extract shared GL helpers into `posecascade/gl/`; never copy a VAO/VBO setup across
  passes.
- Reuse the existing project patterns: `QThread` worker for asset loading, signal/slot for UI
  communication, a single render-thread-owned GL context, double-buffered scene state for
  thread-safe reads from the script host.

### Software Engineering Practices

- Separate concerns: the renderer never reads scene-graph internals directly — it consumes a
  flat, immutable `RenderList` produced by a culling/sorting pass. The script host never touches
  GL — it mutates scene state, the renderer picks up the change next frame.
- Write self-documenting code with clear naming; add comments only for non-obvious "why"
  explanations (e.g. "row-major because glUniformMatrix4fv with `transpose=GL_TRUE` is faster than
  transposing CPU-side on AMD drivers").
- Favor immutability where practical — `Mesh`, `Material`, `Texture`, and `AnimationClip` are
  immutable once uploaded; mutations create a new asset and bump a version counter.
- Handle errors explicitly at system boundaries (file IO, GL calls, script execution); propagate
  exceptions cleanly through internal layers. Wrap every `glGetError`-style check in a helper that
  raises a typed `GLError` — never swallow.
- Keep functions short and focused — one function, one responsibility.
- Delete dead code immediately; do not comment it out or leave unused imports/variables.

### Performance

- **NEVER disable, gate behind a flag, or remove a feature to "fix" performance.** A slow
  feature is a bug to be optimised, not a feature to be hidden. The user is shipping the
  feature for a reason — taking it out trades a clear regression for an invisible cost
  ("now this asset doesn't clip" → "now this asset doesn't clip AND drops below 60 fps").
  When perf-tuning, the deliverable is the same user-visible behaviour at lower CPU/GPU cost,
  full stop. If after honest profiling the only path to acceptable perf truly is to scope
  the feature down (e.g. operate on a vertex subset rather than the whole mesh), surface the
  trade-off to the user explicitly and let them decide — don't unilaterally take features away.
- Always consider and implement the best-performance approach for the task.
- Use lazy loading and on-demand initialization where applicable. Textures and meshes are
  uploaded to the GPU on first draw, not at import time.
- Avoid unnecessary memory allocations and copies — reuse buffers when processing large data
  (vertex arrays, index buffers, animation channels). Numpy views over `bytes` are preferred
  over per-frame `np.array(...)` allocations.
- Prefer batch operations over per-item processing. Group draw calls by material; sort by depth
  only where alpha-blending requires it.
- Use appropriate data structures: dict for O(1) node lookup by id, deque for the asset-load
  queue, set for "dirty" component tracking, struct-of-arrays for hot transform data.
- Profile and measure before optimizing hot paths; avoid premature optimization of cold paths.
  `posecascade/utils/profiling.py` exposes `with frame_section("name"):` — use it before claiming
  a perf win.
- Use generators and iterators for large datasets (scene traversal, animation sample iteration)
  to minimize memory footprint.
- Cache expensive computations with `functools.lru_cache` or manual caching where appropriate
  (shader compilation, bone-matrix palettes per skeleton per frame).
- Never call GL functions from Python in a tight per-vertex / per-bone loop. Push the data into
  a numpy buffer once, upload once, draw once.

### Threading & GL Rules

- The GL context is owned by **exactly one thread** — the render thread. Every `gl*` call MUST
  originate there. Asset loaders run on `QThread`s and emit signals carrying CPU-side data
  (numpy arrays, decoded pixel buffers); the render thread converts them to GL objects.
- Never share a Qt widget's GL context across threads. If background work needs a context (e.g.
  PBO uploads), create a shared offscreen context with explicit `QOpenGLContext::setShareContext`
  and document the lifetime.
- Scene mutations from the script host go through a thread-safe command queue drained at the
  start of each frame. The render thread reads an immutable snapshot — no locks in the hot path.

### Security

- Never hardcode secrets, API keys, tokens, or passwords in source code — use environment
  variables or secure config files.
- Validate and sanitize ALL external input (model files, image textures, user scripts, CLI
  arguments) at system boundaries. Reject malformed glTF JSON before touching binary buffers;
  cap mesh vertex/index counts to a configurable hard limit.
- Sanitize file paths to prevent path traversal — reject `..` segments and absolute paths from
  asset references inside model files (glTF `uri`, OBJ `mtllib`, FBX embedded media). Resolve
  every reference relative to the asset root and assert the resolved path is still under it.
- Apply the principle of least privilege — the scripting host runs in a restricted namespace.
  Forbid `eval`, `exec`, `compile`, `__import__`, `open`, `os`, `sys`, `subprocess`, and direct
  filesystem / network access. Expose only the curated `scene` / `nodes` / `time` / `input` API.
  See **Scripting Sandbox** below for the canonical implementation.
- Avoid `eval()`, `exec()`, `pickle.loads()` on untrusted data, and `subprocess` with `shell=True`.
  User scripts are NEVER run through `exec` directly — they go through the sandbox loader.
- Use secure defaults: HTTPS only for any remote asset fetch, strong hashing (SHA-256+) for asset
  cache keys, constant-time comparisons for any future signature checks.
- Log security-relevant events (rejected paths, malformed assets, sandbox violations) but never
  log secrets or full file contents.

### Unit Tests

Tests are not optional polish — they are part of the change. A feature without tests is an
incomplete feature and MUST NOT be committed. This rule applies equally to bug fixes (regression
test required) and refactors (existing behaviour must remain green; add a test if the refactor
exposes a previously untested path).

**Required coverage for every change:**

- **Happy path** — the new code does what it advertises on a representative input (a tiny inline
  glTF, a 12-vertex cube OBJ, a 2-bone skeleton).
- **Edge cases** — empty scenes, single-mesh scenes, missing optional channels (no normals, no
  UVs, no skin weights), degenerate triangles, identity transforms.
- **Error handling** — every `except` branch is exercised; malformed assets raise the documented
  `AssetError` subclass; sandbox violations raise `ScriptSecurityError`.
- **Boundary conditions** — values just inside and just outside any range, threshold, or enum
  (max bone count per vertex, max texture size, animation time at clip start/end).
- **Round-trips** — anything that serialises (scene snapshot, project file, script state) needs
  a `to_dict → from_dict → equal` test. Same for matrix decompose/compose.

**Required test types for every feature:**

- **Pure-helper tests.** Extract pure logic out of Qt / GL classes (math, parsing, culling,
  skinning matrix math) into helper modules and unit-test those directly without instantiating
  Qt widgets or a GL context. Cheap, fast, deterministic.
- **Qt smoke test.** If the feature has a dialog / dock, instantiate it under the `qapp` fixture
  and assert the visible state. Use `monkeypatch` to auto-confirm `QMessageBox` / `QFileDialog`
  instead of stubbing whole modules.
- **GL test (offscreen).** Renderer / shader / pass changes get an offscreen-context test that
  draws a known scene to an FBO, reads back pixels, and compares against a golden PNG with SSIM
  tolerance. Use `QOffscreenSurface` + `QOpenGLContext` so headless CI can run it.
- **Integration test where the wiring is non-obvious.** End-to-end import → scene-graph → render
  on small synthetic inputs. Script-host integration tests load a tiny script and assert the
  scene state after N simulated frames.

**Mechanics:**

- Use `pytest` style. Module-level functions and `Test*` classes are both fine; follow the
  style of the file you're adding to.
- Test file naming: `tests/test_<module_name>.py`. One test module per production module.
- Use the shared fixtures in `tests/conftest.py` (`qapp`, `gl_context`, `tmp_path`,
  `sample_gltf_cube`, `sample_skinned_mesh`). Do not roll your own QApplication or GL context.
- Tests that need a GL context use the `gl_context` fixture, which spins up an offscreen
  `QOpenGLContext`. Skip cleanly with `pytest.skip("no GL")` on systems where context creation
  fails — never let a CI runner failure silently mask a real regression.
- Never write to the user's real settings file. The autouse `_isolate_user_settings` fixture
  redirects the path; trust it and mutate `user_settings_dict` directly in tests.
- Run `py -m pytest tests/` before committing. If a test was already skipping because of a
  missing optional dependency, leave it skipping — but every NEW test must run, not skip.

### Linter & Static Analysis Compliance (SonarQube / Codacy / pylint / flake8 / ruff)

All new and modified code MUST pass the following rules without warnings. These mirror the
default rule sets of SonarQube, Codacy, pylint, flake8, ruff, and bandit for Python.

#### Complexity & Size

- **Cognitive complexity**: keep each function ≤ 15 (SonarQube `python:S3776`). Break nested
  branches into helper functions when exceeded.
- **Cyclomatic complexity**: keep each function ≤ 10 (pylint `R1260`, radon `C`).
- **Function length**: ≤ 75 logical lines. Split long functions into focused helpers.
- **File length**: ≤ 1000 lines (SonarQube `python:S104`). Split large modules.
- **Parameter count**: ≤ 7 per function (SonarQube `python:S107`). Group related params into a
  dataclass or dict when exceeded. (`RenderState`, `MaterialParams`, `ImportOptions`.)
- **Nesting depth**: ≤ 4 levels (SonarQube `python:S134`). Use early returns / guard clauses.
- **Boolean expression complexity**: ≤ 3 operators in one expression (SonarQube `python:S1067`).
  Extract to named booleans.
- **Return statements**: ≤ 6 per function (pylint `R0911`).
- **Local variables**: ≤ 15 per function (pylint `R0914`).

#### Duplication

- Do NOT copy-paste blocks of ≥ 3 statements across functions or files (SonarQube
  `common-python:DuplicatedBlocks`, Codacy duplication detector). Extract shared logic — VAO/VBO
  setup, uniform binding, attribute layout helpers, glTF accessor decoding all live in one place.
- Do NOT declare the same string literal ≥ 3 times (SonarQube `python:S1192`). Assign to a
  module-level constant. Shader uniform names belong in `posecascade/gl/uniforms.py` constants.

#### Naming (PEP 8)

- `snake_case` for functions, methods, variables, modules (SonarQube `python:S1542`, pylint `C0103`).
- `PascalCase` for classes (pylint `C0103`).
- `UPPER_CASE_WITH_UNDERSCORES` for module-level constants.
- `_leading_underscore` for private attributes / methods.
- No single-letter names except loop indices (`i`, `j`, `k`) or well-known math symbols
  (`x`, `y`, `z`, `u`, `v`, `n`, `t`, `b` for tangent-space, `m` for matrix in obvious local scope).

#### Errors & Exceptions

- Never use bare `except:` — always specify the exception type (SonarQube `python:S5754`,
  flake8 `E722`).
- Never write `except Exception: pass` without a logged reason and comment explaining why it is
  safe.
- Never catch `BaseException` directly (covers `KeyboardInterrupt`, `SystemExit`).
- Raise specific exception types — define a domain hierarchy: `PoseCascadeError` →
  `AssetError` (`MalformedAssetError`, `UnsupportedFormatError`), `GLError`, `SceneError`,
  `ScriptError` (`ScriptSecurityError`, `ScriptRuntimeError`).
- Chain exceptions with `raise X from err` to preserve context (ruff `B904`).
- Never use `assert` for runtime validation (assertions are stripped under `python -O`); use
  explicit `raise` instead. `assert` is only for invariants in tests.

#### Code Smells

- No unused imports, variables, or function parameters (pyflakes `F401`, `F841`, pylint `W0612`,
  `W0613`). Prefix intentionally unused params with `_`.
- No commented-out code. Delete it — git preserves history.
- No `print()` calls in production code; use the project's logger (`posecascade/utils/logging`).
- No `TODO` / `FIXME` / `XXX` left in merged code (SonarQube `python:S1135`). File a ticket
  instead.
- No magic numbers — extract to `UPPER_CASE` constants (SonarQube `python:S109`). Exceptions:
  `0`, `1`, `-1`, `2` in obvious contexts. Common 3D constants (`MAX_BONES_PER_VERTEX = 4`,
  `MAX_LIGHTS_FORWARD = 8`, `SHADOW_MAP_SIZE = 2048`) live in `posecascade/render/constants.py`.
- Use `is None` / `is not None` (never `== None` / `!= None`) (pycodestyle `E711`).
- Use `isinstance(x, T)` instead of `type(x) == T` (pycodestyle `E721`).
- No mutable default arguments (`def f(x=[])`) — use `None` and assign inside (ruff `B006`,
  pylint `W0102`).
- No global mutable state; if unavoidable, encapsulate in a module-level class or singleton
  (the GL context wrapper, the asset cache, the script host registry).
- Prefer f-strings over `.format()` or `%` (ruff `UP032`).
- Always use context managers (`with` blocks) for file / resource handles (ruff `SIM115`). The
  GL helpers expose `with bind(vao):` / `with framebuffer(fbo):` — use them.
- Prefer `dict.get(key, default)` over `if key in dict: ... else: ...` (ruff `SIM401`).
- Use comprehensions / generator expressions instead of `map` + `lambda` or manual `append`
  loops when clearer.
- Close / release Qt and GL resources (`deleteLater`, `disconnect`, `glDeleteBuffers`,
  `glDeleteTextures`) to prevent leaks. Wrap each GL object in a Python class whose
  `__del__` / `close()` deletes it on the GL thread.

#### Security (bandit / SonarQube `python:S*` security rules)

- `pickle.load(s)` on untrusted data is forbidden (`B301`, SonarQube `python:S5135`). Project
  files are JSON or msgpack with a strict schema.
- `yaml.load` without `SafeLoader` is forbidden — use `yaml.safe_load` (`B506`).
- MD5 / SHA-1 are forbidden for security purposes — use SHA-256+ or bcrypt / argon2 (`B303`,
  `B304`, SonarQube `python:S4790`). Allowed for non-security uses (asset cache keys, file
  de-duplication) ONLY with `usedforsecurity=False`.
- `subprocess` with `shell=True` is forbidden when any argument comes from user input (`B602`).
- Never use `eval`, `exec`, `compile` on dynamic input (`B307`). The scripting host is the
  ONLY place that loads user code, and it does so via a restricted-namespace `exec` with
  `# nosec B102  # restricted globals — see scripting/sandbox.py` and explicit input
  validation.
- Never use `tempfile.mktemp()` — use `tempfile.mkstemp()` or `NamedTemporaryFile` (`B306`).
- Network binds must not use `0.0.0.0` unless intentional and documented (`B104`).
- XML parsing (COLLADA `.dae` is XML) must use `defusedxml`, never stdlib `xml.etree` on
  untrusted input (`B405`–`B411`).
- Random number generation for security must use `secrets`, not `random` (`B311`). Procedural
  generation / particle jitter MAY use `random` and should pin a seed for reproducibility.

#### Typing & Documentation

- Public functions and methods SHOULD have type hints on parameters and return type. Use
  `numpy.typing.NDArray[np.float32]` for vertex / matrix arrays, never bare `np.ndarray`.
- Public modules and classes SHOULD have a one-line docstring describing their purpose.
- Private helpers may omit docstrings if names are self-explanatory.
- Shaders carry a header comment naming the pass, expected vertex attributes, and uniform set.

#### Enforcement

When writing or modifying code, mentally check each function against the above rules before
finalising. If unavoidable rule violation (e.g. Qt callback signature forces extra parameters,
or a GL call genuinely needs `bytes` decoded), add a `# noqa: <rule>` or equivalent suppression
with a brief justification comment on the same line.

## Project-Specific Compliance Patterns

### Engine Core vs Importers

The line between `posecascade/` (the main package) and `importers/<format>/` is **not** "anything
file-format-related goes in importers" — it's **dependency surface and failure isolation**.

**A feature is an importer plugin when ANY of the following is true:**

1. It needs a **heavy / optional runtime dependency** that we don't want to force on every user
   (e.g. `pyfbx` / Autodesk FBX SDK for `.fbx`, `pyusd` for USD, `pycollada` for `.dae`).
2. It needs **failure isolation** — a malformed third-party file should never bring down the
   viewport.
3. It needs **independent release cadence** — format support can be updated without re-shipping
   the engine.

**A feature stays in the engine core when:**

- It runs on the default dep set (numpy, Pillow, PySide6, PyOpenGL, defusedxml).
- It's part of the everyday import / view / animate workflow that all users should see by default
  (glTF/GLB, OBJ, STL, PLY are core; FBX, USD, COLLADA are plugins).

#### Directory rules

- **Engine core**: `posecascade/<area>/<feature>.py` for pure logic, `posecascade/ui/<feature>.py`
  for the Qt front-end, registration in `posecascade/app/registry.py`.
- **Importer plugin**: `importers/<format>/__init__.py` (sets `importer_class`),
  `importers/<format>/importer.py` for the adapter, and **all format-internal logic lives INSIDE
  the importer directory**. Never put format-internal parsers under `posecascade/assets/`.
- **Shaders**: bundled GLSL in `shaders/<pass>/*.{vert,frag,geom,comp}`. User-overridable shaders
  go to `~/.posecascade/shaders/` (gitignored) and the engine merges that path on top of the
  bundled set at startup.
- **Models bundled for examples**: `examples/assets/` (gitignored if large; small public-domain
  cubes/spheres only in repo).

#### Testing importer-internal modules

Importers are not on the default `sys.path` — at runtime `posecascade/assets/importer_manager.py`
prepends `importers/` so each format folder becomes importable as a package. `tests/conftest.py`
mirrors that injection at session-collect time, which lets tests in `tests/` import importer
modules with `from <format>.<module> import …`. Do not duplicate the path injection in
individual test files.

#### When in doubt

Ask: "if a user installs PoseCascade with the default `requirements.txt` and never enables an
importer plugin, should this feature work?" If yes → core. If no → importer plugin.

### Scripting Sandbox

User scripts that animate scenes are loaded through `posecascade/scripting/sandbox.py`. The
canonical pattern is:

1. Read the script source from the project file (always relative to the project root,
   path-traversal-checked).
2. Build a restricted globals dict containing only the curated API objects
   (`scene`, `nodes`, `time`, `input`, `math`, `vec3`, `quat`, `lerp`, `clamp`, `noise`).
   Explicitly set `__builtins__` to a minimal whitelist (`len`, `range`, `min`, `max`, `abs`,
   `round`, `enumerate`, `zip`, `print` → routed to logger).
3. Compile with `compile(source, filename, "exec")` so syntax errors carry the user's filename.
4. Execute with `exec(compiled, restricted_globals)` — this is the ONE allowed `exec` call in
   the codebase, and it carries `# nosec B102  # restricted globals; see sandbox.py docstring`.
5. Pull the user's `update(dt)` (or `start()`, `on_event(...)`) callable out of
   `restricted_globals` and store it on the script host.
6. Wrap every per-frame call in a try/except that converts any user exception into a logged
   `ScriptRuntimeError` and disables the offending script — one bad script must never freeze
   the timeline.

Do NOT add new `exec` / `eval` / `compile` calls anywhere else. If you find yourself wanting to,
extend the sandbox API instead.

### Asset Loading & Path Safety

- **All asset path resolution MUST go through `posecascade/assets/path_safety.py::resolve_safe`.**
  It takes a `root` (the project or asset bundle root) and a `reference` (relative path from a
  model file), rejects `..`, rejects absolute paths, rejects symlinks pointing outside `root`,
  and returns a real, absolute path inside `root`.
- Do NOT call `Path.resolve()` directly on user-controlled strings in new code. Import
  `resolve_safe` instead.
- glTF embedded `data:` URIs are decoded in-memory with a hard size cap
  (`MAX_EMBEDDED_BUFFER_BYTES`). Reject anything larger up-front.
- Any future remote-asset fetch (`http(s)://` in a model) is gated behind a `_https_urlopen`
  guard mirroring the Imervue pattern: parse the URL with `urllib.parse.urlparse`, reject any
  scheme other than `https`, then call `urlopen` with `# nosec B310  # scheme validated above`.

### Suppression Comment Conventions

Use the right comment for the right tool. They are NOT interchangeable.

| Tool          | Comment form                            | Placement   | Notes                                               |
|---------------|-----------------------------------------|-------------|-----------------------------------------------------|
| ruff / flake8 | `# noqa: <CODE>` (e.g. `# noqa: S310`)  | line-level  | Must list specific codes — never bare `# noqa`.     |
| bandit        | `# nosec B<NNN>` (e.g. `# nosec B102`)  | line-level  | ruff's `# noqa` does NOT suppress bandit.           |
| SonarCloud    | `# NOSONAR`                             | line-level  | Use for hotspots that cannot be config-skipped.     |
| pylint        | `# pylint: disable=<name>`              | line-level  | Prefer refactor over suppression.                   |

Every suppression MUST include a brief justification on the same line
(`# nosec B102  # restricted globals; see sandbox.py`). Unexplained suppressions will not pass
review.

### Project-Wide Skip Configuration

Systemic false positives are skipped at the config level, never with per-line comments. The
authoritative skip lists live in:

- `.bandit` (YAML, with per-rule justification comments) — the canonical source.
- `pyproject.toml` `[tool.bandit]` — mirror for tooling that only reads `pyproject.toml`. Keep
  both files in sync.

When adding a new bandit skip:
1. Add it to `.bandit` with a `# B<NNN>: <one-line reason>` comment.
2. Mirror it in `pyproject.toml` `[tool.bandit].skips`.
3. Verify locally: `py -m bandit -c pyproject.toml -r posecascade/` must return
   `No issues identified`.

### Local CI Reproduction

Before pushing, reproduce each engine locally so CI does not have to tell you:

- **bandit**: `py -m bandit -c pyproject.toml -r posecascade/`
  (the `-c` flag is REQUIRED — without it, bandit ignores the skip config).
- **ruff**: `py -m ruff check .`
- **pytest**: `py -m pytest tests/`
- **offscreen render tests**: `py -m pytest tests/render/ -k "golden"` — runs against the
  offscreen Qt platform and compares against `tests/golden/*.png` with SSIM tolerance.

### Environment

- Python 3.14.4 in the project-local `.venv/`. Activate with
  `.venv\Scripts\Activate.ps1` (PowerShell) or `.venv\Scripts\activate.bat` (cmd) before running
  `py -m ...` commands, OR call the venv interpreter directly:
  `.venv\Scripts\python.exe -m pytest tests/`.
- Required runtime deps (target): `PySide6`, `PyOpenGL`, `numpy`, `Pillow`, `defusedxml`,
  `pygltflib` (or hand-rolled glTF parser), `trimesh` (optional, for STL/PLY/OBJ helpers).
- Required dev deps: `pytest`, `pytest-qt`, `ruff`, `bandit`, `pillow` (golden-image diff),
  `scikit-image` (SSIM).
