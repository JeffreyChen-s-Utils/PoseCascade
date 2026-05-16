# Development

Contributor / maintainer notes. End-user content (features and how to
use them) lives in [`README.md`](README.md) and [`docs/`](docs/).

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
├── docs/                         # user-facing documentation
├── schemas/                      # JSON schemas (declarative animation)
├── setup.py                      # cythonize build hook
└── pyproject.toml                # project metadata + ruff / bandit config
```

## Definition of Done

Every change must satisfy three gates before commit (see [`CLAUDE.md`](CLAUDE.md)
for the full list). Reproduce locally:

```bash
.venv/Scripts/python.exe -m pytest tests/             # unit + offscreen-GL
.venv/Scripts/python.exe -m ruff check .              # lint + style
.venv/Scripts/python.exe -m bandit -c pyproject.toml -r posecascade/
```

The `-c` flag on bandit is **required** — without it bandit ignores
the project skip config and the run is noisy.

## Rebuilding the Cython kernel

The cloth Cython kernel must be re-built in place after any change to
`posecascade/animation/_cloth_kernels.pyx`:

```bash
.venv/Scripts/python.exe setup.py build_ext --inplace
```

## Tests

Tests mirror the package layout: each production module
`posecascade/<area>/<feature>.py` has a paired
`tests/test_<feature>.py`. Run the whole suite:

```bash
.venv/Scripts/python.exe -m pytest tests/
```

GL-heavy tests use the `gl_context` fixture which spins up an
offscreen `QOpenGLContext` — they skip cleanly with
`pytest.skip("no GL")` on systems where context creation fails, so
headless CI runners don't false-fail.

Golden-image tests under `tests/render/` diff the rendered frame
against `tests/golden/*.png` using SSIM with a documented per-test
tolerance.

## Linting + security

Ruff catches most style issues automatically. The project enforces
SonarQube / Codacy / pylint default rules (complexity ≤ 15, function
length ≤ 75 lines, file length ≤ 1000 lines, no magic numbers, no
bare `except`, no mutable default args). Bandit scans for
security-relevant patterns (`pickle.load`, `yaml.load` without
SafeLoader, MD5 / SHA-1 for security, `shell=True`).

Project-wide skips live in `.bandit` and mirror in `pyproject.toml`
`[tool.bandit]`. Per-line suppressions need a brief justification on
the same line, e.g.
`# nosec B102  # restricted globals; see sandbox.py`.

## Performance notes

The recent optimisation focus has been on the cloth solver and the
renderer hot path.

**Cloth** — 480-vert skirt benchmark
(`posecascade.mcp.server.cloth_benchmark`):

| Stage                                  | ms/step (best) | vs baseline |
|----------------------------------------|---------------:|------------:|
| Baseline (pre-tuning)                  |          3.225 |           — |
| NumPy: einsum + combined-bincount      |          2.085 |        −35% |
| Cython kernel                          |          0.356 |    **−89%** |
| Cython + broad-phase + bin culling     |          0.36–0.38 (single-bin colliders save 30%) | — |

On the bundled 30 k-vert body mesh in the showcase scene, the GPU
compute skinning path turns the same passive_skin LBS + collider push
from a ~9 ms CPU cost into well under 0.05 ms — written directly into
the mesh's existing position + normal VBOs with no CPU readback.

**Renderer** — showcase scene at 768×768 with shadow + projected
shadow + ground + outline + toon, driven by `tools/bench_renderer.py`:

| Stage                                  | ms/frame |    FPS |
|----------------------------------------|---------:|-------:|
| Baseline                               |     7.88 |    127 |
| Per-frame uniform-state cache          |     6.09 |    164 |
| + glUseProgram hoisting in shadow pass |     5.04 |    198 |
| + batched bone-matrix matmul           | **~4.50**| **~220**|

Wins come from a per-program "already uploaded this frame" cache that
skips ~85% of `glUniformMatrix4fv` / `ascontiguousarray` calls,
hoisting `glUseProgram` out of per-mesh inner loops in the depth +
projected-shadow passes, and replacing per-joint Python matmul in
`_compute_bone_matrices` with a single batched `np.matmul`.

**Declarative runtime** — per-frame parent-world rotation cache, an
`lru_cache`-backed AST parse for the expression DSL, and an
`extends` resolver that runs at load time (no per-frame cost). On
showcase the JSON-driven update step holds steady around 0.6 ms.

Per-frame timing for the renderer hot path is wrapped in
`posecascade.utils.profiling.frame_section`. `tools/bench_renderer.py`
drives the showcase scene through a fixed number of frames and
prints the breakdown.

## CI + release pipeline

Two GitHub Actions workflows wire the project up:

* **`tests.yml`** runs `ruff` + `bandit` + the full `pytest` suite
  on every pull request and every push to `main`. Three Python
  versions (3.12 / 3.13 / 3.14) on Ubuntu. Uses
  `QT_QPA_PLATFORM=offscreen` so PySide6 fixtures construct
  without a real display.
* **`wheels.yml`** is the single-workflow release pipeline. On
  every push to `main` it runs six jobs in order:
  1. **`compute_version`** — reads the latest `v*` tag and bumps
     the patch component (or minor / major if the commit subject
     contains `[release minor]` / `[release major]`).
  2. **`build_wheels`** — cibuildwheel matrix across Win / macOS /
     Linux × cp312 / cp313 / cp314. The computed version is
     injected into each build via `SETUPTOOLS_SCM_PRETEND_VERSION`.
  3. **`build_sdist`** — source distribution with the same version.
  4. **`publish`** — uploads wheels + sdist to PyPI through
     Trusted Publishing.
  5. **`build_exe`** — Nuitka `--onefile` standalone Windows
     executable, same version baked in. See
     [`docs/packaging_nuitka.md`](docs/packaging_nuitka.md) for the
     full Nuitka invocation.
  6. **`tag_and_release`** — tags the source commit `v<version>`,
     pushes the tag, and creates / updates the GitHub Release with
     `PoseCascade.exe` attached. The release body points users at
     `pip install posecascade` for the package install path.

Keeping everything in one workflow run sidesteps GitHub Actions'
anti-recursion rule that blocks downstream-workflow triggers from
`GITHUB_TOKEN` pushes — no Personal Access Token is required.

`pull_request` builds run `build_wheels` + `build_sdist` only, so a
contributor's PR exercises the wheel build path without publishing.
`workflow_dispatch` is available for manual reruns; tag pushes
re-trigger the publish path against the existing tag.

### Bump rules

| Head commit message contains | Bump                              |
|------------------------------|-----------------------------------|
| `[release major]`            | major (e.g. `v1.2.3` → `v2.0.0`)  |
| `[release minor]`            | minor (e.g. `v1.2.3` → `v1.3.0`)  |
| (anything else)              | patch (e.g. `v1.2.3` → `v1.2.4`)  |
| `[skip release]`             | no release                        |
| `[skip ci]`                  | no release (and no CI overall)    |

### Versioning

Versions are derived from git tags via
[setuptools-scm](https://github.com/pypa/setuptools_scm).

* Tag-build: version = tag minus the `v` prefix.
* Non-tag build from a git checkout: version =
  `{next-tag}.devN+g{shorthash}` — N is the commit-count since the
  last tag.
* Non-tag build from an sdist with no git metadata: falls back to
  `0.0.0` (`[tool.setuptools_scm] fallback_version`).

`local_scheme = "no-local-version"` keeps non-tag builds PEP 440
clean so TestPyPI / PyPI accept them.

### One-time PyPI configuration

The publish step uses
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/),
so **no API token is stored in the repo**. The maintainer configures
this once on PyPI:

1. Sign in to <https://pypi.org/> as a maintainer of the
   `posecascade` project. (If the project doesn't exist yet, the
   first release uses the
   [pending publisher flow](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).)
2. Project → **Manage** → **Publishing**.
3. Add a GitHub publisher:
   * Owner: `JeffreyChen-s-Utils`
   * Repository: `PoseCascade`
   * Workflow filename: `wheels.yml`
   * Environment name: `release`
4. Save.

The matching `environment: release` in `wheels.yml` is what PyPI
checks against. If you change the environment name in either file,
change it in both.

### Reverting / yanking a release

PyPI's "Yank" is one-click from the project's release page and is
preferable to deleting the version. After yanking, push another
commit to `main` and the patch bump produces a new version that
supersedes the yanked one.

### Local dry-run

```bash
# Read the latest tag.
git tag --list 'v*' --sort=-v:refname | head -n1

# Inspect what setuptools-scm would call this build.
.venv/Scripts/python.exe -m setuptools_scm
```
