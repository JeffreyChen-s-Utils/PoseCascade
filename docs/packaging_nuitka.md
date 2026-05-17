# Packaging with Nuitka

[Nuitka](https://nuitka.net/) compiles Python source to C and links
it against the CPython runtime, producing a standalone binary that
typically starts faster, runs faster on long Python sections, and
ships smaller than the equivalent PyInstaller bundle. The cost is a
significantly longer build (minutes instead of seconds) and a more
fiddly setup for native dependencies like PySide6 and the cloth
Cython kernel.

If your priority is **first-iteration developer experience**, prefer
[PyInstaller](packaging_pyinstaller.md). If your priority is **end-
user cold-start latency** or **binary size**, Nuitka is the better
fit. The two routes are not mutually exclusive — keeping both
spec files up to date is fine and lets you A/B the trade-off per
release.

## Install

```bash
pip install "nuitka>=2.5" ordered-set zstandard
```

* `nuitka` itself is the compiler.
* `ordered-set` speeds up Nuitka's symbol bookkeeping (Nuitka warns
  on startup if missing).
* `zstandard` is required for `--onefile` mode (Nuitka uses Zstd to
  compress the embedded payload).

You also need a working C / C++ toolchain on the build host —
Microsoft Build Tools on Windows, `build-essential` on Debian-based
Linux, Xcode command-line tools on macOS. Nuitka prints a one-line
fix-up command if it can't find one.

## Minimum command (matches the shipping CI build)

```bash
python -m nuitka \
  --standalone \
  --assume-yes-for-downloads \
  --enable-plugin=pyside6 \
  --enable-plugin=numpy \
  --include-package=posecascade \
  --include-package=importers \
  --include-data-dir=shaders=shaders \
  --include-data-dir=schemas=schemas \
  --include-data-dir=importers=importers \
  --output-dir=dist \
  --output-filename=PoseCascade.exe \
  posecascade/__main__.py
```

Note: this produces a **folder** (`dist/__main__.dist/`) containing
`PoseCascade.exe` plus all its runtime dependencies. The folder is what
ships to end users — the *Why standalone, not onefile* section below
explains why we don't compress it into a single self-extracting binary.
The Windows CI workflow renames the folder to `dist/PoseCascade/` and
zips it to `dist/PoseCascade-windows-x86_64.zip` for the GitHub Release.

On Windows, additionally pass:

```bat
  --windows-disable-console
  --windows-icon-from-ico=path\to\PoseCascade.ico
```

`--windows-disable-console` suppresses the black command-prompt window;
drop it during development so the logger output stays visible — without
it, a startup crash gives the user nothing to look at.

On macOS, additionally pass:

```bash
  --macos-create-app-bundle
  --macos-app-icon=path/to/PoseCascade.icns
  --macos-app-name=PoseCascade
```

The first build takes 3–10 minutes depending on hardware (Nuitka
re-emits C for every imported module). Subsequent builds reuse the
build cache and run in ~30 s.

## Why standalone, not onefile

The `--onefile` mode wraps the standalone directory in a single
self-extracting executable. It's tempting for end-user distribution
(one file to download, one icon to double-click) but in our setup it
fails to launch on a clean machine: when the exe self-extracts to its
temp directory, the bundled `posecascade/` package ends up on Python's
module path under the wrong name and the very first
`from posecascade.app.bootstrap import …` raises `ModuleNotFoundError`.

`--standalone` ships the folder directly. The launch path is
`<extracted>/PoseCascade.exe` and the package is visible at the
expected import name. The trade-off is multi-file distribution — which
we paper over by zipping the folder for the GitHub Release.

The `--include-package=posecascade` flag is **mandatory** when the
entry script is `posecascade/__main__.py`: without it, Nuitka adds
`posecascade/` (the directory) to `sys.path`, but the runtime
`from posecascade.app.bootstrap import …` then fails because the
package name isn't on the path. Including the package explicitly puts
it on the bundled path under its real name.

## What each flag does

| Flag                              | Why it's needed                                              |
|-----------------------------------|--------------------------------------------------------------|
| `--standalone`                    | Bundles the Python interpreter + every dependency into a self-contained directory. Without this, the output binary needs the host's Python install. |
| `--assume-yes-for-downloads`      | Auto-accepts the optional ccache / dependency-walker downloads Nuitka offers on first build. Required for CI; harmless for interactive builds. |
| `--enable-plugin=pyside6`         | Nuitka's PySide6 plugin handles the Qt6 shared libs, the Qt platform / image plugins, and the `Q_OBJECT` metatype registration that pure source compilation breaks. **Without this, the app crashes at `QApplication()` startup.** |
| `--enable-plugin=numpy`           | Nuitka's NumPy plugin sets up the C-side numpy ABI for the Cython kernel + per-vert math hot paths. Forgetting this often surfaces as a cryptic "could not find numpy.core.multiarray" at import time. |
| `--include-package=posecascade`   | **Mandatory** when the entry script is `posecascade/__main__.py`. Without it Nuitka puts the package's directory on `sys.path` but not the package name, so the very first `from posecascade.app.bootstrap import …` fails with `ModuleNotFoundError`. |
| `--include-package=importers`     | The per-format importer plugins (`importers/gltf/`, `importers/pmx/`, …) live outside the `posecascade` package; tell Nuitka to compile them anyway and ship their sources alongside the executable. |
| `--include-data-dir=shaders=shaders` | Ship the GLSL pass tree. `LHS=RHS` form: LHS is the on-disk source dir; RHS is the relative path inside the produced bundle. |
| `--include-data-dir=schemas=schemas` | Ship the declarative-animation JSON Schema. |
| `--include-data-dir=importers=importers` | Mirror the importer dirs at runtime — Nuitka's compiled bytecode doesn't carry the per-importer `__init__.py` discovery files in some configurations, and shipping them as data files is the most robust path. |
| `--output-dir=dist`               | Keep build artefacts off the project root. |
| `--output-filename=PoseCascade.exe` | Sets the final binary's name (`PoseCascade.exe` lives inside `dist/__main__.dist/`). |

If your bundle ships example assets, mirror them too:

```
--include-data-dir=examples/assets/herta=examples/assets/herta
--include-data-dir=examples/assets/march7th=examples/assets/march7th
--include-data-dir=examples/scripts=examples/scripts
```

## Cython kernel

Nuitka does **not** re-compile the cloth Cython kernel from its
`.pyx` source — it picks up the already-compiled `.pyd` (Windows) /
`.so` (Linux / macOS) sitting next to `_cloth_kernels.pyx`. Build
that file in-place first:

```bash
python setup.py build_ext --inplace
```

After that, Nuitka's standalone discovery finds the compiled
extension automatically. If the kernel is missing at bundle time
the build still succeeds but the runtime silently falls back to
NumPy — same behaviour as a dev install without a C compiler. Add
`--show-modules` to the Nuitka command to verify the kernel was
picked up:

```
posecascade.animation._cloth_kernels  module compiled (extension)
```

## Per-OS notes

### Windows

* `--windows-disable-console` suppresses the black command-prompt
  window. Drop it to keep the engine's logger output visible — most
  users want it suppressed for shipping.
* `--mingw64` switches to the bundled MinGW64 toolchain; default is
  MSVC. MinGW64 produces slightly smaller binaries; MSVC produces
  slightly faster ones. Either works.
* Sign the resulting `.exe` with `signtool sign /a /tr
  http://timestamp.digicert.com /td sha256 /fd sha256 PoseCascade.exe`
  if you have a code-signing certificate; otherwise SmartScreen
  flags first-launch downloads as untrusted.

### macOS

* `--macos-create-app-bundle` produces `PoseCascade.app` in the
  output dir. The first launch of an unsigned `.app` requires the
  user to right-click → Open (or `xattr -d com.apple.quarantine
  PoseCascade.app` from the command line).
* For App Store / external distribution, sign + notarize after the
  bundle is created — Nuitka does not yet drive `codesign` /
  `notarytool` itself.
* macOS legacy GL caps at 4.1, so the GPU compute passive-skin path
  falls back to CPU LBS. Same as PyInstaller — purely a runtime
  capability of the platform, not a packaging concern.

### Linux

* Build inside a `manylinux2014_x86_64` container to maximise glibc
  compatibility. Nuitka respects the container's libstdc++ / libgcc
  versions and produces binaries that run on older distros.
* If your target is one specific distro, build natively for tighter
  binaries.
* The Qt platform plugin needs `libxcb-cursor0` on Qt 6.5+ runtimes;
  document that as a runtime dep for end users.

## Packaging the folder for distribution

The CI workflow renames `dist/__main__.dist/` → `dist/PoseCascade/`
and zips it for the GitHub Release attachment:

```pwsh
# PowerShell, run from the repo root after the Nuitka build above.
Move-Item dist/__main__.dist dist/PoseCascade
Compress-Archive -Path dist/PoseCascade -DestinationPath dist/PoseCascade-windows-x86_64.zip
```

The zip is what end users download. They extract it anywhere and run
`PoseCascade.exe` from the extracted folder; the exe needs every sibling
file in the folder, so moving the exe out by itself breaks it.

## Smoke testing the bundle

```bash
# Cold-start: should open the empty editor in well under a second.
dist/PoseCascade/PoseCascade.exe --log-level INFO

# Load a bundled scene + animation.
dist/PoseCascade/PoseCascade.exe \
  --scene examples/assets/herta/herta.glb \
  --script examples/scripts/showcase.json
```

If the exe fails to launch silently, **temporarily drop
`--windows-disable-console`** from the Nuitka command and re-build —
the console window will surface the traceback so you can see whether
it's a missing `--include-package`, a missing `--include-data-dir`,
or a runtime issue.

On the same hardware, a Nuitka `--standalone` build cold-starts in
roughly half the time of the equivalent `--onefile` build — the
former runs the exe directly, the latter has to self-extract the
`.zst` payload to a temp directory first. The folder layout is also
easier to debug when something goes wrong (you can `dir` the bundle
and check that `shaders/`, `schemas/`, and `posecascade/i18n/locales/`
are all where you expect).

## Common failures

| Symptom                                                   | Fix                                                                  |
|-----------------------------------------------------------|----------------------------------------------------------------------|
| `Could not find or load the Qt platform plugin "windows"` | `--enable-plugin=pyside6` is missing.                                |
| `ModuleNotFoundError: importers`                          | Add `--include-package=importers` AND `--include-data-dir=importers=importers`. |
| `numpy.core.multiarray failed to import`                  | Missing `--enable-plugin=numpy`.                                     |
| Cloth visibly slower in the bundle than in dev            | `_cloth_kernels.pyx` not compiled before bundling. Re-run `setup.py build_ext --inplace`, then rebuild. |
| GLSL `version` directive errors at first frame            | Shaders not bundled. Add `--include-data-dir=shaders=shaders`.        |
| `RecursionError` deep in `Qt` metatype resolution         | Build host has the wrong `ordered-set` / `zstandard` versions. Reinstall both and rebuild.    |
| Build host OOMs partway through C compilation             | Pass `--jobs=2` (or lower) so fewer translation units compile in parallel. |

## Build-time vs run-time trade-offs (vs PyInstaller)

| Dimension                       | PyInstaller          | Nuitka                                                       |
|---------------------------------|----------------------|--------------------------------------------------------------|
| First-build wall time           | ~30 s                | 3–10 minutes                                                 |
| Re-build wall time (cache hit)  | ~10 s                | ~30 s                                                        |
| Output size (`--standalone`)    | ~120 MB              | ~80 MB                                                       |
| Cold-start latency              | medium               | **low** (`--standalone` runs the exe directly)               |
| Runtime speed in pure-Python    | interpreted          | compiled — ~10-30 % faster on hot Python sections            |
| Debugging when bundle breaks    | clearer tracebacks   | C-level errors first                                         |
| Cross-compilation               | not really           | not really                                                   |

Renderer + cloth hot paths run identically — both bundlers pick up
the same compiled extension modules, so the Cython kernel and the
GPU compute shader perform the same regardless of which bundler
produced the executable.

## What ships with the bundle (folder contents)

After the build + rename, `dist/PoseCascade/` contains:

```text
dist/PoseCascade/
├── PoseCascade.exe                  ← entry point
├── python3.dll                      ← embedded Python runtime
├── PySide6/                         ← Qt shared libs + platform plugins
├── numpy/                           ← NumPy compiled extensions
├── posecascade/                     ← compiled engine package
│   └── i18n/locales/                ← every shipped locale catalog
├── importers/                       ← per-format importer plugins
├── shaders/                         ← GLSL pass tree
├── schemas/                         ← declarative-animation JSON Schema
└── …                                ← supporting DLLs the plugins pull in
```

The locale catalogs in `posecascade/i18n/locales/` ride along
automatically because `--include-package=posecascade` recursively
includes the package's data files. End users get every shipped
language out of the box. To add a community-contributed locale
without re-building, they can drop a new `<code>.json` next to
the existing catalogs — the loader auto-discovers it on next launch.
See [Internationalisation & responsive UI](internationalization.md)
for the catalog format.
