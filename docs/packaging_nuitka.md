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

## Minimum command

```bash
python -m nuitka \
  --standalone \
  --onefile \
  --enable-plugin=pyside6 \
  --enable-plugin=numpy \
  --include-package=importers \
  --include-data-dir=shaders=shaders \
  --include-data-dir=schemas=schemas \
  --include-data-dir=importers=importers \
  --output-dir=dist \
  --output-filename=PoseCascade \
  posecascade/__main__.py
```

On Windows, additionally pass:

```bat
  --windows-disable-console
  --windows-icon-from-ico=path\to\PoseCascade.ico
```

On macOS, additionally pass:

```bash
  --macos-create-app-bundle
  --macos-app-icon=path/to/PoseCascade.icns
  --macos-app-name=PoseCascade
```

The first build takes 3–10 minutes depending on hardware (Nuitka
re-emits C for every imported module). Subsequent builds reuse the
build cache and run in ~30 s.

## What each flag does

| Flag                              | Why it's needed                                              |
|-----------------------------------|--------------------------------------------------------------|
| `--standalone`                    | Bundles the Python interpreter + every dependency into a self-contained directory. Without this, the output binary needs the host's Python install. |
| `--onefile`                       | Compresses the standalone dir into a single executable that self-extracts to a temp dir on launch. Drops `--onefile` if you'd rather ship the directory layout (smaller binary, faster cold start, but multi-file distribution). |
| `--enable-plugin=pyside6`         | Nuitka's PySide6 plugin handles the Qt6 shared libs, the Qt platform / image plugins, and the `Q_OBJECT` metatype registration that pure source compilation breaks. **Without this, the app crashes at `QApplication()` startup.** |
| `--enable-plugin=numpy`           | Nuitka's NumPy plugin sets up the C-side numpy ABI for the Cython kernel + per-vert math hot paths. Forgetting this often surfaces as a cryptic "could not find numpy.core.multiarray" at import time. |
| `--include-package=importers`     | The per-format importer plugins (`importers/gltf/`, `importers/pmx/`, …) live outside the `posecascade` package; tell Nuitka to compile them anyway and ship their sources alongside the executable. |
| `--include-data-dir=shaders=shaders` | Ship the GLSL pass tree. `LHS=RHS` form: LHS is the on-disk source dir; RHS is the relative path inside the produced bundle. |
| `--include-data-dir=schemas=schemas` | Ship the declarative-animation JSON Schema. |
| `--include-data-dir=importers=importers` | Mirror the importer dirs at runtime — Nuitka's compiled bytecode doesn't carry the per-importer `__init__.py` discovery files in some configurations, and shipping them as data files is the most robust path. |
| `--output-dir=dist`               | Keep build artefacts off the project root. |
| `--output-filename=PoseCascade`   | Sets the final binary's name. |

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

## Smoke testing the bundle

```bash
# Cold-start: should open the empty editor in well under a second.
dist/PoseCascade --log-level INFO

# Load a bundled scene + animation.
dist/PoseCascade \
  --scene examples/assets/herta/herta.glb \
  --script examples/scripts/showcase.json
```

On the same hardware, a Nuitka `--onefile` build typically cold-starts
in 2× the time of the equivalent `--standalone` directory build —
the `.zst` payload extraction is the bottleneck. Drop `--onefile`
if cold-start latency matters more than single-file distribution.

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

| Dimension                       | PyInstaller          | Nuitka                  |
|---------------------------------|----------------------|-------------------------|
| First-build wall time           | ~30 s                | 3–10 minutes            |
| Re-build wall time (cache hit)  | ~10 s                | ~30 s                   |
| Output size (`--onefile`)       | ~120 MB              | ~80 MB                  |
| Cold-start latency              | medium (`--onefile`) | low (`--standalone`), medium (`--onefile`) |
| Runtime speed in pure-Python    | interpreted          | compiled — ~10-30% faster on hot Python sections |
| Debugging when bundle breaks    | clearer tracebacks   | C-level errors first     |
| Cross-compilation               | not really           | not really               |

Renderer + cloth hot paths run identically — both bundlers pick up
the same compiled extension modules, so the Cython kernel and the
GPU compute shader perform the same regardless of which bundler
produced the executable.
