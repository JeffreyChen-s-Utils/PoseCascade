# Packaging with PyInstaller

[PyInstaller](https://pyinstaller.org/) bundles PoseCascade + the
Python runtime + every imported dependency into a directory or a
single-file executable that end users can run without installing
Python. PyInstaller is the easier of the two supported routes (the
other being [Nuitka](packaging_nuitka.md)) — faster to build, larger
output, and the runtime is still interpreted Python under the hood.

## Install

```bash
pip install pyinstaller
```

## Minimum command

The entry point is `posecascade/__main__.py`. PyInstaller bundles it
through `pip install -e .`'s `posecascade` script alias, so the
shortest invocation is:

```bash
pyinstaller \
  --name PoseCascade \
  --windowed \
  --icon assets/PoseCascade.ico \
  --collect-all PySide6 \
  --collect-all OpenGL \
  --collect-all Pillow \
  --collect-submodules pygltflib \
  --collect-submodules defusedxml \
  --add-data "shaders:shaders" \
  --add-data "schemas:schemas" \
  --add-data "importers:importers" \
  posecascade/__main__.py
```

`--add-data` uses `:` on macOS / Linux and `;` on Windows — pass the
appropriate separator for your build host.

`--icon assets/PoseCascade.ico` sets the file icon on Windows and
the dock icon on macOS. The same multi-resolution ICO drives the
Nuitka build — see [Packaging with Nuitka](packaging_nuitka.md). Edit
`assets/generate_icon.py` (not the binary) to tweak the design and
re-run it.

On Windows the equivalent is:

```bat
pyinstaller ^
  --name PoseCascade ^
  --windowed ^
  --icon assets\PoseCascade.ico ^
  --collect-all PySide6 ^
  --collect-all OpenGL ^
  --collect-all Pillow ^
  --collect-submodules pygltflib ^
  --collect-submodules defusedxml ^
  --add-data "shaders;shaders" ^
  --add-data "schemas;schemas" ^
  --add-data "importers;importers" ^
  posecascade/__main__.py
```

The output lands in `dist/PoseCascade/` — copy that directory to the
target machine; the `PoseCascade.exe` (Windows) or `PoseCascade`
(macOS / Linux) launcher is inside. Add `--onefile` to collapse the
directory into a single binary at the cost of a slower cold start
(the exe self-extracts into a temp directory on first run).

## What each flag does

| Flag                          | Why it's needed                                                 |
|-------------------------------|-----------------------------------------------------------------|
| `--name PoseCascade`          | Sets the output binary's name; default is the script's stem.    |
| `--windowed`                  | Suppresses the console window on Windows / macOS. Use `--console` if you want stdout for the engine's logger. |
| `--icon assets/PoseCascade.ico` | Sets the Windows file icon + macOS dock icon. The same ICO drives both PyInstaller and Nuitka builds; regenerate via `py assets/generate_icon.py`. |
| `--collect-all PySide6`       | Bundles the Qt6 shared libraries plus the platform / image plugins (`platforms/qwindows.dll`, `imageformats/qjpeg.dll`, …). Without this, the app fails on `QApplication()` with "could not find or load the Qt platform plugin". |
| `--collect-all OpenGL`        | Bundles PyOpenGL's `accelerate` + per-GL-function dispatch modules. PyInstaller's static analysis misses some of these because PyOpenGL uses dynamic imports. |
| `--collect-all Pillow`        | Pulls in the format plugins (`PIL.JpegImagePlugin`, `PIL.WebPImagePlugin`, …) that the importer cache discovers by name at runtime. |
| `--collect-submodules pygltflib` | The glTF importer; static analysis misses sub-imports.       |
| `--collect-submodules defusedxml` | The COLLADA importer's XML parser.                          |
| `--add-data shaders:shaders`  | Bundles the GLSL pass tree the renderer reads at startup.       |
| `--add-data schemas:schemas`  | Bundles the declarative-animation JSON Schema.                  |
| `--add-data importers:importers` | The per-format importer plugins (`importers/gltf/`, `importers/pmx/`, …) live OUTSIDE the `posecascade` package, so they need an explicit `add-data` rule. |

If your bundle ships with bundled assets, add them too:

```
--add-data "examples/assets/herta:examples/assets/herta"
--add-data "examples/assets/march7th:examples/assets/march7th"
--add-data "examples/scripts:examples/scripts"
```

## Cython kernel

The cloth Cython kernel (`posecascade/animation/_cloth_kernels.pyx`)
must be **built in-place before running PyInstaller**, otherwise the
bundle ships without the compiled `.pyd` / `.so` and the engine
silently falls back to the slower NumPy path:

```bash
python setup.py build_ext --inplace
```

After that step, the compiled artefact sits next to its `.pyx`
source and PyInstaller picks it up automatically.

## .spec file

For repeatable builds, save the flags in a spec file. Generate the
initial spec with the command above, then edit `PoseCascade.spec` to
the form below and rebuild with `pyinstaller PoseCascade.spec`:

```python
# PoseCascade.spec
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules

pyside6 = collect_all("PySide6")
opengl = collect_all("OpenGL")
pillow = collect_all("Pillow")

datas = [
    ("shaders", "shaders"),
    ("schemas", "schemas"),
    ("importers", "importers"),
]
datas += pyside6[0] + opengl[0] + pillow[0]
binaries = pyside6[1] + opengl[1] + pillow[1]
hiddenimports = (
    pyside6[2] + opengl[2] + pillow[2]
    + collect_submodules("pygltflib")
    + collect_submodules("defusedxml")
)

a = Analysis(
    ["posecascade/__main__.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "test", "unittest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="PoseCascade",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=True, upx_exclude=[],
    name="PoseCascade",
)
```

The `excludes` list keeps the wheel small — none of `tkinter`,
`test`, `unittest` is needed at runtime.

## Per-OS notes

### Windows

* Install **Microsoft Build Tools** before the first `setup.py
  build_ext --inplace` so the Cython kernel compiles.
* On older PCs, the GL driver may report only OpenGL 3.0; PoseCascade
  needs 3.3 core. The bundle launches but the viewport stays black —
  refer users to driver-update guidance in
  [docs/en/index.rst](en/index.rst) Troubleshooting.

### macOS

* Add `--osx-bundle-identifier com.example.posecascade` so the
  resulting `.app` is a real app bundle the Finder can launch.
* For distribution outside the App Store you'll want to **codesign +
  notarize** the bundle. The `codesign_identity` and
  `entitlements_file` fields in the spec hook into PyInstaller's
  built-in `codesign` flow.
* macOS legacy GL caps at 4.1, so the GPU compute passive-skin path
  falls back to CPU LBS — visually identical, slower on huge meshes.

### Linux

* Bundle against the oldest glibc / Qt you support; downstream users
  on older distros otherwise hit "version GLIBC_X.Y not found" at
  startup. Build inside a manylinux container if you need wide
  compatibility.
* `--collect-all PySide6` pulls in libxcb-* shared libs; on minimal
  desktop installs (LXQt, headless servers + xvfb) you may still
  need to install one or two of `libxcb-icccm4 libxcb-image0
  libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0
  libxcb-xinerama0 libxcb-cursor0` system-wide.

## Smoke testing the bundle

A minimum sanity check before shipping:

```bash
# Cold-start without arguments — should open the empty editor.
dist/PoseCascade/PoseCascade

# Load a bundled scene + script (paths relative to the bundle root).
dist/PoseCascade/PoseCascade \
  --scene examples/assets/herta/herta.glb \
  --script examples/scripts/showcase.json
```

If the bundle includes `examples/`, both commands should work without
internet access on a freshly imaged machine.

## Common failures

| Symptom                                              | Fix                                                                 |
|------------------------------------------------------|---------------------------------------------------------------------|
| `Could not find or load the Qt platform plugin`      | Missing `--collect-all PySide6`. Re-run with that flag.             |
| `ModuleNotFoundError: importers.<format>`            | Missing `--add-data importers:importers`.                           |
| Toon shading is washed out                           | Shaders not bundled. Add `--add-data shaders:shaders`.              |
| Cloth is slow on the bundle but fast in the dev install | Cython kernel didn't build before bundling; re-run `setup.py build_ext --inplace` then re-bundle. |
| `OSError: cannot identify image file …` on PMX load  | Missing Pillow plugins. Re-run with `--collect-all Pillow`.         |
| Animation JSON fails to validate                     | Missing `--add-data schemas:schemas`.                               |
