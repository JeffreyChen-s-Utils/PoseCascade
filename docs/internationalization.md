# Internationalisation & responsive UI

PoseCascade ships an in-process internationalisation (i18n) layer plus a
responsive-sizing policy so the desktop editor reads naturally on any locale
and any display from a 1366×768 laptop to a 4K monitor.

This document is the canonical reference for translators, contributors adding
a new language, and engineers extracting new user-facing strings. The system
is **catalog-driven**: every UI string is keyed, every key resolves through a
flat JSON file in `posecascade/i18n/locales/`, and the active locale is
chosen at startup from a fixed resolution order.

> If you only want to **switch languages** as an end user, open
> `Settings → Language` in the running editor and pick one. Continue reading
> if you want to add a language, extract a new string, or understand the
> resolution order.

## Architecture in one minute

```text
┌─────────────────────────────────┐
│ posecascade/i18n/__init__.py    │   public surface
│   t(key, **kwargs)              │   ←─── every widget calls this
│   set_language(code)            │
│   current_language()
│   available_languages()
│   initialize(preferred=None)    │   ←─── bootstrap calls this once
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ posecascade/i18n/catalog.py     │   pure Python, no Qt
│   load_catalog(dir, lang)       │
│   discover_languages(dir)       │
│   class Catalog                 │
└────────────────┬────────────────┘
                 │ reads
                 ▼
┌─────────────────────────────────┐
│ posecascade/i18n/locales/       │   on-disk catalogs
│   en.json       ←─ source of truth
│   zh-TW.json    ←─ ships with the editor
│   zh-CN.json    ←─ ships with the editor
│   <code>.json   ←─ drop-in for any new locale
│   README.md     ←─ contributor entry point
└─────────────────────────────────┘
```

The split between `__init__.py` and `catalog.py` is deliberate:

* **`catalog.py` is Qt-free** — its tests run without a `QApplication`, so the
  CI matrix can validate every shipped catalog on slots that don't even have
  PySide6 installed.
* **`__init__.py` owns the runtime state** — the active locale, the fallback
  cache, and the `QSettings` bridge. It lazy-imports Qt so the catalog
  primitives stay headless.

## Public API

Import the names you need from `posecascade.i18n`:

```python
from posecascade.i18n import (
    t,                      # translate a key
    set_language,           # switch the active locale + persist via QSettings
    current_language,       # active locale code, e.g. "zh-TW"
    available_languages,    # list of locale codes shipped on disk
    language_display_name,  # human-readable label for the Settings menu
    initialize,             # called once by bootstrap.run_app
)
```

### `t(key, **kwargs) -> str`

Look up `key` in the active catalog. If the catalog has no entry for `key`,
fall back to the English catalog. If even English has no entry, log a
WARNING and return `key` unchanged so the UI keeps running.

```python
self.setWindowTitle(t("app.window_title"))
self._fps_label.setText(t("status.fps", value=1.0 / avg_dt))
QMessageBox.warning(
    self,
    t("dialog.empty_scene.title"),
    t("dialog.empty_scene.body", name=script_path.name),
)
```

Interpolation uses `str.format`, so the catalog value can declare named
placeholders (`{name}`, `{value:5.1f}`) and the call site passes matching
keyword arguments. A failed interpolation (typo, missing argument) logs a
WARNING and returns the un-interpolated text rather than raising.

### `set_language(code) -> None`

Switch the active locale and persist the choice via
`QSettings("PoseCascade", "PoseCascade")`. Unlike `initialize`, this **raises
`CatalogError`** when `code` has no catalog on disk, so a Settings-menu pick
fails loudly rather than silently keeping the old language. The Settings →
Language menu in `MainWindow` shows the error in a `QMessageBox` and asks
the user to restart for the change to take effect.

> Full live retranslation (without restart) requires
> `changeEvent(QEvent.LanguageChange)` plumbing on every widget. That's a
> future PR — the first cut intentionally keeps the surface area finite.

### `current_language() -> str`

The locale code that's currently active (`"en"`, `"zh-TW"`, …). Useful for
showing a checkmark on the active entry in the language menu.

### `available_languages() -> list[str]`

Scan `posecascade/i18n/locales/` for `*.json` files and return their stems
sorted with English first, then the rest alphabetically. Adding a language
is **one file drop-in** — `available_languages()` discovers it immediately,
no code change needed.

### `language_display_name(code) -> str`

Return the human-readable label shown in the Settings → Language menu. Codes
without an entry in the `_DISPLAY_NAMES` dict in `posecascade/i18n/__init__.py`
show the raw code, which is correct but ugly — when you add a translation,
register its display name in that dict too (one line).

### `initialize(preferred=None) -> str`

Resolve the active locale once at app startup and load the catalog. Called
by `posecascade.app.bootstrap.run_app` before the main window is built, so
every widget's constructor sees the active locale on the first `t()` call.

Resolution order:

| Priority | Source                                                  |
|----------|---------------------------------------------------------|
| 1        | The `preferred` argument (tests + CLI overrides)        |
| 2        | The `POSECASCADE_LANG` environment variable             |
| 3        | `QSettings("PoseCascade", "PoseCascade")["ui/language"]` |
| 4        | `QLocale.system().name()` (e.g. `"zh_TW"` → normalised to `"zh-TW"`) |
| 5        | `DEFAULT_LANGUAGE` (`"en"`)                             |

Unknown codes fall back silently — first to the bare language tag (`"en-GB"`
→ `"en"`), then to English. This keeps startup robust against a stale
`QSettings` value referencing a locale that's since been removed.

## Catalog format

Each `*.json` in `posecascade/i18n/locales/` is a **flat object** mapping
dot-separated keys to display strings:

```json
{
  "menu.file": "&File",
  "menu.file.open_scene": "&Open Scene…",
  "status.fps": "FPS: {value:5.1f}",
  "dialog.empty_scene.body": "Script {name!r} is being loaded into an empty scene.\n\n…"
}
```

### Rules enforced by `load_catalog`

The loader rejects malformed catalogs at startup so a typo never silently
becomes a runtime bug:

* The top-level must be a JSON object (`{}`). Arrays or scalars are rejected.
* Every value must be a string. Numbers, booleans, nested objects, or `null`
  fail loading with a `CatalogError`.
* Every key must be a string.
* Missing files raise `CatalogError("catalog not found: …")`.
* Malformed JSON raises `CatalogError("catalog … is not valid JSON: …")` with
  the underlying `json.JSONDecodeError` message preserved.

### Key naming convention

Keys are flat strings; the dots are **naming convention**, not syntax — the
loader doesn't split or namespace on them. The convention helps humans:

```
<area>.<sub_area>.<element>[.<state>]

menu.file.open_scene           ← File menu, "Open Scene"
status.scene_prefix            ← Status bar "scene:" prefix
dialog.empty_scene.title       ← Empty-scene warning dialog title
dialog.empty_scene.body        ← Same dialog, body text
inspector.spring.tooltip.stiffness   ← Spring chain editor's "Stiffness" tooltip
```

When adding a string, follow the prefixes already in `en.json`. A new namespace
is fine for a genuinely new area; reuse an existing one when the string fits.

### Placeholder support

Values may contain `{name}` placeholders that the engine substitutes at
display time via `str.format`. Every translation **must keep the same set of
placeholders** as the English source — drop one and the fallback path kicks
in for the whole string, which means the locale silently reverts to English
for that call site.

| English value                                 | Required placeholders |
|-----------------------------------------------|-----------------------|
| `"FPS: {value:5.1f}"`                         | `value`               |
| `"scene: {name}"`                             | `name`                |
| `"Failed to open project:\n{error}"`          | `error`               |
| `"Script {name!r} is being loaded into an empty scene…"` | `name`     |

The format-spec syntax (`:5.1f`, `!r`) is preserved when translating — Python
runs the format on the **translated** value, so `"幀率：{value:5.1f}"` still
renders `值: 60.0` correctly.

### Menu accelerators

Ampersand-prefixed letters in menu strings (`&File` → underlined `F`) drive
Qt's keyboard accelerators. When translating, pick a letter that doesn't
collide with sibling menu entries in the target language. Chinese locales
typically suffix `(&F)` instead of inlining `&F` because the menu label
itself doesn't contain a Latin letter to underline:

```
"menu.file":                "&File"             ← en
"menu.file":                "檔案(&F)"          ← zh-TW
"menu.file":                "文件(&F)"          ← zh-CN
```

The order of the accelerator (first character vs. trailing) is by language
convention; Qt accepts either form.

## Adding a new language

Every step:

1. **Copy the English catalog.**
   ```bash
   cp posecascade/i18n/locales/en.json posecascade/i18n/locales/<code>.json
   ```
   Use an [IETF BCP 47](https://www.rfc-editor.org/info/bcp47) code:
   `ja`, `ko`, `pt-BR`, `de-AT`, `ar`, `ru`, … . The loader normalises
   `zh_TW` to `zh-TW` (hyphen, region uppercased) so an OS-reported underscore
   form maps to your file automatically.

2. **Translate every value.** Keep every `{placeholder}` from the English
   source; keep the ampersand accelerator format (or adapt it to your
   language's convention).

3. **Add a display name (recommended).** Open `posecascade/i18n/__init__.py`
   and add an entry to `_DISPLAY_NAMES`:
   ```python
   _DISPLAY_NAMES: dict[str, str] = {
       "en": "English",
       "zh-TW": "繁體中文",
       "zh-CN": "简体中文",
       "ja":    "日本語",      # ← your addition
   }
   ```
   Without this entry the Settings → Language menu shows the raw code (e.g.
   `"ja"`), which works but is ugly.

4. **Run the test suite.**
   ```bash
   py -m pytest tests/test_i18n_catalog.py tests/test_i18n_runtime.py
   ```
   `test_shipped_catalogs_have_consistent_keys` walks every shipped catalog
   and asserts the key set matches English exactly — a missing key is
   reported as `<code> missing keys: [...]`, an extra key as `<code> extra
   keys not in en: [...]`. Both are ship-blocking. Fix and re-run.

5. **Manual smoke check.**
   ```bash
   POSECASCADE_LANG=<code> py -m posecascade
   ```
   Verify menus, dock titles, dialogs, and the status bar render in your
   language; verify accelerators work; verify the Settings → Language menu
   lists your locale.

6. **Open a PR.** The CI runs the same `pytest` gate plus `ruff` and
   `bandit`. Bundle the JSON + the one-line `_DISPLAY_NAMES` addition in
   the same commit.

## Extracting a new string

When adding a UI surface that needs a translatable string:

1. **Pick a key** following the naming convention. Reuse an existing prefix
   when possible (`menu.*`, `dialog.*`, `inspector.*`, `export.*`).
2. **Add it to `en.json` first.** English is the source of truth; every
   other catalog must carry the same key (see consistency test above).
3. **Add the same key to every other shipped catalog** with the translated
   value. The consistency test fails the build if you forget.
4. **Replace the literal in the widget**:
   ```python
   # before
   self.setWindowTitle("New Settings")

   # after
   from posecascade.i18n import t
   self.setWindowTitle(t("settings.title"))
   ```
5. **Test it.** `py -m pytest` must stay green; the file-level Qt smoke
   tests usually catch missing imports for free.

> The locale catalogs were extracted from ~90 UI literals in one commit
> spanning the main window, bootstrap, and 12 dock / dialog widgets. See
> `git log --grep="i18n + responsive sizing"` for the original extraction
> pattern — match its style when adding new strings.

## Responsive UI sizing (companion policy)

The same commit that introduced the i18n catalog also retired every
hardcoded pixel literal from the editor's user-visible widgets. This matters
because Chinese, Japanese, and Korean glyphs are typically taller than
Latin glyphs at the same point size, and on a 200 % HiDPI display the old
640×480 minimum viewport read as a postage stamp. Both problems collapse to
the same fix: **size widgets in font-metric units, not pixels.**

### What changed in bootstrap

`posecascade/app/bootstrap.py:_configure_high_dpi()` enables Qt6's
`HighDpiScaleFactorRoundingPolicy.PassThrough` policy before constructing
the `QApplication`:

```python
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication

QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough,
)
```

The default `Round` policy snaps to 100 % → 200 % → 300 % steps, which
produces a visibly chunky layout on a 150 % laptop scale. `PassThrough`
tracks the OS scale exactly so font-metric sizing maps cleanly to the user's
display.

### What changed in widgets

Every pixel literal in user-visible code became a font-metric multiplier:

| Old                                          | New                                                            |
|----------------------------------------------|----------------------------------------------------------------|
| `window.resize(1280, 800)`                   | `screen.availableGeometry() × 0.75`                            |
| `viewport.setMinimumSize(640, 480)`          | `fm.height() × _VIEWPORT_MIN_ROWS_W` (= 30)                    |
|                                              | `fm.height() × _VIEWPORT_MIN_ROWS_H` (= 22)                    |
| `components_list.setMaximumHeight(120)`      | `fm.height() × _COMPONENTS_LIST_MAX_ROWS` (= 5)                |
| `fps_label.setMinimumWidth(80)`              | `digit_w × _FPS_LABEL_MIN_WIDTH_CHARS` (= 8)                   |

`fm` is `widget.fontMetrics()`; `digit_w` is `fm.horizontalAdvance("0")`.
The constants live at module top — `_VIEWPORT_MIN_ROWS_W = 30` reads as
"viewport is at least 30 text rows wide", which is the unit the user
actually perceives.

### Startup window size

```python
_STARTUP_WINDOW_FRACTION = 0.75
screen = QGuiApplication.primaryScreen()
available = screen.availableGeometry()
width = int(available.width() * _STARTUP_WINDOW_FRACTION)
height = int(available.height() * _STARTUP_WINDOW_FRACTION)
window.resize(width, height)
```

The window opens at 75 % of the primary screen's available geometry (which
already excludes the taskbar / menu bar), centered by Qt's default placement.
This gives a sensible window on a 1366×768 laptop (1024×576) and on a 4K
panel (2880×1620) without further tuning. A `_FALLBACK_WINDOW_SIZE = (1280,
800)` constant covers the headless / very-early-bootstrap path where no
screen is reported.

### Adding new widgets

When introducing a new widget, **don't** hardcode pixels. Either:

* Let the layout choose — `QHBoxLayout` / `QVBoxLayout` size children
  according to their `sizeHint()`. This is usually right.
* Derive a minimum from `fontMetrics()` when the widget needs a floor:
  ```python
  fm = self.fontMetrics()
  self.setMinimumWidth(fm.horizontalAdvance("0") * 12)   # 12 digits wide
  self.setMinimumHeight(fm.height() * 4)                 # 4 text rows tall
  ```
* Use `setSizePolicy()` to grow / shrink with the parent.

## Why JSON and not Qt's `.ts` / `.qm` toolchain?

Qt Linguist (`pyside6-lupdate` / `pyside6-lrelease`) is the canonical Qt
path but adds a translator-tooling dependency that's overkill for a small
developer-facing project. JSON keeps contributions to a single text file,
plays nicely with `git diff`, and lets users without Qt installed still add
translations.

The trade-offs we deliberately accept:

* **No `tr()` / no `QTranslator`** — `t("key")` is a function call, not a
  string-literal annotation, so Qt's plural-aware machinery and Linguist's
  context-aware UI aren't available. That's fine: the editor doesn't have
  enough plural-sensitive surfaces to justify the toolchain.
* **No automatic extraction** — `lupdate` would crawl source for `tr()`
  calls and produce a `.ts` template. We hand-curate `en.json` and ask the
  consistency test to enforce parity across locales instead.
* **No translator-facing tooling beyond a text editor** — Linguist's
  fuzzy-match UI is genuinely nice for big projects with paid translators;
  hand-translated JSON is fine for a project with three locales.

If the project ever grows enough to justify Linguist, the catalog can be
ported with a one-pass script (each key becomes a `.ts` context-string,
each value becomes a translation). The decision is reversible.

## Tests

The i18n module ships with 31 unit tests across two files:

| File                                  | Covers                                                       |
|---------------------------------------|--------------------------------------------------------------|
| `tests/test_i18n_catalog.py`          | Pure-Python catalog loader: JSON parsing, schema rejection, discovery. No Qt. |
| `tests/test_i18n_runtime.py`          | Runtime surface: `t()`, fallback, interpolation, locale switching, shipped-catalog parity. |

Notable tests:

* **`test_shipped_catalogs_have_consistent_keys`** — every shipped locale
  must define exactly the same keys as English. Drift here is the most
  common i18n bug; we treat both missing and extra keys as ship-blocking.
* **`test_t_falls_back_to_english_when_key_missing_in_locale`** — proves
  the fallback path works on a partial locale, so a translation-in-progress
  PR can still merge and ship.
* **`test_t_returns_key_when_missing_everywhere`** — confirms the
  "render the key + log a warning" behaviour rather than raising, so
  a typo in a `t()` call never crashes the editor.
* **`test_t_falls_back_on_bad_interpolation`** — same idea for `str.format`
  failures: render the template, log a warning, keep going.

Run them with:

```bash
py -m pytest tests/test_i18n_catalog.py tests/test_i18n_runtime.py -v
```

## Troubleshooting

### "My language picker shows raw codes (`ja`, `ko`) instead of names"

You shipped the locale JSON but forgot to add it to `_DISPLAY_NAMES` in
`posecascade/i18n/__init__.py`. The menu still works — clicking the raw
code switches the language correctly — but the label is ugly. One-line fix.

### "I added a new key but the locales failed to load"

`load_catalog` rejects non-string values, nested objects, and non-object
roots. Check the JSON: every value must be a plain string. If you need
structured data, flatten the key (`form.foo.bar` instead of
`{"form": {"foo": {"bar": "..."}}}`).

### "My new string shows the raw key in the UI"

Three possibilities, in order of likelihood:

1. The key isn't in `en.json` — the fallback path can't find it either.
2. The key is in `en.json` but you typo'd it in the `t()` call (silent
   warning in the logger, but no crash).
3. The catalog failed to load entirely (check stderr for `CatalogError`).

### "Language switch doesn't take effect until restart"

By design — full in-place retranslate requires per-widget
`changeEvent(QEvent.LanguageChange)` handling, which is a future PR. The
restart dialog tells the user this explicitly.

### "I want to override the OS-reported locale during development"

```bash
# Force a specific catalog regardless of QSettings / OS locale
POSECASCADE_LANG=zh-TW py -m posecascade

# Force English even though zh-TW is shipped
POSECASCADE_LANG=en py -m posecascade
```

`POSECASCADE_LANG` is the highest-priority source in the resolution order;
it overrides both `QSettings` and `QLocale.system()`.
