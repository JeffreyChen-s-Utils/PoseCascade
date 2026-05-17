# PoseCascade GUI translations

Each `*.json` file in this directory is one locale catalog. Adding a language is
one drop-in file — no Python edits, no toolchain.

## File format

A flat JSON object mapping translation **keys** (dot-separated namespaces) to
the **display string** the user sees:

```json
{
  "menu.file.open_scene": "&Open Scene…",
  "dialog.about.title": "About PoseCascade"
}
```

- Keys are stable. Translators only edit values.
- Values may include `{name}` placeholders that the engine substitutes at
  display time (`status.fps` → `"FPS: {value:5.1f}"`). Keep every placeholder
  from the English catalog — drop one and the string falls back to English.
- Ampersand-prefixed letters (`&File`) define keyboard accelerators in menus;
  pick a letter that does not collide with sibling menu entries in your
  language.

## Adding a new language

1. Copy `en.json` to `<code>.json` where `<code>` follows the
   [IETF BCP 47](https://www.rfc-editor.org/info/bcp47) form
   (`ja`, `ko`, `pt-BR`, `de-AT`, `ar`, ...).
2. Translate every value. Missing keys silently fall back to English — useful
   while a translation is in progress, but ship-blocking when complete.
3. (Optional) Add a human-readable label for your locale to `_DISPLAY_NAMES`
   in `posecascade/i18n/__init__.py`. Without this entry the menu shows the
   raw code, which is correct but ugly.
4. Run `py -m pytest tests/i18n/` to confirm the loader accepts your catalog.

## Active locale resolution

At startup the engine picks the active locale in this order:

1. `POSECASCADE_LANG` environment variable.
2. The user's `Settings → Language` choice (persisted via `QSettings`).
3. `QLocale.system().name()` (the OS-reported locale).
4. `"en"` (the source-of-truth catalog).

If a chosen code has no catalog on disk, the loader silently falls back to the
bare language tag (`zh-HK` → `zh`) and then to `en`.

## Why JSON and not Qt's `.ts`/`.qm`

The Qt Linguist toolchain (`pyside6-lupdate` / `pyside6-lrelease`) is the
canonical Qt path but adds a translator-tooling dependency that's overkill for
a small developer-facing project. JSON keeps contributions to a single text
file, plays nicely with `git diff`, and lets users without Qt installed still
add translations.
