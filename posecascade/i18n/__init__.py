"""Lightweight runtime localisation for the PoseCascade GUI.

Public surface (the only names UI code should import):

* :func:`t` — translate a key, with optional ``{name}`` interpolation.
* :func:`set_language` — switch the active locale; persists via :class:`QSettings`.
* :func:`current_language` — currently active locale code (e.g. ``"en"``).
* :func:`available_languages` — auto-discovered list of locales shipped under
  ``posecascade/i18n/locales/*.json`` plus any user drop-ins.
* :func:`initialize` — call once at app startup; resolves the locale from
  (env var → QSettings → ``QLocale.system().name()`` → ``"en"``) and loads the
  catalog. Idempotent.

Catalogs are flat JSON dicts (``{"menu.file.open_scene": "Open Scene…"}``) and
every missing key transparently falls back to English so a partially-translated
locale never shows a raw key to the user. Adding a language is one file:
``posecascade/i18n/locales/<code>.json``.

Design choices flagged in the project plan:

* No ``QTranslator`` / ``.ts`` pipeline — the JSON catalog keeps the build
  toolchain-free and lets contributors add a language without installing the
  Qt linguist tools.
* Pure-Python core (:mod:`posecascade.i18n.catalog`) is importable without Qt
  so the unit tests don't need a ``QApplication``.
* Locale switching writes ``QSettings`` and asks the user to restart — full
  in-place retranslate would need ``changeEvent(QEvent.LanguageChange)``
  plumbing on every widget; that's a future PR.
"""
from __future__ import annotations

import os
from pathlib import Path

from posecascade.i18n.catalog import (
    DEFAULT_LANGUAGE,
    Catalog,
    CatalogError,
    discover_languages,
    load_catalog,
)
from posecascade.utils.logging import get_logger

__all__ = [
    "DEFAULT_LANGUAGE",
    "CatalogError",
    "available_languages",
    "current_language",
    "initialize",
    "language_display_name",
    "set_language",
    "t",
]

_LOCALES_DIR = Path(__file__).resolve().parent / "locales"
_ENV_VAR = "POSECASCADE_LANG"
_QSETTINGS_KEY = "ui/language"
_log = get_logger(__name__)

# Human-readable label for each shipped locale. Keys not listed here fall back
# to the raw code in the UI menu — adding a language only requires the JSON
# catalog; the display name is a one-line addition here (or the code is shown
# as-is if the contributor forgot to update this dict).
_DISPLAY_NAMES: dict[str, str] = {
    "en": "English",
    "zh-TW": "繁體中文",
    "zh-CN": "简体中文",
}

# Module-private active state. We deliberately avoid a class here — the i18n
# layer is process-global by nature (every widget reads the same active locale)
# and a class would just add ``i18n.get_instance().t(...)`` noise at call sites.
_active_catalog: Catalog | None = None
_active_language: str = DEFAULT_LANGUAGE
_fallback_catalog: Catalog | None = None


def initialize(preferred: str | None = None) -> str:
    """Resolve the locale once at startup and load the catalog.

    Resolution order:

    1. ``preferred`` argument, if given (used by tests and CLI overrides).
    2. ``POSECASCADE_LANG`` environment variable.
    3. ``QSettings("PoseCascade", "PoseCascade").value("ui/language")``.
    4. ``QLocale.system().name()`` (e.g. ``"zh_TW"`` → normalised to ``"zh-TW"``).
    5. :data:`DEFAULT_LANGUAGE` (``"en"``).

    Returns the locale code that was actually loaded (may differ from the
    request if the requested code has no catalog shipped).
    """
    requested = preferred or _read_env() or _read_qsettings() or _read_system_locale()
    chosen = _select_supported(requested)
    _load(chosen)
    return chosen


def t(key: str, **kwargs: object) -> str:
    """Translate ``key`` against the active catalog, with ``{name}`` interpolation.

    Falls back to the English catalog when a key is missing; if even English
    lacks the key, returns ``key`` unchanged and logs a warning so missing
    strings show up in the developer log without crashing the UI.
    """
    catalog = _active_catalog if _active_catalog is not None else _ensure_fallback()
    text = catalog.get(key)
    if text is None:
        fallback = _ensure_fallback()
        text = fallback.get(key)
        if text is None:
            _log.warning("missing translation key: %r (no entry in active or en catalog)", key)
            return key
    if not kwargs:
        return text
    try:
        return text.format(**kwargs)
    except (KeyError, IndexError) as err:
        _log.warning("interpolation failed for key %r: %s", key, err)
        return text


def set_language(code: str) -> None:
    """Switch the active locale and persist the choice via :class:`QSettings`.

    Raises :class:`CatalogError` if ``code`` has no catalog on disk so callers
    (e.g. a Settings menu) can surface the failure rather than silently keeping
    the old language.
    """
    chosen = _select_supported(code, strict=True)
    _load(chosen)
    _write_qsettings(chosen)


def current_language() -> str:
    """Return the currently active locale code (e.g. ``"zh-TW"``)."""
    return _active_language


def available_languages() -> list[str]:
    """Return every locale with a catalog on disk, sorted with English first."""
    codes = discover_languages(_LOCALES_DIR)
    # Keep ``en`` at the head of the menu — it's the source-of-truth catalog
    # and the universal fallback, so it deserves the conventional top slot.
    return sorted(codes, key=lambda c: (c != DEFAULT_LANGUAGE, c.lower()))


def language_display_name(code: str) -> str:
    """Human-readable label for a locale code (for the Settings menu)."""
    return _DISPLAY_NAMES.get(code, code)


# ---- internals ----------------------------------------------------------------


def _load(code: str) -> None:
    # The i18n layer is a process-global singleton by design — every widget
    # reads from the same active locale, so a class wrapper would add noise
    # without removing real shared state. PLW0603 is suppressed here for
    # that reason; the suppression on the global statement itself carries
    # the same justification.
    global _active_catalog, _active_language  # noqa: PLW0603
    _active_catalog = load_catalog(_LOCALES_DIR, code)
    _active_language = code


def _ensure_fallback() -> Catalog:
    global _fallback_catalog  # noqa: PLW0603
    if _fallback_catalog is None:
        _fallback_catalog = load_catalog(_LOCALES_DIR, DEFAULT_LANGUAGE)
    return _fallback_catalog


def _select_supported(code: str | None, *, strict: bool = False) -> str:
    """Map a requested code to a code that actually has a catalog on disk.

    With ``strict=True``, raises :class:`CatalogError` if the code is unknown
    (used by :func:`set_language` so a bad menu pick fails loudly). Otherwise
    silently falls back to English so startup never aborts because of a
    missing or misspelled preference.
    """
    if code is None:
        return DEFAULT_LANGUAGE
    normalised = _normalise(code)
    available = set(discover_languages(_LOCALES_DIR))
    if normalised in available:
        return normalised
    # Try the bare language tag (``"en-GB"`` → ``"en"``).
    bare = normalised.split("-", 1)[0]
    if bare in available:
        return bare
    if strict:
        raise CatalogError(
            f"no catalog for language {code!r}; available: {sorted(available)}",
        )
    _log.info("no catalog for %r — falling back to %r", code, DEFAULT_LANGUAGE)
    return DEFAULT_LANGUAGE


def _normalise(code: str) -> str:
    """Convert ``"zh_TW"`` / ``"zh_tw"`` to the canonical ``"zh-TW"`` form."""
    if not code:
        return code
    parts = code.replace("_", "-").split("-", 1)
    if len(parts) == 1:
        return parts[0].lower()
    return f"{parts[0].lower()}-{parts[1].upper()}"


def _read_env() -> str | None:
    raw = os.environ.get(_ENV_VAR)
    return raw.strip() if raw else None


def _read_qsettings() -> str | None:
    """Read the persisted language preference. Returns ``None`` if Qt isn't loaded.

    We intentionally avoid forcing a Qt import at module load time — the pure
    :mod:`catalog` submodule must remain importable in headless tests.
    """
    try:
        from PySide6.QtCore import QSettings  # noqa: PLC0415 — lazy Qt import
    except ImportError:
        return None
    raw = QSettings("PoseCascade", "PoseCascade").value(_QSETTINGS_KEY)
    if raw is None:
        return None
    return str(raw).strip() or None


def _write_qsettings(code: str) -> None:
    try:
        from PySide6.QtCore import QSettings  # noqa: PLC0415 — lazy Qt import
    except ImportError:
        return
    QSettings("PoseCascade", "PoseCascade").setValue(_QSETTINGS_KEY, code)


def _read_system_locale() -> str | None:
    try:
        from PySide6.QtCore import QLocale  # noqa: PLC0415 — lazy Qt import
    except ImportError:
        return None
    name = QLocale.system().name()
    return name or None
