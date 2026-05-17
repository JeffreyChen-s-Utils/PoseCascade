"""Runtime tests for the :mod:`posecascade.i18n` public surface.

Covers translation lookup, English fallback, interpolation, locale switching,
and the shipped catalogs themselves (the en / zh-TW / zh-CN trio must all
parse and have matching key sets so a translated UI never shows a raw key).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from posecascade.i18n import (
    DEFAULT_LANGUAGE,
    CatalogError,
    available_languages,
    current_language,
    initialize,
    language_display_name,
    set_language,
    t,
)
from posecascade.i18n import catalog as catalog_module


@pytest.fixture(autouse=True)
def _reset_i18n_state() -> None:
    """Force every test to start from a clean slate.

    The i18n module caches the active and fallback catalogs at module scope
    (that's the whole point — every widget reads from the same active locale).
    Without a reset, test order changes results. We reach into the module's
    privates here on purpose; alternative would be a public ``_reset_for_tests``
    that adds API surface the rest of the codebase doesn't need.
    """
    from posecascade import i18n  # noqa: PLC0415

    i18n._active_catalog = None  # noqa: SLF001
    i18n._fallback_catalog = None  # noqa: SLF001
    i18n._active_language = DEFAULT_LANGUAGE  # noqa: SLF001
    yield


def test_initialize_loads_default_language() -> None:
    chosen = initialize("en")
    assert chosen == "en"
    assert current_language() == "en"


def test_initialize_loads_zh_tw() -> None:
    chosen = initialize("zh-TW")
    assert chosen == "zh-TW"
    assert current_language() == "zh-TW"


def test_initialize_normalises_underscore_form() -> None:
    """QLocale-style ``zh_TW`` should map to our shipped ``zh-TW`` catalog."""
    chosen = initialize("zh_TW")
    assert chosen == "zh-TW"


def test_initialize_falls_back_for_unknown_locale() -> None:
    """A locale with no catalog should fall back silently to English, not raise."""
    chosen = initialize("xx-XX")
    assert chosen == "en"


def test_initialize_falls_back_to_bare_language_tag() -> None:
    """``en-GB`` should resolve to ``en`` when ``en-GB`` itself isn't shipped."""
    chosen = initialize("en-GB")
    assert chosen == "en"


def test_t_returns_english_value() -> None:
    initialize("en")
    # ``app.window_title`` is always English (the brand name) in every catalog.
    assert t("app.window_title") == "PoseCascade"


def test_t_returns_translated_value() -> None:
    initialize("zh-TW")
    # ``status.ready`` is one of the catalog's most basic entries.
    assert t("status.ready") == "就緒"


def test_t_falls_back_to_english_when_key_missing_in_locale(tmp_path: Path) -> None:
    """A locale that doesn't translate every key transparently falls back to en.

    We build a stub locales dir with a partial ``fr.json`` so we can prove the
    fallback path without depending on the shipped catalogs being incomplete.
    """
    (tmp_path / "en.json").write_text(
        json.dumps({"a": "alpha", "b": "beta"}), encoding="utf-8",
    )
    (tmp_path / "fr.json").write_text(
        json.dumps({"a": "alpha-fr"}), encoding="utf-8",
    )
    with patch.object(catalog_module, "DEFAULT_LANGUAGE", "en"), \
         patch("posecascade.i18n._LOCALES_DIR", tmp_path):
        initialize("fr")
        assert t("a") == "alpha-fr"      # translated
        assert t("b") == "beta"          # English fallback


def test_t_returns_key_when_missing_everywhere(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Truly missing keys round-trip the raw key + log a warning.

    Returning the key (rather than raising) keeps the UI usable when a
    translation is in progress; the warning surfaces the omission in dev logs.
    """
    (tmp_path / "en.json").write_text(json.dumps({"a": "alpha"}), encoding="utf-8")
    with patch("posecascade.i18n._LOCALES_DIR", tmp_path):
        initialize("en")
        with caplog.at_level("WARNING"):
            assert t("does.not.exist") == "does.not.exist"
        assert any("missing translation key" in r.message for r in caplog.records)


def test_t_interpolates_kwargs() -> None:
    initialize("en")
    # ``status.scene_prefix`` is one of the formatted-string entries.
    assert t("status.scene_prefix", name="cube.glb") == "scene: cube.glb"


def test_t_falls_back_on_bad_interpolation(caplog: pytest.LogCaptureFixture) -> None:
    """Malformed interpolation should log + return the un-interpolated text.

    Better to render the template as-is than to crash a window-paint because
    a contributor introduced a typo in the catalog.
    """
    initialize("en")
    with caplog.at_level("WARNING"):
        # ``status.fps`` expects ``value``; passing the wrong key triggers
        # the KeyError branch in ``str.format``.
        result = t("status.fps", wrong="x")
    assert "FPS" in result   # base text returned
    assert any("interpolation failed" in r.message for r in caplog.records)


def test_set_language_switches_active_catalog() -> None:
    initialize("en")
    set_language("zh-CN")
    assert current_language() == "zh-CN"
    assert t("status.ready") == "就绪"


def test_set_language_rejects_unknown_locale() -> None:
    """Unlike ``initialize``, ``set_language`` raises on unknown codes — a
    Settings menu pick should fail loudly so the UI can show the user what
    went wrong, not silently keep the previous language."""
    initialize("en")
    with pytest.raises(CatalogError):
        set_language("xx-XX")


def test_available_languages_puts_english_first() -> None:
    """The Settings menu order surfaces English at the top by convention."""
    langs = available_languages()
    assert "en" in langs
    assert langs[0] == "en"


def test_available_languages_includes_shipped_locales() -> None:
    """en / zh-TW / zh-CN are the curated baseline — guard against an
    accidental rename or deletion under ``posecascade/i18n/locales/``."""
    langs = set(available_languages())
    assert {"en", "zh-TW", "zh-CN"}.issubset(langs)


def test_language_display_name_known_codes() -> None:
    assert language_display_name("en") == "English"
    assert language_display_name("zh-TW") == "繁體中文"
    assert language_display_name("zh-CN") == "简体中文"


def test_language_display_name_falls_back_to_code() -> None:
    """An unmapped code (e.g. a community-contributed locale) should show the
    raw code rather than ``"<unknown>"`` so users can still tell what they
    picked. The catalog ships before the display-name entry needs to."""
    assert language_display_name("xx-XX") == "xx-XX"


def test_shipped_catalogs_have_consistent_keys() -> None:
    """Every shipped locale must define the same keys as English.

    Drift here is the most common i18n bug: an extra key in a translation
    that doesn't exist in en is dead code; a missing key in a translation
    silently falls back, making the locale look partially translated. We
    treat both as ship-blocking.
    """
    locales_dir = Path(__file__).resolve().parent.parent / "posecascade" / "i18n" / "locales"
    en_keys = set(json.loads((locales_dir / "en.json").read_text(encoding="utf-8")).keys())
    for code in ("zh-TW", "zh-CN"):
        path = locales_dir / f"{code}.json"
        keys = set(json.loads(path.read_text(encoding="utf-8")).keys())
        missing = en_keys - keys
        extra = keys - en_keys
        assert not missing, f"{code} missing keys: {sorted(missing)}"
        assert not extra, f"{code} extra keys not in en: {sorted(extra)}"
