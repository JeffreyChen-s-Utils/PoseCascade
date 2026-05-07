"""Tests for the Sketchfab fetch helper.

Network calls are not exercised — only pure helpers (UID extraction, env-var
token loading) and the safety contract (token never appears in the argparse
namespace or in --help output).
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

import fetch_sketchfab  # noqa: E402

_FAKE_UID = "0123456789abcdef0123456789abcdef"
_FAKE_TOKEN = "synthetic-token-for-tests-only"


def test_extract_uid_from_full_url() -> None:
    url = f"https://sketchfab.com/3d-models/example-slug-{_FAKE_UID}"
    assert fetch_sketchfab.extract_uid(url) == _FAKE_UID


def test_extract_uid_lowercases_hex() -> None:
    upper = "FEDCBA9876543210FEDCBA9876543210"
    assert fetch_sketchfab.extract_uid(f"https://example/{upper}") == upper.lower()


def test_extract_uid_accepts_bare_uid() -> None:
    assert fetch_sketchfab.extract_uid(_FAKE_UID) == _FAKE_UID


def test_extract_uid_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        fetch_sketchfab.extract_uid("not-a-sketchfab-url")


def test_extract_uid_rejects_short_hex() -> None:
    with pytest.raises(ValueError):
        fetch_sketchfab.extract_uid("only31charsofhex0000000000000000"[:31])


def test_read_token_from_env_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fetch_sketchfab._TOKEN_ENV_VAR, _FAKE_TOKEN)  # noqa: SLF001
    assert fetch_sketchfab._read_token() == _FAKE_TOKEN  # noqa: SLF001


def test_read_token_falls_back_to_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """When the env var is unset, _read_token reads <project_root>/.sketchfab_token."""
    monkeypatch.delenv(fetch_sketchfab._TOKEN_ENV_VAR, raising=False)  # noqa: SLF001
    monkeypatch.setattr(fetch_sketchfab, "_project_root", lambda: tmp_path)
    (tmp_path / fetch_sketchfab._TOKEN_FILE_NAME).write_text(  # noqa: SLF001
        f"{_FAKE_TOKEN}\n", encoding="utf-8",
    )
    assert fetch_sketchfab._read_token() == _FAKE_TOKEN  # noqa: SLF001


def test_read_token_raises_when_neither_env_nor_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv(fetch_sketchfab._TOKEN_ENV_VAR, raising=False)  # noqa: SLF001
    monkeypatch.setattr(fetch_sketchfab, "_project_root", lambda: tmp_path)
    with pytest.raises(RuntimeError):
        fetch_sketchfab._read_token()  # noqa: SLF001


def test_read_token_prefers_env_over_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """If both env var and file are present, env var wins (faster / fewer FS ops)."""
    other_token = "this-should-not-be-returned"
    monkeypatch.setenv(fetch_sketchfab._TOKEN_ENV_VAR, _FAKE_TOKEN)  # noqa: SLF001
    monkeypatch.setattr(fetch_sketchfab, "_project_root", lambda: tmp_path)
    (tmp_path / fetch_sketchfab._TOKEN_FILE_NAME).write_text(  # noqa: SLF001
        f"{other_token}\n", encoding="utf-8",
    )
    assert fetch_sketchfab._read_token() == _FAKE_TOKEN  # noqa: SLF001


def test_token_does_not_leak_into_help(monkeypatch: pytest.MonkeyPatch) -> None:
    """--help must not print the token even when the env var is set."""
    monkeypatch.setenv(fetch_sketchfab._TOKEN_ENV_VAR, _FAKE_TOKEN)  # noqa: SLF001
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err), pytest.raises(SystemExit):
        fetch_sketchfab.main(["--help"])
    combined = out.getvalue() + err.getvalue()
    assert _FAKE_TOKEN not in combined


def test_main_exits_when_no_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(fetch_sketchfab._TOKEN_ENV_VAR, raising=False)  # noqa: SLF001
    # Redirect the file-fallback root so the real .sketchfab_token in the
    # repo (if present) does not satisfy the lookup.
    monkeypatch.setattr(fetch_sketchfab, "_project_root", lambda: tmp_path)
    err = io.StringIO()
    with redirect_stderr(err):
        rc = fetch_sketchfab.main([f"https://sketchfab.com/3d-models/x-{_FAKE_UID}"])
    assert rc == 1
    assert _FAKE_TOKEN not in err.getvalue()


def test_no_cli_token_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--token must NOT exist (a CLI flag would leak the token to process listings)."""
    monkeypatch.setenv(fetch_sketchfab._TOKEN_ENV_VAR, _FAKE_TOKEN)  # noqa: SLF001
    err = io.StringIO()
    with redirect_stderr(err), pytest.raises(SystemExit) as exc_info:
        fetch_sketchfab.main(["--token", "anything", _FAKE_UID])
    # argparse exits 2 on unknown argument.
    assert exc_info.value.code == 2


def test_help_text_does_not_contain_token_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """argparse default-formatting must not echo the env var contents into --help."""
    monkeypatch.setenv(fetch_sketchfab._TOKEN_ENV_VAR, _FAKE_TOKEN)  # noqa: SLF001
    out = io.StringIO()
    with redirect_stdout(out), pytest.raises(SystemExit):
        fetch_sketchfab.main(["--help"])
    assert _FAKE_TOKEN not in out.getvalue()
    # Sanity: the env var is still set after --help (we read it fresh per-call).
    assert os.environ.get(fetch_sketchfab._TOKEN_ENV_VAR) == _FAKE_TOKEN  # noqa: SLF001
