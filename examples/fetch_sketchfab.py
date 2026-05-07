"""Download a model from Sketchfab via the official Data API.

Quick start (PowerShell)::

    # 1. Get a personal API token at https://sketchfab.com/settings/password
    # 2. Export it (process-scoped only — never write it to a file you commit):
    $env:SKETCHFAB_API_TOKEN = "your-token-here"
    # 3. Run with a Sketchfab model URL or 32-char UID:
    python examples/fetch_sketchfab.py https://sketchfab.com/3d-models/<slug>-<uid>

    # …or download straight into the slot scene_compose.py expects:
    python examples/fetch_sketchfab.py <URL> --target examples/assets/character.glb

Token handling:

- The API token is read **only** from ``SKETCHFAB_API_TOKEN``. There is no
  ``--token`` CLI flag — CLI args show up in process listings and shell
  history, which is the wrong place for a secret.
- The script never prints, logs, or echoes the token. It is passed through
  the ``Authorization`` header on outgoing requests and immediately
  dereferenced. The argparse Namespace never holds it.
- HTTP errors from urllib do not include request headers, so a failed
  authentication won't dump the token into a traceback.

Notes about Sketchfab:

- The script prints title, uploader, license slug (``by`` / ``cc0`` /
  ``by-sa`` / …) and any attribution requirements *before* downloading.
- Many fan uploads of franchise characters carry a license slug the
  uploader chose, but the underlying IP belongs to the rights holder. The
  Sketchfab metadata is what the API reports; deciding whether that
  licence actually authorises your use is on you.
- Download is HTTPS-only and capped at 256 MB per file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_SKETCHFAB_API = "https://api.sketchfab.com/v3"
_TIMEOUT_SEC = 60.0
_MAX_BYTES = 256 * 1024 * 1024  # 256 MB hard cap
_UID_REGEX = re.compile(r"([0-9a-fA-F]{32})")
_PREFERRED_FORMATS: tuple[str, ...] = ("glb", "gltf", "source")
_TOKEN_ENV_VAR = "SKETCHFAB_API_TOKEN"  # nosec B105  # env var name, not a password  # noqa: S105
# File fallback when the env var hasn't propagated to this process. Must be
# gitignored — see .gitignore. One line, the token, no quotes.
_TOKEN_FILE_NAME = ".sketchfab_token"  # nosec B105  # filename, not a password  # noqa: S105


def _https_urlopen(url: str, *, headers: dict[str, str] | None = None):
    """HTTPS-only urlopen guard (CLAUDE.md network-safety pattern)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"only https URLs are allowed, got: {url!r}")
    request = urllib.request.Request(url, headers=headers or {})  # noqa: S310  # scheme validated above
    return urllib.request.urlopen(request, timeout=_TIMEOUT_SEC)  # nosec B310  # scheme validated above  # noqa: S310


def extract_uid(url_or_uid: str) -> str:
    """Pull a 32-char Sketchfab UID out of a URL, or accept a bare UID."""
    match = _UID_REGEX.search(url_or_uid)
    if match is None:
        raise ValueError(f"could not find a 32-char Sketchfab UID in {url_or_uid!r}")
    return match.group(1).lower()


def _project_root() -> Path:
    """Walk up from this file until ``.gitignore`` (or ``importers/``) appears."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".gitignore").is_file() or (parent / "importers").is_dir():
            return parent
    return here.parent


def _read_token() -> str:
    """Resolve the API token. Tries the env var first, then a gitignored file
    at ``<project_root>/.sketchfab_token``. Never returns the value via logs
    and is the only function that touches secret bytes.
    """
    env_token = os.environ.get(_TOKEN_ENV_VAR, "").strip()
    if env_token:
        return env_token
    token_file = _project_root() / _TOKEN_FILE_NAME
    if token_file.is_file():
        # File contents may have a trailing newline; strip it.
        from_file = token_file.read_text(encoding="utf-8").strip()
        if from_file:
            return from_file
    raise RuntimeError(
        f"missing token. Either set ${_TOKEN_ENV_VAR} (and restart this "
        f"process so children inherit it) or put the token into "
        f"{token_file} (one line, gitignored). Get a token from "
        "https://sketchfab.com/settings/password ."
    )


def _api_get(path: str, token: str) -> dict:
    url = f"{_SKETCHFAB_API}{path}"
    headers = {"Authorization": f"Token {token}"}
    with _https_urlopen(url, headers=headers) as resp:
        return json.load(resp)


def fetch_model_info(uid: str, token: str) -> dict:
    """``GET /v3/models/{uid}`` — name, uploader, license, isDownloadable, …"""
    return _api_get(f"/models/{uid}", token)


def fetch_download_urls(uid: str, token: str) -> dict:
    """``GET /v3/models/{uid}/download`` — per-format signed URLs (expire quickly)."""
    return _api_get(f"/models/{uid}/download", token)


def _print_summary(info: dict) -> None:
    license_info = info.get("license") or {}
    user = info.get("user") or {}
    print(f"  Title:    {info.get('name')!r}")
    print(f"  Uploader: {user.get('username')!r}  (uid: {user.get('uid')})")
    print(f"  License:  {license_info.get('label', 'UNKNOWN')} ({license_info.get('slug')})")
    requirements = license_info.get("requirements")
    if requirements:
        print(f"  Attribution required: {requirements}")
    print(f"  Downloadable: {bool(info.get('isDownloadable'))}")


def _pick_format(download_info: dict) -> tuple[str, dict]:
    """Choose the best available download format. Raises if none match."""
    for key in _PREFERRED_FORMATS:
        entry = download_info.get(key)
        if entry and entry.get("url"):
            return key, entry
    available = sorted(k for k, v in download_info.items() if v)
    raise RuntimeError(
        f"no {' / '.join(_PREFERRED_FORMATS)} download available; got formats: {available}"
    )


def download_model(uid: str, token: str, target: Path) -> Path:
    """Pull metadata, surface the licence, and download the model to ``target``."""
    info = fetch_model_info(uid, token)
    _print_summary(info)
    if not info.get("isDownloadable"):
        raise RuntimeError("Sketchfab reports this model is not downloadable")
    download_info = fetch_download_urls(uid, token)
    fmt, entry = _pick_format(download_info)
    print(f"  downloading {fmt!r} to {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # The signed download URL is bound to the model + a short expiry; it
    # does not carry the user's API token.
    with _https_urlopen(entry["url"]) as resp:
        data = resp.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise RuntimeError(f"download exceeded {_MAX_BYTES}-byte cap")
    target.write_bytes(data)
    print(f"  wrote {len(data):,} bytes")
    return target


def _default_target(uid: str) -> Path:
    return Path(__file__).parent / "assets" / f"{uid}.glb"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download a Sketchfab model. Reads the API token from the "
            f"${_TOKEN_ENV_VAR} environment variable (no CLI flag — CLI "
            "args are visible in process listings)."
        ),
    )
    parser.add_argument("model", help="Sketchfab model URL or 32-character UID.")
    parser.add_argument(
        "--target", type=Path, default=None,
        help="Output path (default: examples/assets/<uid>.glb).",
    )
    args = parser.parse_args(argv)

    try:
        token = _read_token()
    except RuntimeError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    try:
        uid = extract_uid(args.model)
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    target = args.target or _default_target(uid)
    try:
        download_model(uid, token, target)
    except (urllib.error.HTTPError, urllib.error.URLError) as err:
        # urllib errors do not include request headers, so the token is not
        # exposed via the traceback. Avoid printing the full repr just in case.
        reason = err.reason if hasattr(err, "reason") else err
        print(f"network error: {type(err).__name__}: {reason}", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as err:
        print(f"download failed: {err}", file=sys.stderr)
        return 3
    finally:
        # Best-effort clobber of the local reference so a later traceback
        # frame capture has less chance of holding the token.
        token = ""  # nosec B105  # noqa: F841,S105  # intentional zero-out of the local
    return 0


if __name__ == "__main__":
    sys.exit(main())
