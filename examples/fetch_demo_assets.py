"""Download the CC-licensed sample assets used by ``scene_compose.py``.

The demo uses two Khronos glTF Sample Assets as stand-ins:

- ``CesiumMan`` (CC-BY 4.0, Analytical Graphics, Inc.) — humanoid character
  placeholder. Acheron from Honkai: Star Rail is owned by HoYoverse and is
  not redistributed publicly; the example uses CesiumMan by default and
  accepts ``--character path/to/your.glb`` so you can plug in your own
  legally-obtained model.
- ``Fox`` (CC0 by PixelMannen, walk animation CC-BY 4.0 by tomkranis) —
  quadruped placeholder for the German Shepherd.

The room is generated procedurally inside the engine — no download needed.

Run::

    python examples/fetch_demo_assets.py
"""
from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# Pinned commit on KhronosGroup/glTF-Sample-Assets. Update by checking the
# repo and replacing the SHA below; raw.githubusercontent.com serves the
# exact commit so the download is reproducible.
# Using `main` here as a pragmatic default; production code should pin a SHA.
_REF = "main"
_BASE = (
    f"https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/{_REF}/Models"
)
_TIMEOUT_SEC = 30.0
_MAX_BYTES = 64 * 1024 * 1024  # 64 MB hard cap per file

_ASSETS: tuple[tuple[str, str, str], ...] = (
    (
        "character.glb",
        f"{_BASE}/CesiumMan/glTF-Binary/CesiumMan.glb",
        "CesiumMan — CC-BY 4.0 by Analytical Graphics, Inc.",
    ),
    (
        "dog.glb",
        f"{_BASE}/Fox/glTF-Binary/Fox.glb",
        "Fox — CC0 (model) by PixelMannen, animation CC-BY 4.0 by tomkranis",
    ),
)


def _https_urlopen(url: str):
    """HTTPS-only urlopen guard (mirrors the pattern from CLAUDE.md)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"only https URLs are allowed, got: {url!r}")
    return urllib.request.urlopen(url, timeout=_TIMEOUT_SEC)  # nosec B310  # scheme validated above  # noqa: S310


def fetch(target_dir: Path, *, force: bool = False) -> list[Path]:
    """Download every asset to ``target_dir``. Skips files that already exist
    unless ``force`` is true. Returns the list of resulting paths."""
    target_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for filename, url, attribution in _ASSETS:
        out = target_dir / filename
        if out.exists() and not force:
            print(f"  skip (exists): {out}")
            out_paths.append(out)
            continue
        print(f"  fetching {filename} from {url}")
        with _https_urlopen(url) as resp:
            data = resp.read(_MAX_BYTES + 1)
        if len(data) > _MAX_BYTES:
            raise RuntimeError(f"{filename} exceeded {_MAX_BYTES}-byte cap")
        out.write_bytes(data)
        print(f"    wrote {len(data):,} bytes — {attribution}")
        out_paths.append(out)
    return out_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).parent / "assets",
        help="Where to put the downloaded .glb files (default: examples/assets/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a file already exists at the target path.",
    )
    args = parser.parse_args(argv)
    fetch(args.target, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
