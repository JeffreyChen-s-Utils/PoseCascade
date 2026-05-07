"""CLI entry point: ``python -m posecascade --scene <path>``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from posecascade.app.bootstrap import run_app
from posecascade.utils.logging import configure_logging, get_logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="posecascade")
    parser.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="Path to a scene file (.glb / .gltf / .obj / .stl / .ply) to load on startup.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=None,
        help="Path to a user script to attach to the loaded scene.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, configure logging, and launch the Qt application."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    log = get_logger(__name__)
    log.info("starting PoseCascade")
    return run_app(scene_path=args.scene, script_path=args.script)


if __name__ == "__main__":
    sys.exit(main())
