"""Serialise a :class:`ProjectFile` to a JSON byte buffer / file."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from posecascade.project.schema import ProjectFile

_INDENT = 2


def serialize_project(project: ProjectFile) -> str:
    """Return ``project`` as a deterministic, human-readable JSON string.

    Tuples nested inside the dataclass are converted to lists by
    :func:`json.dumps` — the reader accepts both forms when decoding.
    """
    payload = _project_to_dict(project)
    return json.dumps(payload, indent=_INDENT, ensure_ascii=False) + "\n"


def save_project(project: ProjectFile, path: Path) -> None:
    """Write ``project`` to ``path`` (UTF-8 + LF)."""
    path = Path(path)
    path.write_text(serialize_project(project), encoding="utf-8")


def _project_to_dict(project: ProjectFile) -> dict:
    """Convert via :func:`dataclasses.asdict` and normalise tuples → lists.

    ``asdict`` recursively walks frozen dataclasses; the only manual fix
    is that ``audio`` is ``None`` when absent, and ``asdict`` would
    leave it as a ``dict`` for the dataclass instance form. We accept
    that — JSON renders ``None`` as ``null``.
    """
    return asdict(project)
