"""Project file: save / load the whole authoring session.

A ``.posecascade`` file is JSON wrapped in a strict schema. Every model
+ motion + audio reference is stored *relative* to the project root so
the file works after the user re-organises their assets folder. Path
resolution at load time goes through
:func:`posecascade.assets.path_safety.resolve_safe`.

Why JSON instead of the plan's msgpack? Two reasons: stdlib ships a
JSON codec (``json`` + ``msgpack`` would add a dep), and a text-format
project file diffs cleanly in git — which matters for collaborative
authoring more than the binary-format size win does. The schema's
versioned + migrations are wired in, so we can swap codecs later
without breaking existing files.
"""

from posecascade.project.reader import load_project, parse_project
from posecascade.project.schema import (
    CURRENT_SCHEMA_VERSION,
    ProjectAudio,
    ProjectExternalParent,
    ProjectFile,
    ProjectPlayback,
    ProjectSlot,
    ProjectVersionError,
)
from posecascade.project.writer import save_project, serialize_project

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "ProjectAudio",
    "ProjectExternalParent",
    "ProjectFile",
    "ProjectPlayback",
    "ProjectSlot",
    "ProjectVersionError",
    "load_project",
    "parse_project",
    "save_project",
    "serialize_project",
]
