"""Project-file dataclasses + version constant.

The schema is intentionally narrow: it captures the *intent* of an
authoring session — which models, which motions, where the playhead
sits — not derived state. Bone trajectories, morph values, etc. all
re-derive from re-loading the underlying VMDs against the underlying
PMXes; the project file only references them.

Every path is stored as a string relative to the project root; the
load step runs each through
:func:`posecascade.assets.path_safety.resolve_safe`, so a hostile
``../../etc/passwd`` reference never reaches the importer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from posecascade.errors import PoseCascadeError

CURRENT_SCHEMA_VERSION = 1


class ProjectVersionError(PoseCascadeError):
    """The on-disk file declares a schema version we can't load."""


@dataclass(frozen=True)
class ProjectExternalParent:
    """One slot bone follows another slot's bone."""

    self_bone_name: str
    target_slot_name: str
    target_bone_name: str


@dataclass(frozen=True)
class ProjectSlot:
    """One model + (optional) motion + placement.

    Translation / rotation are stored as plain tuples so the schema
    survives JSON encode/decode without a numpy dependency. The
    rotation tuple is ``(x, y, z, w)`` quaternion order — the same
    convention every other module in the engine uses.
    """

    name: str
    model_path: str
    motion_path: str = ""
    visible: bool = True
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    external_parents: tuple[ProjectExternalParent, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProjectAudio:
    """Audio backing for the timeline (single WAV at the project root)."""

    path: str
    offset_seconds: float = 0.0


@dataclass(frozen=True)
class ProjectPlayback:
    """Playback range + tempo. ``fps`` defaults to MMD's 30 Hz."""

    fps: int = 30
    start_frame: int = 0
    end_frame: int = 1000
    loop: bool = True
    current_frame: int = 0


@dataclass(frozen=True)
class ProjectFile:
    """Top-level project bundle — what writer.save_project serialises."""

    version: int = CURRENT_SCHEMA_VERSION
    name: str = ""
    slots: tuple[ProjectSlot, ...] = field(default_factory=tuple)
    audio: ProjectAudio | None = None
    playback: ProjectPlayback = field(default_factory=ProjectPlayback)
    # Effect chain serialised via
    # :func:`posecascade.render.effects.loader.serialize_chain_to_toml`;
    # storing it as TOML keeps the project file self-describing and lets
    # users edit chains by hand when needed.
    effect_chain_toml: str = ""
