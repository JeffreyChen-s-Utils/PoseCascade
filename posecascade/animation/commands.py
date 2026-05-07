"""Command-pattern undo / redo stack for :class:`AnimationDocument`.

Each :class:`Command` owns a tiny piece of edit state (the keyframe to
insert, the key to delete) and exposes ``execute`` / ``undo`` mutators
on the document. :class:`CommandStack` runs them in order and lets the
UI rewind / replay; pushing a new command after an undo clears the
redo stack — matches the standard editor behaviour where editing
mid-undo abandons the redo branch.

Command classes are deliberately thin: anything more elaborate (e.g.
multi-keyframe drag-edits) will compose multiple commands inside a
single :class:`CompoundCommand` rather than growing a one-off subclass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from vmd.types import (
    VmdBoneKeyframe,
    VmdCameraKeyframe,
    VmdLightKeyframe,
    VmdMorphKeyframe,
    VmdSelfShadowKeyframe,
)

from posecascade.animation.document import AnimationDocument


class Command(ABC):
    """Abstract command — every concrete edit is one of these."""

    @abstractmethod
    def execute(self) -> None:
        """Apply the edit to the document."""

    @abstractmethod
    def undo(self) -> None:
        """Reverse the most-recently-applied execution."""


@dataclass
class InsertBoneKeyframe(Command):
    """Insert / overwrite a bone keyframe.

    Captures the original keyframe at the same ``(bone_name, frame)`` (if
    any) on first execute so undo restores the prior state instead of
    just deleting blindly.
    """

    document: AnimationDocument
    keyframe: VmdBoneKeyframe
    _previous: VmdBoneKeyframe | None = field(default=None, init=False)
    _has_previous: bool = field(default=False, init=False)

    def execute(self) -> None:
        if not self._has_previous:
            self._previous = self.document.find_bone_keyframe(
                self.keyframe.bone_name, self.keyframe.frame,
            )
            self._has_previous = True
        self.document.insert_bone_keyframe(self.keyframe)

    def undo(self) -> None:
        self.document.delete_bone_keyframe(
            self.keyframe.bone_name, self.keyframe.frame,
        )
        if self._previous is not None:
            self.document.insert_bone_keyframe(self._previous)


@dataclass
class DeleteBoneKeyframe(Command):
    document: AnimationDocument
    bone_name: str
    frame: int
    _removed: VmdBoneKeyframe | None = field(default=None, init=False)

    def execute(self) -> None:
        self._removed = self.document.delete_bone_keyframe(self.bone_name, self.frame)

    def undo(self) -> None:
        if self._removed is not None:
            self.document.insert_bone_keyframe(self._removed)


@dataclass
class InsertMorphKeyframe(Command):
    document: AnimationDocument
    keyframe: VmdMorphKeyframe
    _previous: VmdMorphKeyframe | None = field(default=None, init=False)
    _has_previous: bool = field(default=False, init=False)

    def execute(self) -> None:
        if not self._has_previous:
            for kf in self.document.morph_keyframes:
                if (
                    kf.morph_name == self.keyframe.morph_name
                    and kf.frame == self.keyframe.frame
                ):
                    self._previous = kf
                    break
            self._has_previous = True
        self.document.insert_morph_keyframe(self.keyframe)

    def undo(self) -> None:
        self.document.delete_morph_keyframe(
            self.keyframe.morph_name, self.keyframe.frame,
        )
        if self._previous is not None:
            self.document.insert_morph_keyframe(self._previous)


@dataclass
class InsertCameraKeyframe(Command):
    document: AnimationDocument
    keyframe: VmdCameraKeyframe
    _previous: VmdCameraKeyframe | None = field(default=None, init=False)
    _has_previous: bool = field(default=False, init=False)

    def execute(self) -> None:
        if not self._has_previous:
            for kf in self.document.camera_keyframes:
                if kf.frame == self.keyframe.frame:
                    self._previous = kf
                    break
            self._has_previous = True
        self.document.insert_camera_keyframe(self.keyframe)

    def undo(self) -> None:
        self.document.delete_camera_keyframe(self.keyframe.frame)
        if self._previous is not None:
            self.document.insert_camera_keyframe(self._previous)


@dataclass
class InsertLightKeyframe(Command):
    document: AnimationDocument
    keyframe: VmdLightKeyframe
    _previous: VmdLightKeyframe | None = field(default=None, init=False)
    _has_previous: bool = field(default=False, init=False)

    def execute(self) -> None:
        if not self._has_previous:
            for kf in self.document.light_keyframes:
                if kf.frame == self.keyframe.frame:
                    self._previous = kf
                    break
            self._has_previous = True
        self.document.insert_light_keyframe(self.keyframe)

    def undo(self) -> None:
        self.document.delete_light_keyframe(self.keyframe.frame)
        if self._previous is not None:
            self.document.insert_light_keyframe(self._previous)


@dataclass
class InsertSelfShadowKeyframe(Command):
    document: AnimationDocument
    keyframe: VmdSelfShadowKeyframe
    _previous: VmdSelfShadowKeyframe | None = field(default=None, init=False)
    _has_previous: bool = field(default=False, init=False)

    def execute(self) -> None:
        if not self._has_previous:
            for kf in self.document.self_shadow_keyframes:
                if kf.frame == self.keyframe.frame:
                    self._previous = kf
                    break
            self._has_previous = True
        self.document.insert_self_shadow_keyframe(self.keyframe)

    def undo(self) -> None:
        self.document.delete_self_shadow_keyframe(self.keyframe.frame)
        if self._previous is not None:
            self.document.insert_self_shadow_keyframe(self._previous)


@dataclass
class CompoundCommand(Command):
    """Run a tuple of commands as a single undo / redo step."""

    children: tuple[Command, ...]

    def execute(self) -> None:
        for child in self.children:
            child.execute()

    def undo(self) -> None:
        for child in reversed(self.children):
            child.undo()


@dataclass
class CommandStack:
    """Two-deque undo / redo manager.

    Pushing a new command clears the redo stack — the canonical "any
    fresh edit branches off the timeline" behaviour every editor ships.
    """

    _undo: list[Command] = field(default_factory=list)
    _redo: list[Command] = field(default_factory=list)

    def push(self, command: Command) -> None:
        command.execute()
        self._undo.append(command)
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> Command | None:
        if not self._undo:
            return None
        command = self._undo.pop()
        command.undo()
        self._redo.append(command)
        return command

    def redo(self) -> Command | None:
        if not self._redo:
            return None
        command = self._redo.pop()
        command.execute()
        self._undo.append(command)
        return command

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
