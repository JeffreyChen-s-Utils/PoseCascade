"""Undo / redo command stack for animation document edits.

Wraps mutating operations on :class:`AnimationJsonDocument` so the user
gets a familiar Ctrl+Z / Ctrl+Y workflow across every edit surface (the
phase list buttons, the inline form, the timeline drag handles, the
JSON text editor). Each command captures the document's serialised
state BEFORE the change and applies the new state on redo; undo
restores the captured state.

The whole-document snapshot is a memory hog only on impractically
large authoring sessions (each snapshot of showcase.json is ~5 KB; a
1000-step session is ~5 MB), and dodges the bookkeeping that per-field
inverse-operations would need. We keep at most :data:`_MAX_HISTORY`
entries so an extremely long session doesn't grow unboundedly.

Public surface:

* :meth:`begin_transaction` / :meth:`end_transaction` — wrap multi-step
  user actions so undo collapses them into one step. The text-editor
  dock uses this around per-keystroke ``set_text`` calls (debounced) so
  one undo skips back to the previous canonical state, not to the
  previous character.
* :meth:`push_snapshot` — record the current document state under a
  human-readable label.
* :meth:`undo` / :meth:`redo` — pop the last / next snapshot.
* ``changed`` signal — fires whenever the can-undo / can-redo state
  flips, so UI buttons can toggle their enabled state.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from posecascade.ui.animation_json_document import AnimationJsonDocument
from posecascade.utils.logging import get_logger

_log = get_logger(__name__)

_MAX_HISTORY = 200


class AnimationCommandStack(QObject):
    """Document-snapshot command stack for declarative animations."""

    changed = Signal()

    def __init__(
        self,
        document: AnimationJsonDocument,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._document = document
        self._undo: list[tuple[str, str]] = []   # (label, serialised JSON)
        self._redo: list[tuple[str, str]] = []
        self._tx_depth = 0
        self._tx_label = ""
        self._tx_snapshot: str | None = None

    # ----- public API ---------------------------------------------------

    def begin_transaction(self, label: str) -> None:
        """Open a snapshot transaction; ``end_transaction`` commits it.

        Nested transactions collapse into the outermost — only one
        snapshot is recorded for the whole user action.
        """
        if self._tx_depth == 0:
            self._tx_label = label
            self._tx_snapshot = self._document.text()
        self._tx_depth += 1

    def end_transaction(self) -> None:
        if self._tx_depth == 0:
            return
        self._tx_depth -= 1
        if self._tx_depth > 0:
            return
        snapshot = self._tx_snapshot
        label = self._tx_label
        self._tx_snapshot = None
        self._tx_label = ""
        if snapshot is None or snapshot == self._document.text():
            # No-op transaction — don't pollute the undo stack with a
            # snapshot the user can't tell apart from the current state.
            return
        self._record(label, snapshot)

    def push_snapshot(self, label: str) -> None:
        """Record the current document state under ``label`` for undo.

        Use this OUTSIDE a transaction for one-shot mutations
        (insert phase, delete phase, …). Inside a transaction this
        falls through to ``begin_transaction`` semantics — only the
        outermost call's pre-state is what gets recorded on commit.
        """
        if self._tx_depth > 0:
            return
        snapshot = self._document.text()
        self._record(label, snapshot)

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        """Restore the last recorded snapshot. Returns ``True`` if anything happened."""
        if not self._undo:
            return False
        label, snapshot = self._undo.pop()
        current = self._document.text()
        self._document.set_text(snapshot)
        self._redo.append((label, current))
        _log.debug("undo: %s", label)
        self.changed.emit()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        label, snapshot = self._redo.pop()
        current = self._document.text()
        self._document.set_text(snapshot)
        self._undo.append((label, current))
        _log.debug("redo: %s", label)
        self.changed.emit()
        return True

    def clear(self) -> None:
        """Drop the whole history — used when loading a fresh file."""
        if not self._undo and not self._redo:
            return
        self._undo.clear()
        self._redo.clear()
        self.changed.emit()

    # ----- internals ----------------------------------------------------

    def _record(self, label: str, snapshot: str) -> None:
        # Duplicate-suppression: only skip if the new snapshot is
        # identical to the top of the undo stack (i.e. we just
        # recorded this exact state). Comparing to the document's
        # CURRENT text is wrong — snapshots are pre-mutation by
        # construction, so they match the live document at the
        # moment ``push_snapshot`` is called.
        if self._undo and self._undo[-1][1] == snapshot:
            return
        self._undo.append((label, snapshot))
        if len(self._undo) > _MAX_HISTORY:
            del self._undo[: len(self._undo) - _MAX_HISTORY]
        # A new edit invalidates the redo stack — standard convention.
        if self._redo:
            self._redo.clear()
        self.changed.emit()


__all__ = ["AnimationCommandStack"]
