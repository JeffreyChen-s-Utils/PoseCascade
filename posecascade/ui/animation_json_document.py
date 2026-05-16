"""Shared in-memory model for declarative animation JSON editing.

Backs both the text-editor dock (MVP-1) and the phase-blocks dock
(MVP-2). A single :class:`AnimationJsonDocument` instance is wired to
both views in MVP-3 so an edit on either side updates the other through
the model's ``changed`` signal — no double-bookkeeping, no manual
re-serialise on every edit.

The model is intentionally small: it owns the raw dict that ``json.loads``
produced, plus a file path. Mutations are explicit methods (``set_text``,
``add_phase``, ``move_phase``, …) so the change-tracking surface stays
narrow; any view that mutates does so through these so the ``changed``
signal fires exactly once per operation. Subclassing for a richer
"named field" tree (e.g. typed phase objects) was considered and
deferred — the dict-of-dicts shape matches the JSON 1:1 and keeps the
schema as the source of truth.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from posecascade.utils.logging import get_logger

_log = get_logger(__name__)


class AnimationJsonDocument(QObject):
    """Observable dict-backed view of a declarative animation JSON.

    Signals:
        ``changed`` — emitted whenever the underlying document mutates,
        whether through ``set_text``, ``add_phase``, ``move_phase``,
        ``update_phase``, or ``remove_phase``. Views connect to refresh
        their rendering.
    """

    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._doc: dict[str, Any] = _empty_document()
        self._path: Path | None = None

    # ----- file IO ------------------------------------------------------

    def load_file(self, path: Path) -> bool:
        """Read ``path``, replace the document, emit ``changed``.

        Returns ``False`` (and logs) when the file is unreadable or
        produces invalid JSON — the model stays at its previous value
        so views don't end up with a partially-loaded buffer.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as err:
            _log.error("failed to read %s: %s", path, err)
            return False
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as err:
            _log.error("malformed JSON in %s: %s", path, err)
            return False
        if not isinstance(doc, dict):
            _log.error("animation root in %s is not a JSON object", path)
            return False
        self._doc = doc
        self._path = path
        self.changed.emit()
        return True

    def save_file(self) -> bool:
        """Write the current document back to its bound path.

        Returns ``False`` (and logs) when no path is bound or the write
        fails; the in-memory document is untouched in both cases.
        """
        if self._path is None:
            _log.error("save_file: no path bound to this document")
            return False
        try:
            self._path.write_text(self.text(), encoding="utf-8")
        except OSError as err:
            _log.error("failed to write %s: %s", self._path, err)
            return False
        return True

    # ----- raw access ---------------------------------------------------

    def text(self) -> str:
        """Pretty-printed JSON of the current document.

        Two-space indent (matches the bundled examples) and
        ``ensure_ascii=False`` so authored CJK strings stay legible.
        """
        return json.dumps(self._doc, indent=2, ensure_ascii=False) + "\n"

    def set_text(self, text: str) -> bool:
        """Parse ``text`` and replace the document if valid.

        Returns ``False`` (and leaves the model untouched) when the
        text doesn't parse. Callers that want to surface the parse
        error to a UI strip should validate first via ``json.loads``.
        """
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            return False
        if not isinstance(doc, dict):
            return False
        self._doc = doc
        self.changed.emit()
        return True

    def doc(self) -> dict[str, Any]:
        """The live dict. Callers MUST NOT mutate directly — use the methods."""
        return self._doc

    @property
    def path(self) -> Path | None:
        return self._path

    # ----- phase helpers ------------------------------------------------

    def phases(self) -> list[dict[str, Any]]:
        """Live phases list. Reading is safe; mutating bypasses ``changed``."""
        raw = self._doc.get("phases")
        return raw if isinstance(raw, list) else []

    def phase_count(self) -> int:
        return len(self.phases())

    def add_phase(self, phase: dict[str, Any] | None = None, at: int = -1) -> int:
        """Insert ``phase`` at index ``at`` (``-1`` → append). Returns the new index.

        ``phase=None`` inserts a minimal placeholder (``{"name": "new_phase",
        "duration_sec": 1.0}``) so first-time authors get a working block
        out of the gate.
        """
        if phase is None:
            phase = {"name": _next_phase_name(self.phases()), "duration_sec": 1.0}
        phases = self._doc.setdefault("phases", [])
        if at < 0 or at > len(phases):
            phases.append(phase)
            new_idx = len(phases) - 1
        else:
            phases.insert(at, phase)
            new_idx = at
        self.changed.emit()
        return new_idx

    def duplicate_phase(self, idx: int) -> int | None:
        """Deep-copy phase ``idx`` and insert it immediately after.

        The copy's ``name`` gets a ``_copy`` suffix so the duplicate is
        immediately distinguishable. Returns the new index, or ``None``
        if the source index is out of range.
        """
        phases = self.phases()
        if not 0 <= idx < len(phases):
            return None
        clone = copy.deepcopy(phases[idx])
        clone["name"] = f"{clone.get('name', 'phase')}_copy"
        return self.add_phase(clone, at=idx + 1)

    def remove_phase(self, idx: int) -> bool:
        """Delete phase ``idx``. Returns ``True`` on success."""
        phases = self.phases()
        if not 0 <= idx < len(phases):
            return False
        del phases[idx]
        self.changed.emit()
        return True

    def move_phase(self, from_idx: int, to_idx: int) -> bool:
        """Move phase from ``from_idx`` to ``to_idx``. Returns ``True`` on success.

        ``to_idx`` is the destination INDEX after removal — so moving 0
        to 2 in a 3-phase list lands the moved phase at position 2
        (last). Out-of-range or no-op moves return ``False``.
        """
        phases = self.phases()
        n = len(phases)
        if not 0 <= from_idx < n or not 0 <= to_idx < n or from_idx == to_idx:
            return False
        phase = phases.pop(from_idx)
        phases.insert(to_idx, phase)
        self.changed.emit()
        return True

    def update_phase_field(self, idx: int, key: str, value: Any) -> bool:
        """Set ``phases[idx][key] = value``. Returns ``True`` on success.

        Used by form-field edits in the phase-blocks dock. Removing a
        field is done by passing ``value=None`` for the optional keys
        (``pose``, ``hand_L``, ``hand_R``) — the runtime treats absence
        as "no preset" and the JSON looks cleaner without the empty
        entry. For required keys (``name``, ``duration_sec``) ``None``
        is rejected.
        """
        phases = self.phases()
        if not 0 <= idx < len(phases):
            return False
        phase = phases[idx]
        if value is None and key in _REQUIRED_PHASE_FIELDS:
            return False
        if value is None:
            phase.pop(key, None)
        else:
            phase[key] = value
        self.changed.emit()
        return True


# Phases require these to parse — refuse "delete" requests on them.
_REQUIRED_PHASE_FIELDS = frozenset({"name", "duration_sec"})


def _empty_document() -> dict[str, Any]:
    """Return a parseable minimal document so views never face ``None``."""
    return {"schema_version": 1, "phases": []}


def _next_phase_name(existing: list[dict[str, Any]]) -> str:
    """Pick ``"phase_<n>"`` where ``n`` keeps the list unique."""
    used = {p.get("name") for p in existing if isinstance(p.get("name"), str)}
    n = 1
    while f"phase_{n}" in used:
        n += 1
    return f"phase_{n}"


__all__ = ["AnimationJsonDocument"]
