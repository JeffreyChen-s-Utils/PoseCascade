"""In-editor JSON editing dock for declarative animation files.

Provides MVP-1 of the animation editing roadmap: a Qt dock that lets the
user open a ``.json`` declarative-animation document, edit it in place,
get live schema + parser validation, save back to disk, and reload the
edited document into the running script host without quitting the
editor. The user no longer needs an external text editor + restart loop
to iterate on an animation.

Design choices
--------------

* **One ``QPlainTextEdit`` + one validation strip**. A full IDE-style
  experience (syntax tree, completion, multi-cursor) is out of scope for
  the MVP and would dwarf the rest of the editor. The plain editor is
  enough: JSON is small, errors point at line/column, and the validation
  strip surfaces parse failures inline.
* **Debounced validation**. The text-changed handler kicks a ``QTimer``
  with a short interval so a fast typist doesn't repeatedly re-parse the
  document mid-keystroke.
* **Save / reload are explicit**. Auto-save would surprise users who
  open a file just to skim. The two buttons are labelled ``Save`` and
  ``Reload``; Ctrl+S triggers Save without leaving the editor.
* **No syntax highlighting yet**. A ``QSyntaxHighlighter`` can drop in
  later — the dock's public API is small enough that adding a child
  highlighter won't be a structural change. Same logic applies to a
  future "outline tree" of phases on the side.
"""
from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from posecascade.scripting.declarative import (
    DeclarativeAnimationError,
    parse_animation,
    resolve_extends,
)
from posecascade.ui.animation_command_stack import AnimationCommandStack
from posecascade.ui.animation_json_document import AnimationJsonDocument
from posecascade.ui.code_editor import CodeEditor
from posecascade.ui.json_highlighter import JsonHighlighter
from posecascade.utils.logging import get_logger

_log = get_logger(__name__)

# Debounce window for live validation. Long enough to dodge per-keystroke
# re-parsing; short enough that the visible error message updates while
# the user is still looking at the change.
_VALIDATION_DEBOUNCE_MS = 250

_STATUS_OK_STYLE = "color: #4caf50;"               # green
_STATUS_ERROR_STYLE = "color: #f44336;"            # red
_STATUS_NEUTRAL_STYLE = "color: #909090;"          # grey — "no document loaded"
_TAB_WIDTH_SPACES = 2
_FORMAT_INDENT = 2


class AnimationJsonDock(QDockWidget):
    """Right-dock JSON editor + live validator for declarative animations.

    Signals:
        ``reload_requested(str)`` — emitted when the user clicks Reload
        or presses F5. The string payload is the current editor text;
        listeners (typically the bootstrap) should re-attach it through
        ``load_animation`` so the running runtime picks up the edit
        without a window restart.
    """

    reload_requested = Signal(str)

    def __init__(
        self,
        document: AnimationJsonDocument | None = None,
        command_stack: AnimationCommandStack | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Animation JSON", parent)
        # When no document is supplied the dock owns its own — keeps
        # MVP-1 callers working unchanged. MVP-3 wires the same
        # ``AnimationJsonDocument`` instance into both this dock and
        # the phase-blocks dock so edits propagate via ``changed``.
        self._document = document if document is not None else AnimationJsonDocument(self)
        self._stack = command_stack if command_stack is not None else AnimationCommandStack(
            self._document, self,
        )
        # Snapshot once per "typing session" instead of per character.
        # Cleared on every model-driven refresh; set to True on the
        # first user keystroke; reset on save / reload.
        self._snapshot_taken_this_session = False
        self._editor = CodeEditor()
        self._editor.setPlaceholderText(
            "Open a .json animation via File → Open Script, "
            "or paste JSON here.",
        )
        # Monospace font + tab → 2 spaces matches the bundled examples'
        # indent style; keeps copy-paste between an external editor and
        # this dock looking identical.
        mono = QFont("Consolas, Menlo, monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._editor.setFont(mono)
        self._editor.setTabStopDistance(
            self._editor.fontMetrics().horizontalAdvance(" ") * _TAB_WIDTH_SPACES,
        )
        self._highlighter = JsonHighlighter(self._editor.document())
        self._status = QLabel("No document loaded.")
        self._status.setStyleSheet(_STATUS_NEUTRAL_STYLE)
        self._save_btn = QPushButton("Save")
        self._format_btn = QPushButton("Format")
        self._reload_btn = QPushButton("Reload into runtime")
        self._path_label = QLabel("(no file)")
        self._validation_timer = QTimer(self)
        self._validation_timer.setSingleShot(True)
        self._validation_timer.setInterval(_VALIDATION_DEBOUNCE_MS)
        # Suppress the ``textChanged`` → ``set_text`` echo while the
        # dock refreshes itself from a ``changed`` signal it caused.
        self._echo_block = False
        # Dirty flag — set on every user-driven textChanged, cleared on
        # save / load / reload-from-document. The dock title shows a
        # trailing ``*`` so the author always knows whether the buffer
        # is in sync with disk.
        self._dirty = False
        self._build_ui()
        self._wire_signals()
        self._sync_from_document()

    # ----- public API ---------------------------------------------------

    @property
    def document(self) -> AnimationJsonDocument:
        return self._document

    def set_document(self, document: AnimationJsonDocument) -> None:
        """Swap the bound document (MVP-3 uses one shared instance)."""
        # First call against a model we just constructed has nothing
        # connected yet — silence the no-op disconnect across Qt's
        # RuntimeError and the TypeError some bindings raise.
        with contextlib.suppress(TypeError, RuntimeError):
            self._document.changed.disconnect(self._sync_from_document)
        self._document = document
        self._document.changed.connect(self._sync_from_document)
        self._sync_from_document()

    @property
    def path(self) -> Path | None:
        """Path of the currently-loaded document, or ``None`` if unsaved scratch."""
        return self._document.path

    def load_file(self, path: Path) -> bool:
        """Open ``path`` into the editor via the shared document.

        The document fires ``changed``, the text editor refreshes, and
        an initial validation pass runs so the user sees status before
        they type.
        """
        if not self._document.load_file(path):
            self._set_status(f"failed to read {path}", ok=False)
            return False
        # ``changed`` will have synced the editor already; just refresh
        # the path label and run the inline validator for the new text.
        self._path_label.setText(str(path))
        self._validate_now()
        return True

    def current_text(self) -> str:
        """Whatever's currently in the editor pane — what Save would write."""
        return self._editor.toPlainText()

    # ----- UI plumbing --------------------------------------------------

    def _build_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.addWidget(self._path_label, stretch=1)
        header.addWidget(self._format_btn)
        header.addWidget(self._save_btn)
        header.addWidget(self._reload_btn)
        layout.addLayout(header)

        layout.addWidget(self._editor, stretch=1)
        layout.addWidget(self._status)
        self.setWidget(container)

    def _wire_signals(self) -> None:
        self._editor.textChanged.connect(self._on_text_changed)
        self._validation_timer.timeout.connect(self._validate_now)
        self._save_btn.clicked.connect(self._on_save_clicked)
        self._format_btn.clicked.connect(self._on_format_clicked)
        self._reload_btn.clicked.connect(self._on_reload_clicked)
        self._document.changed.connect(self._sync_from_document)
        # Keyboard shortcut: Ctrl+S triggers save when the dock is
        # focused. We attach via QAction on the dock so the shortcut
        # works whether the user is in the editor or the buttons.
        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._on_save_clicked)
        self.addAction(save_action)
        # Ctrl+Z / Ctrl+Y over the shared command stack. The
        # QPlainTextEdit's built-in undo only walks per-character edits
        # which clashes badly with model-driven syncs — we suppress its
        # action and route through the stack instead.
        self._editor.setUndoRedoEnabled(False)
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self._stack.undo)
        self.addAction(undo_action)
        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self._stack.redo)
        self.addAction(redo_action)

    def _sync_from_document(self) -> None:
        """Refresh the editor pane from the bound document.

        Called when the document changes via *another* view (the
        phase-blocks dock). ``_echo_block`` keeps the ``textChanged``
        handler from immediately writing back into the same document
        on the round-trip. We also clear the dirty flag here: a
        document-driven sync represents the canonical state, so the
        buffer is by definition in sync with the model.
        """
        canonical = self._document.text()
        if self._editor.toPlainText() == canonical:
            return
        self._echo_block = True
        try:
            self._editor.blockSignals(True)
            try:
                self._editor.setPlainText(canonical)
            finally:
                self._editor.blockSignals(False)
            self._path_label.setText(
                str(self._document.path) if self._document.path else "(no file)",
            )
            self._set_dirty(False)
            # Document just synced through us — a new typing session
            # starts. The next keystroke will snapshot the current
            # canonical state as the undo target.
            self._snapshot_taken_this_session = False
            self._validate_now()
        finally:
            self._echo_block = False

    def _on_text_changed(self) -> None:
        # Debounce: restart the timer on every keystroke; it fires once
        # the user stops typing for ``_VALIDATION_DEBOUNCE_MS`` ms.
        # During a programmatic refresh from ``_sync_from_document`` we
        # skip propagation entirely.
        if self._echo_block:
            return
        if not self._snapshot_taken_this_session:
            # First user-driven edit since the buffer last matched the
            # document — snapshot the pre-edit state so Ctrl+Z lands at
            # the previous canonical buffer instead of the previous
            # character.
            self._stack.push_snapshot("edit JSON")
            self._snapshot_taken_this_session = True
        self._set_dirty(True)
        # Push the edit into the document so the other view sees it.
        # ``set_text`` returns False for unparseable buffers — we still
        # run validation so the inline strip reports the error.
        self._document.set_text(self._editor.toPlainText())
        self._validation_timer.start()

    def _on_format_clicked(self) -> None:
        """Pretty-print the buffer through ``json.dumps``.

        Only fires on a valid buffer — a malformed JSON document would
        either lose the user's in-progress edit or write garbage to
        disk; in that case we surface the error in the status strip
        and leave the text alone for the author to fix.
        """
        text = self._editor.toPlainText()
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as err:
            self._set_status(
                f"Format aborted: JSON parse error at line {err.lineno}",
                ok=False,
            )
            return
        formatted = json.dumps(doc, indent=_FORMAT_INDENT, ensure_ascii=False) + "\n"
        if formatted == text:
            self._set_status("Already canonical.", ok=True)
            return
        cursor = self._editor.textCursor()
        position = cursor.position()
        self._editor.setPlainText(formatted)
        # Best-effort cursor restore — clamp to the formatted length so
        # ``setPlainText`` on a shrinking buffer doesn't leave the
        # caret past the end.
        cursor.setPosition(min(position, len(formatted)))
        self._editor.setTextCursor(cursor)
        self._set_status("Formatted.", ok=True)

    def _on_save_clicked(self) -> None:
        if self._document.path is None:
            self._set_status("Save: no path bound; load a file first.", ok=False)
            return
        # Push the editor's exact text into the document first so the
        # save reflects what the user sees. If the text is malformed
        # JSON, ``set_text`` rejects it — fall back to writing the raw
        # text anyway since the author may want to save and fix later.
        text = self._editor.toPlainText()
        self._document.set_text(text)
        try:
            self._document.path.write_text(text, encoding="utf-8")
        except OSError as err:
            self._set_status(f"Save failed: {err}", ok=False)
            return
        self._set_dirty(False)
        self._set_status(f"Saved → {self._document.path}", ok=True)

    def _on_reload_clicked(self) -> None:
        """Emit ``reload_requested`` with the live text for the host to re-attach.

        Validation runs first; an unparseable document refuses to reload
        so the user sees the inline error and fixes it before the
        runtime sees anything broken.
        """
        errors = _validate(self._editor.toPlainText(), self._document.path)
        if errors:
            self._set_status(f"Reload blocked: {errors[0].message}", ok=False)
            return
        self.reload_requested.emit(self._editor.toPlainText())
        self._set_status("Reloaded into runtime.", ok=True)

    def _validate_now(self) -> None:
        text = self._editor.toPlainText()
        if not text.strip():
            self._editor.set_error_line(None)
            self._set_status("Empty document.", ok=False)
            return
        errors = _validate(text, self._document.path)
        if errors:
            err = errors[0]
            self._editor.set_error_line(err.line, err.message)
            self._set_status(err.message, ok=False)
        else:
            self._editor.set_error_line(None)
            self._set_status("OK — schema + parser passed.", ok=True)

    def _set_dirty(self, dirty: bool) -> None:
        """Mirror the buffer-vs-disk state into the dock title."""
        if self._dirty == dirty:
            return
        self._dirty = dirty
        base = "Animation JSON"
        self.setWindowTitle(f"{base} *" if dirty else base)

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def _set_status(self, text: str, *, ok: bool) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(
            _STATUS_OK_STYLE if ok else _STATUS_ERROR_STYLE,
        )


class _ValidationError:
    """One error from the JSON / extends / parser chain.

    Carries a line number when known (json.JSONDecodeError gives one
    naturally) so the editor can paint a margin mark on the offending
    row. Parser errors lack a line — the field is ``None`` there and
    the editor falls back to a status-strip-only display.
    """

    __slots__ = ("line", "message")

    def __init__(self, message: str, line: int | None = None) -> None:
        self.line = line
        self.message = message


def _validate(text: str, source_path: Path | None) -> list[_ValidationError]:
    """Run the JSON parse + extends merge + ``parse_animation`` chain.

    Returns a list of :class:`_ValidationError` records. Empty means the
    document loaded cleanly. The first error is the most actionable —
    surface that in the UI; the rest are kept for a future
    "show all errors" panel.
    """
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as err:
        return [_ValidationError(
            f"JSON parse error at line {err.lineno}: {err.msg}", line=err.lineno,
        )]
    try:
        merged = resolve_extends(doc, source_path.parent if source_path else None)
    except DeclarativeAnimationError as err:
        return [_ValidationError(f"extends: {err}")]
    try:
        parse_animation(merged)
    except DeclarativeAnimationError as err:
        return [_ValidationError(str(err))]
    return []


__all__ = ["AnimationJsonDock"]


# Re-exports so callers building dock chains can resolve helpers without
# importing the underlying declarative module directly. ``_validate`` is
# private but a few tests reach for it; the public API is the dock.
_ = Callable  # keep the import alive for downstream stubs that pick it up
