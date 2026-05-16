"""JSON syntax highlighter for :class:`QPlainTextEdit`.

A focused subclass of :class:`QSyntaxHighlighter` that paints the five
token classes JSON cares about: keys, strings, numbers, literals
(``true`` / ``false`` / ``null``), and punctuation. Errors-as-you-type
land in a separate margin-mark layer (see ``AnimationJsonDock``) — this
class is purely about syntactic colouring and never re-parses.

The token regexes intentionally match per-line. JSON allows strings to
contain escaped quotes (``"with a \\" quote"``) but disallows raw
newlines inside strings, so a line-local scanner is sufficient and
avoids the multi-block state machine that a multi-line highlighter
would need.
"""
from __future__ import annotations

import re

from PySide6.QtGui import (
    QColor,
    QFont,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextDocument,
)

# Solarized-ish palette tuned for the dark editor background. Keys read
# in a warm blue so the document outline pops; strings in green; numbers
# and literals in muted cyan / orange so they don't pull the eye away
# from structure.
_KEY_COLOR = "#7aa6ff"          # blue
_STRING_COLOR = "#9ecf6d"       # green
_NUMBER_COLOR = "#d28b54"       # orange
_LITERAL_COLOR = "#c87aff"      # violet
_PUNCT_COLOR = "#9090a0"        # cool grey

# ``"..."`` followed by ``:`` is a key (account for escaped quotes
# inside the string body). Strings that don't precede a colon are
# values. We test for the key case first so the value-string regex
# doesn't over-match.
_KEY_RE = re.compile(r'"(?:[^"\\]|\\.)*"(?=\s*:)')
_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
# Numbers: integer / float / scientific, with optional leading sign.
_NUMBER_RE = re.compile(r"-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")
_LITERAL_RE = re.compile(r"\b(?:true|false|null)\b")
_PUNCT_RE = re.compile(r"[{}\[\],:]")


def _format(color: str, *, bold: bool = False) -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setForeground(QColor(color))
    if bold:
        fmt.setFontWeight(QFont.Weight.Bold)
    return fmt


class JsonHighlighter(QSyntaxHighlighter):
    """Per-line JSON syntax highlighter.

    ``QSyntaxHighlighter.highlightBlock`` runs once per visible line on
    layout; the per-line regexes keep that loop cheap even on a 1000-
    line animation document.
    """

    def __init__(self, document: QTextDocument | None = None) -> None:
        super().__init__(document)
        self._key_fmt = _format(_KEY_COLOR, bold=True)
        self._string_fmt = _format(_STRING_COLOR)
        self._number_fmt = _format(_NUMBER_COLOR)
        self._literal_fmt = _format(_LITERAL_COLOR, bold=True)
        self._punct_fmt = _format(_PUNCT_COLOR)

    def highlightBlock(self, text: str) -> None:  # noqa: N802 — Qt override
        # Track which character offsets are already coloured as a key
        # so the unconditional value-string pass below can skip them.
        consumed: set[int] = set()
        for match in _KEY_RE.finditer(text):
            self.setFormat(
                match.start(), match.end() - match.start(), self._key_fmt,
            )
            consumed.update(range(match.start(), match.end()))
        for match in _STRING_RE.finditer(text):
            if match.start() in consumed:
                continue
            self.setFormat(
                match.start(), match.end() - match.start(), self._string_fmt,
            )
        for match in _NUMBER_RE.finditer(text):
            # Numbers inside strings would be picked up by the regex —
            # the string passes paint over them, so order matters: we
            # paint numbers AFTER strings so string colour wins on
            # overlap. But we still need the no-overlap guard for the
            # rare case where a key like ``"42"`` matched: regex matches
            # the digits inside the string. The ``consumed`` set covers
            # keys; strings are caught by checking the existing format.
            existing = self.format(match.start())
            if existing.foreground().color() == QColor(_STRING_COLOR):
                continue
            if existing.foreground().color() == QColor(_KEY_COLOR):
                continue
            self.setFormat(
                match.start(), match.end() - match.start(), self._number_fmt,
            )
        for match in _LITERAL_RE.finditer(text):
            existing = self.format(match.start())
            if existing.foreground().color() == QColor(_STRING_COLOR):
                continue
            self.setFormat(
                match.start(), match.end() - match.start(), self._literal_fmt,
            )
        for match in _PUNCT_RE.finditer(text):
            self.setFormat(
                match.start(), match.end() - match.start(), self._punct_fmt,
            )


__all__ = ["JsonHighlighter"]
