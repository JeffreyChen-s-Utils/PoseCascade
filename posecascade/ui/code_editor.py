"""Reusable code-editor widget: ``QPlainTextEdit`` + line-number gutter.

Adapted from the Qt example pattern: a left-side gutter painted with
line numbers and an optional error mark on a flagged line. The gutter
is a child :class:`QWidget` that hands its paint event off to the host
editor so the editor controls colours and offsets.
"""
from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QResizeEvent
from PySide6.QtWidgets import QPlainTextEdit, QWidget

_GUTTER_PAD = 6
_GUTTER_BG = QColor(40, 42, 50)
_GUTTER_FG = QColor(120, 124, 140)
_ERROR_MARK_COLOR = QColor(244, 67, 54)   # matches the status strip red
_ERROR_LINE_BG = QColor(58, 28, 32)       # subtle row tint for the bad line


class _LineNumberArea(QWidget):
    """Gutter widget that proxies its paint to the host editor."""

    def __init__(self, editor: CodeEditor) -> None:
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self):  # noqa: N802 — Qt override
        from PySide6.QtCore import QSize  # noqa: PLC0415

        return QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 — Qt override
        self._editor._paint_line_number_area(event)  # noqa: SLF001 — sibling


class CodeEditor(QPlainTextEdit):
    """Plain text editor with a line-number gutter and an error-line mark.

    Public surface is small:

    * :meth:`set_error_line(lineno_1based, message)` paints a red mark
      on that gutter row and tints the row background; ``None`` clears.
    * The gutter width self-tunes from the current line count.

    Everything else is standard :class:`QPlainTextEdit` (read via
    ``setPlainText`` / ``toPlainText``, signals via ``textChanged``).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._gutter = _LineNumberArea(self)
        self._error_line: int | None = None
        self._error_message: str | None = None
        self.blockCountChanged.connect(self._update_gutter_width)
        self.updateRequest.connect(self._on_update_request)
        self._update_gutter_width()

    # ----- public API ---------------------------------------------------

    def set_error_line(self, lineno: int | None, message: str | None = None) -> None:
        """Flag a line as the source of the current parser error.

        ``lineno`` is 1-based to match :class:`json.JSONDecodeError`'s
        convention. Pass ``None`` to clear. The gutter repaints
        immediately so the mark shows on the next event cycle.
        """
        self._error_line = lineno
        self._error_message = message
        self.viewport().update()
        self._gutter.update()

    def error_line(self) -> int | None:
        return self._error_line

    # ----- gutter geometry / paint --------------------------------------

    def line_number_area_width(self) -> int:
        digits = max(2, len(str(max(1, self.blockCount()))))
        return _GUTTER_PAD * 2 + self.fontMetrics().horizontalAdvance("9") * digits

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 — Qt override
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._gutter.setGeometry(
            QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height()),
        )

    def _update_gutter_width(self) -> None:
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _on_update_request(self, rect: QRect, dy: int) -> None:
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_gutter_width()

    def _paint_line_number_area(self, event: QPaintEvent) -> None:
        painter = QPainter(self._gutter)
        painter.fillRect(event.rect(), _GUTTER_BG)
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = round(
            self.blockBoundingGeometry(block).translated(self.contentOffset()).top(),
        )
        bottom = top + round(self.blockBoundingRect(block).height())
        font_height = self.fontMetrics().height()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                lineno_1based = block_number + 1
                if self._error_line == lineno_1based:
                    # Row background tint on the editor side (not the
                    # gutter) — paint via a separate pass below.
                    pass
                painter.setPen(
                    _ERROR_MARK_COLOR if self._error_line == lineno_1based
                    else _GUTTER_FG,
                )
                painter.drawText(
                    0, top, self._gutter.width() - _GUTTER_PAD,
                    font_height, Qt.AlignmentFlag.AlignRight,
                    str(lineno_1based),
                )
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            block_number += 1

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 — Qt override
        # Paint the error-line tint UNDER the text — call super last so
        # the glyphs land on top of the background fill.
        if self._error_line is not None:
            painter = QPainter(self.viewport())
            block = self.document().findBlockByNumber(self._error_line - 1)
            if block.isValid():
                rect = self.blockBoundingGeometry(block).translated(
                    self.contentOffset(),
                ).toRect()
                # Extend to viewport width — the geometry's width tracks
                # the block's actual content, which makes the tint look
                # ragged on short lines.
                rect.setLeft(0)
                rect.setRight(self.viewport().width())
                painter.fillRect(rect, _ERROR_LINE_BG)
            painter.end()
        super().paintEvent(event)


__all__ = ["CodeEditor"]
